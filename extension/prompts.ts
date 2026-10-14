import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { envObserveBudget } from "./state.ts";

const EXTENSION_DIR = dirname(fileURLToPath(import.meta.url));

/**
 * Prompt files live in a `prompts/` directory next to this extension file
 * (resolved from import.meta.url, never from cwd). Override with VFE_PROMPTS_DIR.
 */
export function promptsDir(): string {
	return process.env.VFE_PROMPTS_DIR?.trim() || join(EXTENSION_DIR, "..", "prompts");
}

function loadPrompt(name: string, replacements?: Record<string, string>): string {
	let text = readFileSync(join(promptsDir(), `${name}.md`), "utf8").trim();
	if (replacements) {
		for (const [key, value] of Object.entries(replacements)) {
			text = text.split(`{{${key}}}`).join(value);
		}
	}
	return text;
}

export const SKILL_FORMAT = loadPrompt("skill_format");
export const STABLE_SKILL_SYSTEM = loadPrompt("stable_skill_system", { SKILL_FORMAT });
export const ADAPTIVE_SKILL_SYSTEM = loadPrompt("adaptive_skill_system", { SKILL_FORMAT });
export const GROUND_SKILL_SYSTEM = loadPrompt("ground_skill_system");
export const PROPOSE_PLAN_DESCRIPTION = loadPrompt("propose_plan_description").split("\n").join(" ");
export const PROPOSE_SEQUENCES_DESCRIPTION = loadPrompt("propose_sequences_description", {
	OBSERVE_BUDGET: String(envObserveBudget()),
}).split("\n").join(" ");
export const ASSESS_STEP_DESCRIPTION = loadPrompt("assess_step_description").split("\n").join(" ");
export const VFE_SYSTEM_PROMPT = loadPrompt("vfe_system_prompt", { OBSERVE_BUDGET: String(envObserveBudget()) });
export const REVISE_STATES_SYSTEM = loadPrompt("revise_states_system");
export const REFLECT_SYSTEM = loadPrompt("reflect_system");

/** Top-phase user message: task + recon evidence + experience render. */
export function buildTopPrompt(input: {
	instruction: string;
	reconEvidence: string;
	experienceText: string;
	envObs?: string;
	envConstraints?: Array<{ title: string; text: string }>;
}): string {
	const parts = [
		"TASK (full instruction — TOP planning layer; ONLY tool this phase is propose_plan; do NOT call bash/read/ls or any other tool):",
		input.instruction.trim(),
	];
	if (input.reconEvidence.trim()) {
		parts.push(
			"ENVIRONMENT EVIDENCE (actually read from the task environment — GROUND TRUTH; ground methodology in it and NEVER assert anything it contradicts):",
			input.reconEvidence.trim().slice(0, 14000),
		);
	}
	if (input.envConstraints && input.envConstraints.length > 0) {
		parts.push(
			`ENV CONSTRAINTS (static facts the agent must respect):\n${input.envConstraints
				.map((c) => `- [${c.title}] ${c.text}`)
				.join("\n")}`,
		);
	}
	parts.push(input.experienceText.trim() || "(no prior experience)");
	if (input.envObs?.trim()) {
		parts.push(`ENV (recon):\n${input.envObs.trim().slice(0, 600)}`);
	}
	parts.push(
		"Produce the adaptive skill (Markdown only) and call propose_plan with {todos, objective, target_files, skill_summary, skill}. Your next tool call must be propose_plan — nothing else exists in this phase.",
	);
	return parts.join("\n\n");
}

export function groundUserPrompt(evidence: string, skillMd: string): string {
	return `EVIDENCE:\n${evidence.slice(0, 12000)}\n\nSKILL:\n${skillMd.slice(0, 12000)}`;
}

function skillCorrectionsBlock(corr: string[] | undefined): string {
	const list = (corr ?? []).filter(Boolean);
	if (list.length === 0) {
		return "VALIDATED CORRECTIONS: none yet. The skill above is method reference only; deviating from it costs nothing.";
	}
	return [
		"VALIDATED CORRECTIONS (skill claims contradicted by evidence — do NOT follow the skill on these points):",
		...list.slice(0, 8).map((c) => `- ${c}`),
	].join("\n");
}

/** Static container paths the agent can open for ground truth each step. */
const AVAILABLE_FILES = [
	"AVAILABLE FILES (open these when the feedback references them):",
	"- /tmp/vfe_last_pytest.out — last pytest full output (host-mirrored as last_pytest_full.out in the run dir)",
	"- /tmp/vfe_local_memory.json — objective, todos, env constraints, recent steps",
	"- /tmp/vfe_diary.txt — recent step commands/outcomes",
	"- /tmp/vfe_objectives.txt / /tmp/vfe_failures.txt / /tmp/vfe_adaptive_skill.md — plan, hard failures, skill",
].join("\n");

function requiredOutputsBlock(required: string[] | undefined): string {
	const list = (required ?? []).filter(Boolean);
	if (list.length === 0) return "";
	return ["REQUIRED OUTPUTS (still missing — produce these next):", ...list.slice(0, 12).map((p) => `- ${p}`)].join(
		"\n",
	);
}

function failureHistoryBlock(history: string[] | undefined): string {
	const list = (history ?? []).filter(Boolean);
	if (list.length === 0) return "";
	return ["FAILURE HISTORY (recent settles):", ...list.slice(-5).map((h) => `- ${h}`)].join("\n");
}

function practiceHitsBlock(hits: Array<{ name: string; steps: string[] }> | undefined): string {
	const list = hits ?? [];
	if (list.length === 0) return "";
	const lines: string[] = ["MATURE PRACTICES (validated global sequences — prefer these moves):"];
	for (const h of list) {
		lines.push(`- ${h.name}`);
		for (const s of h.steps.slice(0, 8)) lines.push(`    ${s}`);
	}
	return lines.join("\n");
}

export function buildFirstStepPrompt(input: {
	objective: string;
	targetFiles: string[];
	skillSummary: string;
	maxSteps: number;
	skillCorrections?: string[];
	practiceHits?: Array<{ name: string; steps: string[] }>;
	requiredOutputs?: string[];
	worldState?: string;
}): string {
	const files = input.targetFiles.length > 0 ? input.targetFiles.join(", ") : "(see objective)";
	return [
		"PHASE=step. One agent turn = one step.",
		`CURRENT OBJECTIVE:\n${input.objective}`,
		`Deliverable(s): ${files}`,
		input.worldState ? input.worldState : "",
		input.skillSummary ? `APPLICABLE SKILL:\n${input.skillSummary.slice(0, 3000)}` : "",
		skillCorrectionsBlock(input.skillCorrections),
		requiredOutputsBlock(input.requiredOutputs),
		practiceHitsBlock(input.practiceHits),
		AVAILABLE_FILES,
		`Rules for this step (max ${input.maxSteps} steps):`,
		"1. FIRST call propose_sequences with 2-3 candidate command sequences (each 1-8 shell commands) and pick ONE with `selected`. Weigh, in this order: expected effect (changes the tested artifact or an observable test result and advances the objective) > exploration (only when information is missing) > deviation (judge only against skill_corrections, never the original skill). Each candidate MUST be executable from the WORLD STATE above (if it runs a file, create that file in the same candidate); infeasible candidates are dropped.",
		`2. After your chosen sequence runs, use at most ${envObserveBudget()} more calls: read/ls/find/grep/run_test to observe, plus write/edit/run to refine. Do NOT propose another sequence this turn.`,
		"3. Do NOT re-read files already shown; do NOT repeat a fix that already failed.",
		`4. End the turn as soon as observation is enough (target: within ${envObserveBudget()} observes); BEFORE ending, call assess_step {verdict: better|worse|same, evidence} — your judgment of this step after seeing its result. settle/feedback will be injected automatically. Do not keep tool-calling.`,
	]
		.filter(Boolean)
		.join("\n\n");
}

function assertionBlock(input: { assertionText?: string; assertionDelta?: number; assertionStreak: number }): string {
	if (!input.assertionText) return "";
	const delta =
		input.assertionDelta !== undefined
			? `Δ vs previous settle: ${input.assertionDelta > 0 ? "+" : ""}${input.assertionDelta}`
			: "Δ vs previous settle: n/a";
	return `ASSERTION: ${input.assertionText} | ${delta} | identical ×${input.assertionStreak}`;
}

export function buildFeedbackPrompt(input: {
	step: number;
	maxSteps: number;
	signal: string;
	summary: string;
	note?: string;
	objective: string;
	targetFiles: string[];
	fails: string[];
	passed: number;
	failed: number;
	improved: boolean;
	assertionText?: string;
	assertionDelta?: number;
	assertionStreak: number;
	stall: number;
	hint?: string;
	skillCorrections?: string[];
	requiredOutputs?: string[];
	failureHistory?: string[];
	practiceHits?: Array<{ name: string; steps: string[] }>;
	worldState?: string;
}): string {
	const parts = [
		`STEP ${input.step}/${input.maxSteps} settle feedback.`,
		`STEP OUTCOME: ${input.signal}`,
		input.summary ? `LAST-STEP REVIEW:\n${input.summary}` : "",
		input.note?.trim() ? input.note.trim() : "",
		`tests: passed=${input.passed} failed=${input.failed}` +
			(input.fails.length > 0 ? ` FAILED=[${input.fails.slice(0, 8).join(", ")}]` : ""),
		assertionBlock(input),
		input.worldState ? input.worldState : "",
		`CURRENT OBJECTIVE: ${input.objective}`,
		input.targetFiles.length > 0 ? `target_files: ${input.targetFiles.join(", ")}` : "",
		skillCorrectionsBlock(input.skillCorrections),
		requiredOutputsBlock(input.requiredOutputs),
		failureHistoryBlock(input.failureHistory),
		practiceHitsBlock(input.practiceHits),
		AVAILABLE_FILES,
		input.stall >= 3
			? `STALL: ${input.stall} consecutive step(s) you self-assessed as same/worse. Do NOT wholesale-rewrite: first diff your current artifact against your best-known version (or the reference), localize the single divergence that dominates the score, and fix it with a one-variable edit. Switch approach only if localization shows the path itself is wrong.`
			: "",
		input.hint ?? "",
		"Next step: FIRST call propose_sequences with 2-3 candidate command sequences and pick ONE with `selected` (weigh expected effect > exploration > deviation from skill_corrections). Each candidate MUST be executable from the WORLD STATE above (if it runs a file, create that file in the same candidate); infeasible candidates are dropped. Your chosen sequence runs automatically; then use at most " +
			envObserveBudget() +
			" further calls (read/ls/find/grep/run_test to observe, write/edit/run to refine); BEFORE ending the turn, call assess_step {verdict: better|worse|same, evidence} — your judgment of this step after seeing its result (facts from the test list override your verdict). One turn = one step.",
	];
	return parts.filter(Boolean).join("\n\n");
}

export function reviseUserPrompt(input: {
	skillMd: string;
	todosJson: string;
	objective: string;
	evidence: string;
	fails: string[];
	fileState: Record<string, string>;
}): string {
	return [
		`SKILL (methodology — prior constraint):\n${input.skillMd.slice(0, 16000) || "(no skill)"}`,
		`CURRENT TODO LIST:\n${input.todosJson}`,
		`CURRENT OBJECTIVE (in_progress):\n${input.objective}`,
		`TEST FAILURES: ${JSON.stringify(input.fails).slice(0, 600)}`,
		`FILE STATE: ${JSON.stringify(input.fileState).slice(0, 600)}`,
		`EVIDENCE (predicted vs actual of executed candidates):\n${input.evidence}`,
	].join("\n");
}

export function reflectUserPrompt(input: {
	predicted: string;
	actual: string;
	observation: string;
	resultLabel: string;
	verdictEvidence?: string;
	newFails: string[];
	fixedFails: string[];
}): string {
	const extra =
		input.newFails.length > 0 || input.fixedFails.length > 0
			? ` REGRESSION=${JSON.stringify(input.newFails)}; FIXED=${JSON.stringify(input.fixedFails)}`
			: "";
	const evidence = input.verdictEvidence ? ` | 自评依据: ${input.verdictEvidence}` : "";
	return `PREDICTED:\n${input.predicted}\nACTUAL:\n${input.actual}\nOBSERVATION:\n${input.observation}\n本步结果: ${input.resultLabel}${evidence}${extra}`;
}
