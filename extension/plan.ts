import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { defineTool, type ExtensionAPI, type ExtensionContext } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { completeText } from "./llm.ts";
import { maturePracticeHits } from "./memory.ts";
import { buildFirstStepPrompt, GROUND_SKILL_SYSTEM, groundUserPrompt, PROPOSE_PLAN_DESCRIPTION } from "./prompts.ts";
import { envMaxSteps, normTodos, resetStepRuntime, runDirFor, type VfeBelief, type VfeState } from "./state.ts";

export const PROPOSE_PLAN_TOOL = "propose_plan";

const PARAMS = Type.Object({
	todos: Type.Array(
		Type.Object({
			task: Type.String({ description: "Shell-actionable subtask" }),
			status: Type.Union([Type.Literal("pending"), Type.Literal("in_progress"), Type.Literal("completed")]),
		}),
		{ description: "Full ordered todo list; exactly one in_progress (the first)" },
	),
	objective: Type.String({ description: "Must equal the in_progress todo task" }),
	target_files: Type.Array(Type.String(), {
		description: "Absolute /app paths the current objective must create/modify",
	}),
	skill_summary: Type.String({ description: "Skill parts for the current objective" }),
	skill: Type.String({ description: "Full adaptive skill Markdown" }),
	stable_skill: Type.Optional(Type.String({ description: "Optional stable skill Markdown" })),
	skill_corrections: Type.Optional(
		Type.Array(Type.String(), { description: "Skill claims contradicted by evidence" }),
	),
});

function ok(text: string, details: Record<string, unknown>) {
	return {
		content: [{ type: "text" as const, text }],
		details,
	};
}

function fail(text: string) {
	return {
		content: [{ type: "text" as const, text }],
		details: { error: true },
	};
}

function writeArtifacts(state: VfeState, belief: VfeBelief, adaptive: string, stable?: string): string {
	if (!state.task) throw new Error("No task set");
	const dir = runDirFor(state.task, state.rep ?? 0);
	if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
	const w = (name: string, body: string) => writeFileSync(join(dir, name), body, "utf8");
	w("adaptive.md", adaptive);
	if (stable) w("stable.md", stable);
	w(
		"init_state.json",
		JSON.stringify(
			{
				objective: belief.objective,
				todos: belief.todos,
				target_files: belief.target_files,
				history: belief.history,
				skill_summary: belief.skill_summary,
				skill_corrections: belief.skill_corrections,
			},
			null,
			2,
		),
	);
	return dir;
}

/**
 * Top-phase belief write: validate plan, ground adaptive skill against recon
 * evidence (nested complete; falls back to raw skill), persist artifacts, flip to step.
 */
export function registerProposePlan(pi: ExtensionAPI, state: VfeState, containerTools: readonly string[]): void {
	pi.registerTool(
		defineTool({
			name: PROPOSE_PLAN_TOOL,
			label: "propose_plan",
			description: PROPOSE_PLAN_DESCRIPTION,
			promptSnippet: "Submit top-layer plan (skill + todos + objective)",
			parameters: PARAMS,
			async execute(_id, params, _signal, _onUpdate, ctx: ExtensionContext) {
				if (state.phase !== "top") {
					return fail(`propose_plan only valid in top phase (current: ${state.phase})`);
				}
				const todos = normTodos(params.todos);
				if (todos.length === 0) return fail("todos must be non-empty");
				let objective = params.objective.trim();
				const ip = todos.find((t) => t.status === "in_progress");
				if (ip) objective = ip.task;
				else if (!objective && todos.length > 0) {
					todos[0].status = "in_progress";
					objective = todos[0].task;
				}
				const targetFiles = params.target_files.map(String).filter((p) => p.startsWith("/"));
				if (!objective) return fail("missing objective");
				let adaptive = params.skill.trim();
				if (!adaptive) return fail("skill must be non-empty");
				const evidence = state.reconEvidence ?? "";
				if (evidence.trim()) {
					const grounded = await completeText(
						ctx,
						GROUND_SKILL_SYSTEM,
						groundUserPrompt(evidence, adaptive),
						4000,
					);
					if (grounded) adaptive = grounded;
				}
				const belief: VfeBelief = {
					objective,
					todos,
					target_files: targetFiles,
					history: [],
					skill_summary: params.skill_summary.trim(),
					skill_corrections: (params.skill_corrections ?? []).map((c) => c.trim()).filter(Boolean),
				};
				const stable = params.stable_skill?.trim();
				let dir = "";
				try {
					dir = writeArtifacts(state, belief, adaptive, stable);
				} catch (err) {
					return fail(`artifact write failed: ${err instanceof Error ? err.message : String(err)}`);
				}
				state.belief = belief;
				state.objective = belief.objective;
				state.adaptiveSkill = adaptive;
				if (stable) state.stableSkill = stable;
				resetStepRuntime(state);
				state.phase = "step";
				state.selectedSequence = undefined;
				pi.setActiveTools([...containerTools]);
				// followUp from a mid-run tool drains in the same agent run (no second
				// agent_start), so arm settle here; agent_end will settle this turn.
				state.awaitingStepTurn = false;
				state.pendingSettle = true;
				state.turnTools = [];
				const practiceQuery = `${objective} ${targetFiles.join(" ")}`;
				const practiceHits = state.store ? maturePracticeHits(state.store, practiceQuery) : [];
				const firstPrompt = buildFirstStepPrompt({
					objective,
					targetFiles,
					skillSummary: belief.skill_summary,
					maxSteps: envMaxSteps(),
					skillCorrections: belief.skill_corrections,
					practiceHits,
				});
				return ok(
					[
						`Plan accepted. objective=${objective}`,
						`todos=${todos.length} target_files=${targetFiles.length}`,
						`artifacts=${dir}`,
						`phase=step tools=${containerTools.join(" ")}`,
						"",
						firstPrompt,
					].join("\n"),
					{ objective, todos: todos.length, target_files: targetFiles, dir, phase: "step" },
				);
			},
		}),
	);
}
