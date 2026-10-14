import { defineTool, type ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { appendEfeDebug } from "./efe.ts";
import { truncateOutput } from "./paths.ts";
import { ASSESS_STEP_DESCRIPTION, PROPOSE_SEQUENCES_DESCRIPTION } from "./prompts.ts";
import type { SelectedSequence, SequenceCandidate, VfeState } from "./state.ts";
import { collectWorldState, inferConsumed, renderWorldState, validateCandidate } from "./world.ts";

export const PROPOSE_SEQUENCES_TOOL = "propose_sequences";
export const ASSESS_STEP_TOOL = "assess_step";

const PARAMS = Type.Object({
	candidates: Type.Array(
		Type.Object({
			label: Type.String({ description: "Short name for this candidate approach" }),
			commands: Type.Array(Type.String(), {
				description: "1-8 shell commands to run in order as this candidate sequence",
			}),
			assumes: Type.Optional(
				Type.Array(Type.String(), {
					description:
						"Preconditions this candidate needs before running (absolute path, 'bin:<name>', or 'port:<n>'). Omit if self-contained.",
				}),
			),
			provides: Type.Optional(
				Type.Array(Type.String(), {
					description: "Artifacts this candidate creates (absolute paths).",
				}),
			),
		}),
		{ description: "Exactly 2 or 3 candidate sequences to choose from", minItems: 2, maxItems: 3 },
	),
	selected: Type.Union([Type.Literal("c1"), Type.Literal("c2"), Type.Literal("c3")], {
		description: "The candidate YOU choose to execute; it must exist in candidates",
	}),
	why: Type.Optional(
		Type.String({
			description:
				"<=240 chars: your trade-off across expected effect, exploration, and deviation from validated corrections",
		}),
	),
});

const ASSESS_PARAMS = Type.Object({
	verdict: Type.Union([Type.Literal("better"), Type.Literal("worse"), Type.Literal("same")], {
		description:
			"Your assessment of THIS step after seeing its result. Fact lines from actual test results (REGRESSION/IMPROVEMENT) take precedence over this.",
	}),
	evidence: Type.String({
		description: "<=160 chars: the concrete content backing your verdict (numbers, test names, observed diffs)",
	}),
});

type Exec = (cmd: string, timeoutSec?: number) => Promise<{ output: string; exitCode: number; infra?: boolean }>;

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

/**
 * Step entry: model submits 2-3 candidates AND picks one with `selected`;
 * only the chosen candidate's commands run, the rest are discarded.
 */
export function registerProposeSequences(pi: ExtensionAPI, state: VfeState, exec: Exec): void {
	pi.registerTool(
		defineTool({
			name: PROPOSE_SEQUENCES_TOOL,
			label: "propose_sequences",
			description: PROPOSE_SEQUENCES_DESCRIPTION,
			promptSnippet: "Submit 2-3 candidate sequences and pick one to run",
			parameters: PARAMS,
			async execute(_id, params, _signal, _onUpdate, _ctx) {
				if (state.phase !== "step") {
					return fail(`propose_sequences only valid in step phase (current: ${state.phase})`);
				}
				if (state.selectedSequence) {
					return fail(
						`A sequence was already selected this step (${state.selectedSequence.label}). Observe/refine and end the turn.`,
					);
				}
				const selectedId = String(params.selected ?? "").trim();
				if (!/^c[1-3]$/.test(selectedId)) {
					return fail(`selected must be "c1", "c2" or "c3"; got "${selectedId}".`);
				}
				const raw = params.candidates ?? [];
				if (raw.length < 2 || raw.length > 3) {
					return fail("candidates must be exactly 2 or 3");
				}
				const candidates: SequenceCandidate[] = [];
				for (let i = 0; i < raw.length; i++) {
					const label = String(raw[i]?.label ?? "").trim() || `cand-${i + 1}`;
					const commands = (raw[i]?.commands ?? [])
						.map(String)
						.map((c) => c.trim())
						.filter(Boolean);
					if (commands.length === 0) return fail(`candidate "${label}" has no commands`);
					if (commands.length > 8) {
						return fail(`candidate "${label}" has ${commands.length} commands; max 8 per sequence`);
					}
					const assumes = (raw[i]?.assumes ?? [])
						.map(String)
						.map((s) => s.trim())
						.filter(Boolean);
					const provides = (raw[i]?.provides ?? [])
						.map(String)
						.map((s) => s.trim())
						.filter(Boolean);
					candidates.push({
						id: `c${i + 1}`,
						label,
						commands,
						...(assumes.length > 0 ? { assumes } : {}),
						...(provides.length > 0 ? { provides } : {}),
					});
				}
				// Deterministic grounding: snapshot the container, then drop candidates whose
				// declared/inferred preconditions are absent. Keeps the multi-candidate design
				// but never lets an infeasible branch reach selection or execution.
				const requiredPaths = [
					...new Set(
						candidates.flatMap((c) => [
							...(c.assumes ?? []).filter((s) => s.startsWith("/")),
							...inferConsumed(c.commands),
						]),
					),
				].slice(0, 24);
				const world = await collectWorldState(exec, requiredPaths);
				state.worldState = world;
				const feasibilities = candidates.map((c) => validateCandidate(c, world));
				const feasible = candidates.filter((_c, i) => feasibilities[i].feasible);
				const dropped = feasibilities.filter((f) => !f.feasible);
				for (const d of dropped) {
					appendEfeDebug({
						phase: "step",
						step: state.steps,
						tool: PROPOSE_SEQUENCES_TOOL,
						action: `candidate ${d.id} (${d.label})`,
						decision: "infeasible",
						reasons: d.reasons,
						objective: state.belief?.objective ?? state.objective ?? "",
					});
				}
				if (feasible.length === 0) {
					return fail(
						[
							"All candidates are infeasible against the current container state:",
							...dropped.map((d) => `- ${d.id} (${d.label}): ${d.reasons.join("; ")}`),
							"",
							renderWorldState(world),
							"",
							"Re-propose 2-3 sequences that are each executable from THIS state. If a candidate runs an artifact, include the command that creates it in the SAME candidate (or declare assumes/provides).",
						].join("\n"),
					);
				}
				const chosen = feasible.find((c) => c.id === selectedId);
				if (!chosen) {
					const isDropped = dropped.find((d) => d.id === selectedId);
					if (isDropped) {
						return fail(
							[
								`selected=${isDropped.id} (${isDropped.label}) is infeasible: ${isDropped.reasons.join("; ")}`,
								"",
								renderWorldState(world),
								"",
								"Re-propose 2-3 sequences and select a feasible one.",
							].join("\n"),
						);
					}
					return fail(
						`selected must be one of the candidate ids you submitted (${candidates
							.map((c) => c.id)
							.join(", ")}); got "${selectedId}".`,
					);
				}
				const why = String(params.why ?? "")
					.trim()
					.slice(0, 240);
				const selection: SelectedSequence = {
					id: chosen.id,
					label: chosen.label,
					commands: chosen.commands,
					...(why ? { why } : {}),
				};
				state.selectedSequence = selection;
				appendEfeDebug({
					phase: "step",
					step: state.steps,
					tool: PROPOSE_SEQUENCES_TOOL,
					action: `AGENT SELECTED ${selection.id} (${selection.label}): ${selection.commands.join(" && ").slice(0, 800)}`,
					decision: "agent_select",
					selected: selection.id,
					why,
					candidates: candidates.map((c) => ({ id: c.id, label: c.label })),
					dropped: dropped.map((d) => d.id),
					objective: state.belief?.objective ?? state.objective ?? "",
				});

				const parts: string[] = [
					`You selected ${selection.id} (${selection.label}).`,
					"Other candidates were NOT executed.",
				];
				if (dropped.length > 0) {
					parts.push(
						`Infeasible (dropped before selection): ${dropped
							.map((d) => `${d.id}(${d.reasons.slice(0, 2).join("; ")})`)
							.join(" | ")}`,
					);
				}
				parts.push("", "Executing your choice:");
				let lastExit = 0;
				// Chosen sequences are long-running by design (VM boots, apt, dataset fetches);
				// the 120s sidecar default killed them mid-flight. VFE_SEQ_TIMEOUT overrides.
				const seqTimeoutEnv = Number(process.env.VFE_SEQ_TIMEOUT ?? 0);
				const seqTimeout = Number.isFinite(seqTimeoutEnv) && seqTimeoutEnv > 0 ? Math.floor(seqTimeoutEnv) : 600;
				for (const cmd of selection.commands) {
					let output = "";
					let exitCode = 1;
					let infra = false;
					try {
						const r = await exec(cmd, seqTimeout);
						output = r.output;
						exitCode = r.exitCode;
						infra = r.infra === true;
					} catch (err) {
						output = err instanceof Error ? err.message : String(err);
						exitCode = 1;
					}
					parts.push(`$ ${cmd}\n${output}(exit ${exitCode})`);
					lastExit = exitCode;
					state.turnTools.push({
						tool: "run",
						input: JSON.stringify({ commands: [cmd] }),
						output: output.slice(0, 4000),
						isError: exitCode !== 0,
						exitCode,
						...(infra ? { infra: true } : {}),
					});
					if (state.turnTools.length > 40) state.turnTools.splice(0, state.turnTools.length - 40);
					if (infra) {
						parts.push(`[INFRA FAULT] container unavailable; stopping sequence.`);
						break;
					}
					if (exitCode !== 0) {
						parts.push(`[stopped at failing command; exit ${exitCode}]`);
						break;
					}
				}
				return ok(truncateOutput(parts.join("\n")), {
					chosen: selection.id,
					label: selection.label,
					commands: selection.commands,
					why: selection.why ?? "",
					exitCode: lastExit,
				});
			},
		}),
	);
}

/**
 * Closing call of a step turn: the agent's self-assessment of THIS step,
 * after it has run a sequence and observed the result. Settle consumes it
 * the same turn, so verdict and settle facts always refer to the same step.
 * Omitted → settle falls back to a facts-only (neutral) signal.
 */
export function registerAssessStep(pi: ExtensionAPI, state: VfeState): void {
	pi.registerTool(
		defineTool({
			name: ASSESS_STEP_TOOL,
			label: "assess_step",
			description: ASSESS_STEP_DESCRIPTION,
			promptSnippet: "Assess this step (better/worse/same) before ending the turn",
			parameters: ASSESS_PARAMS,
			async execute(_id, params, _signal, _onUpdate, _ctx) {
				if (state.phase !== "step") {
					return fail(`assess_step only valid in step phase (current: ${state.phase})`);
				}
				if (!state.selectedSequence) {
					return fail(
						"Run your chosen sequence (propose_sequences) and observe its result before assessing the step.",
					);
				}
				const v = String(params.verdict ?? "");
				if (v !== "better" && v !== "worse" && v !== "same") {
					return fail(`verdict must be better, worse or same; got "${v}"`);
				}
				state.turnVerdict = v;
				state.turnVerdictEvidence = String(params.evidence ?? "")
					.trim()
					.slice(0, 240);
				return ok(
					`Assessment recorded: ${v}. Settle will use it as your self-assessed outcome for this step (fact lines from actual test results take precedence).`,
					{ verdict: v, evidence: state.turnVerdictEvidence },
				);
			},
		}),
	);
}
