import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { DockerClient } from "./docker.ts";
import { appendEfeDebug } from "./efe.ts";
import { dumpInjected, loadStore, renderAllExperiences } from "./memory.ts";
import { PROPOSE_PLAN_TOOL, registerProposePlan } from "./plan.ts";
import { buildTopPrompt, VFE_SYSTEM_PROMPT } from "./prompts.ts";
import { probeEnv, probeEnvConstraintsWith, reconEvidence } from "./recon.ts";
import { ASSESS_STEP_TOOL, PROPOSE_SEQUENCES_TOOL, registerAssessStep, registerProposeSequences } from "./sequence.ts";
import { settleStep, type VerifySandbox } from "./settle.ts";
import { createState, envMaxSteps, envObserveBudget, READONLY_GATE, resetStepRuntime } from "./state.ts";
import { registerContainerTools } from "./tools.ts";

const CONTAINER_TOOLS = ["read", "write", "edit", "grep", "find", "ls", "run", "run_test"] as const;
const STEP_TOOLS = [...CONTAINER_TOOLS, PROPOSE_SEQUENCES_TOOL, ASSESS_STEP_TOOL];
const TOP_TOOLS = ["propose_plan"] as const;
const OBSERVE_TOOLS = new Set(["read", "ls", "find", "grep", "run_test", "write", "edit", "run"]);
const READONLY_TOOLS = new Set(["read", "ls", "find", "grep"]);
const TB21_TASKS = process.env.VFE_TASKS_DIR?.trim() || "tb21/tasks";

function parseStartArgs(args: string): { task: string; rep: number } | undefined {
	const parts = args.trim().split(/\s+/).filter(Boolean);
	if (parts.length === 0) return undefined;
	const task = parts[0];
	let rep = 0;
	for (let i = 1; i < parts.length; i++) {
		const p = parts[i];
		if (p === "--rep" || p === "-r") {
			const v = Number(parts[i + 1]);
			if (Number.isFinite(v)) rep = v;
			i++;
		} else if (p.startsWith("--rep=")) {
			const v = Number(p.slice("--rep=".length));
			if (Number.isFinite(v)) rep = v;
		}
	}
	return { task, rep };
}

function loadInstruction(task: string): string {
	const p = join(TB21_TASKS, task, "instruction.md");
	if (!existsSync(p)) return task;
	return readFileSync(p, "utf8");
}

export default function vfeExtension(pi: ExtensionAPI) {
	const state = createState();
	const docker = new DockerClient();

	const exec = (cmd: string, timeoutSec = 120) => {
		if (!state.container) {
			return Promise.reject(new Error("No container. Run /vfe-start <task> first."));
		}
		return docker.exec(state.container, cmd, timeoutSec);
	};

	// Test/verification exec: injects the task's tests into a private mount namespace
	// so they never exist in the agent-visible filesystem, and runs in an overlay so
	// the suite's own writes leave no residue for later runs (in-loop feedback).
	const execTests = (cmd: string, timeoutSec = 120) => {
		if (!state.container || !state.task) {
			return Promise.reject(new Error("No container/task. Run /vfe-start <task> first."));
		}
		return docker.execTests(state.container, state.task, cmd, timeoutSec, true);
	};

	// Official verifier exec: same tests injection, but on the real rootfs — the
	// official suite must see exactly what official TB gives it (no overlay).
	const execOfficialTests = (cmd: string, timeoutSec = 120) => {
		if (!state.container || !state.task) {
			return Promise.reject(new Error("No container/task. Run /vfe-start <task> first."));
		}
		return docker.execTests(state.container, state.task, cmd, timeoutSec, false);
	};

	// Host-side recovery for settle: docker start (preserve /app) or recreate via sidecar.
	// On recreate the container name is deterministic (task+rep), so state.container stays valid;
	// preclones re-seed inside the sidecar.
	const recoverInfra = async (): Promise<string> => {
		if (!state.container) throw new Error("no container to recover");
		return docker.restartContainer(state.container, state.task ?? "", state.rep ?? 0);
	};

	// Fresh verifier container (tb2verify-*): pristine image + the agent's final
	// /app, used by settle when the in-container official run contradicts an
	// in-loop all-pass (test residue outside /app polluted it).
	let verifyContainer: string | undefined;
	const verifyExec = (cmd: string, timeoutSec = 120) => {
		if (!verifyContainer) return Promise.reject(new Error("No verify container."));
		return docker.exec(verifyContainer, cmd, timeoutSec);
	};
	const verifyExecTests = (cmd: string, timeoutSec = 120) => {
		if (!verifyContainer || !state.task) return Promise.reject(new Error("No verify container/task."));
		return docker.execTests(verifyContainer, state.task, cmd, timeoutSec, false);
	};
	const verifySandbox: VerifySandbox = {
		prepare: async () => {
			if (!state.container || !state.task) throw new Error("No container/task for verify prepare.");
			await docker.commitVerifyImage(state.task, state.rep ?? 0, state.container);
		},
		start: async () => {
			if (!state.container || !state.task) throw new Error("No container/task for fresh verify.");
			verifyContainer = await docker.startVerifyContainer(state.task, state.rep ?? 0, state.container);
		},
		stop: async () => {
			const c = verifyContainer;
			verifyContainer = undefined;
			if (c) await docker.stopContainer(c);
		},
		exec: verifyExec,
		execTests: verifyExecTests,
	};

	registerContainerTools(pi, state, exec, execTests);
	registerProposePlan(pi, state, STEP_TOOLS);
	registerProposeSequences(pi, state, exec);
	registerAssessStep(pi, state);

	// Replace the default coding-assistant prompt while VFE is active so the
	// model does not try bash/read before propose_plan (top) or invent tools.
	// forceSystemPrompt persists across mid-run tool-loadout changes.
	pi.on("before_agent_start", (event) => {
		if (state.phase === "idle") return undefined;
		const base = event.systemPrompt;
		return { systemPrompt: base ? `${base}\n\n${VFE_SYSTEM_PROMPT}` : VFE_SYSTEM_PROMPT };
	});

	// Step phase rule: before selection only propose_sequences; after selection
	// only observe/refine tools (write/edit/run count against the observe budget).
	pi.on("tool_call", async (event, _ctx) => {
		// Every block is a harness decision the model saw; log it (with reason)
		// so gate behaviour is diagnosable after the run.
		const blocked = (reason: string, terminate = false) => {
			appendEfeDebug({
				phase: state.phase,
				step: state.steps,
				tool: event.toolName,
				action: String(event.toolName).slice(0, 200),
				decision: terminate ? "block_terminate" : "block",
				reason: reason.slice(0, 300),
				observeCount: state.observeCount,
				budget: envObserveBudget(),
				selected: state.selectedSequence?.id,
			});
			return terminate ? { block: true, terminate: true, reason } : { block: true, reason };
		};
		if (state.phase === "top") {
			if (event.toolName !== PROPOSE_PLAN_TOOL) {
				return blocked(`top phase only allows ${PROPOSE_PLAN_TOOL}`);
			}
			return undefined;
		}
		if (state.phase === "end" || state.phase === "idle") {
			return blocked(`phase=${state.phase}: no tools allowed`);
		}
		if (state.phase !== "step") return undefined;
		// Closing call: exempt from the observe budget (this is how a
		// budget-exhausted turn still records its assessment), but a sequence
		// must have run first — there is nothing to assess before that.
		if (event.toolName === ASSESS_STEP_TOOL) {
			if (!state.selectedSequence) {
				return blocked(
					"Run your chosen sequence (propose_sequences) and observe its result before assessing the step.",
				);
			}
			return undefined;
		}
		if (event.toolName === PROPOSE_PLAN_TOOL) {
			return blocked("propose_plan only valid in top phase");
		}
		if (event.toolName === PROPOSE_SEQUENCES_TOOL) {
			if (state.selectedSequence) {
				return blocked(
					`Sequence already selected this step (${state.selectedSequence.label}). Observe/refine; call assess_step {verdict, evidence}, then end the turn.`,
				);
			}
			return undefined;
		}
		// The observe/refine budget only applies AFTER a sequence was selected —
		// that is the only way observeCount accrues. Protocol tools handled above
		// stay callable, so the steer to assess_step (exempt) is never contradicted
		// by another gate.
		if (state.selectedSequence && state.observeCount >= envObserveBudget()) {
			state.budgetBlocks += 1;
			// One steer to assess_step, then a hard stop: the model gets a closure
			// channel, but a model that ignores the steer must not loop on blocked calls.
			if (state.budgetBlocks >= 2) {
				return blocked(
					`Observation budget exhausted (${envObserveBudget()}). End the turn now; settle/feedback will run automatically. Do not call more tools.`,
					true,
				);
			}
			return blocked(
				`Observation budget exhausted (${envObserveBudget()}). Call assess_step {verdict: better|worse|same, evidence} with your verdict for this step, then end the turn. No other tools.`,
			);
		}
		if (!state.selectedSequence) {
			return blocked(
				`Call ${PROPOSE_SEQUENCES_TOOL} first with 2-3 candidate sequences and select one to run. If you were about to run a different tool first, include that intended action as one of the candidates.`,
			);
		}
		if (!OBSERVE_TOOLS.has(event.toolName)) {
			return blocked(
				"A sequence was already selected and executed this step. Only read/ls/find/grep/run_test/write/edit/run are allowed until settle.",
			);
		}
		if (READONLY_TOOLS.has(event.toolName)) {
			const input = event.input as Record<string, unknown>;
			const key = `${event.toolName} ${String(input.path ?? input.pattern ?? "")}`.slice(0, 120);
			const prior = state.readonlyCounts[key] ?? 0;
			if (prior >= READONLY_GATE) {
				return blocked(
					`Redundant read: ${key} already ran ${prior} time(s) with no new info. Propose an action that CHANGES something, or end the turn.`,
				);
			}
			state.readonlyCounts[key] = prior + 1;
		}
		return undefined;
	});

	// Collect tool outputs for settle diary / trajectory + tool-practice credit.
	pi.on("tool_result", (event) => {
		if (state.phase !== "step") return undefined;
		if (event.toolName === PROPOSE_PLAN_TOOL) return undefined;
		// Winner commands are already pushed as run rows by propose_sequences;
		// assess_step records its verdict in state, not in the diary batch.
		if (event.toolName === PROPOSE_SEQUENCES_TOOL) return undefined;
		if (event.toolName === ASSESS_STEP_TOOL) return undefined;
		const text = event.content
			.filter((c): c is { type: "text"; text: string } => c.type === "text")
			.map((c) => c.text)
			.join("\n");
		state.turnTools.push({
			tool: event.toolName,
			input: JSON.stringify(event.input).slice(0, 400),
			output: text.slice(0, 4000),
			isError: event.isError,
			exitCode: event.isError
				? 1
				: /exit(?: code)?[:=]\s*(-?\d+)/i.exec(text)?.[1] !== undefined
					? Number(/exit(?: code)?[:=]\s*(-?\d+)/i.exec(text)?.[1])
					: 0,
			// Daemon/OCI errors may appear even when the tool reports isError=false.
			// Keep this pattern narrow: task text (assert messages) must not flag infra.
			...(/OCI runtime exec failed|Error response from daemon|Cannot connect to the Docker daemon|No such container:/i.test(
				text,
			)
				? { infra: true }
				: {}),
		});
		if (state.turnTools.length > 40) state.turnTools.splice(0, state.turnTools.length - 40);
		if (state.selectedSequence && OBSERVE_TOOLS.has(event.toolName) && !event.isError) {
			state.observeCount++;
		}
		return undefined;
	});

	// Step state machine: one agent turn = one step; settle on agent_end.
	// First step is armed in propose_plan (mid-run followUp does not re-emit
	// agent_start). Subsequent steps arm here when settle queues followUp.
	pi.on("agent_start", () => {
		if (state.phase !== "step" || !state.awaitingStepTurn) return;
		state.awaitingStepTurn = false;
		state.pendingSettle = true;
		state.turnTools = [];
		state.selectedSequence = undefined;
		state.observeCount = 0;
		state.budgetBlocks = 0;
		// Belt-and-braces: settle already consumed/cleared it; never let a
		// verdict leak into a turn it does not describe.
		state.turnVerdict = undefined;
		state.turnVerdictEvidence = undefined;
		state.turnProviderFault = false;
	});
	pi.on("agent_end", async (event, ctx) => {
		if (state.phase !== "step" || !state.pendingSettle) return;
		state.pendingSettle = false;
		// A provider error ends the agent loop before any tool call runs, so the
		// turn looks tool-less. Flag it so settle freezes the streak counters
		// instead of counting a 503 as a degenerate EMPTY-TURN.
		for (let i = event.messages.length - 1; i >= 0; i--) {
			const m = event.messages[i];
			if (m.role !== "assistant") continue;
			state.turnProviderFault = m.stopReason === "error";
			break;
		}
		try {
			await settleStep(
				pi,
				ctx,
				state,
				exec,
				execTests,
				envMaxSteps(),
				recoverInfra,
				verifySandbox,
				execOfficialTests,
			);
		} catch (err) {
			const msg = err instanceof Error ? err.message : String(err);
			ctx.ui.notify(`settle failed at step ${state.steps}: ${msg}`, "error");
			state.awaitingStepTurn = true;
			pi.sendUserMessage(
				`Settle failed at step ${state.steps} (${msg}). Skip tool use; reply only with the one word CONTINUE.`,
				{ deliverAs: "followUp" },
			);
		}
	});

	pi.registerCommand("vfe-start", {
		description: "Start TB21 task container and enter VFE top phase: /vfe-start <task> [--rep N]",
		handler: async (args, ctx) => {
			const parsed = parseStartArgs(args);
			if (!parsed) {
				ctx.ui.notify("Usage: /vfe-start <task> [--rep N]", "warning");
				return;
			}
			ctx.ui.notify(`Starting container for ${parsed.task} (rep=${parsed.rep})…`, "info");
			try {
				if (state.container) {
					await docker.stopContainer(state.container).catch(() => undefined);
				}
				const container = await docker.startContainer(parsed.task, parsed.rep);
				state.task = parsed.task;
				state.rep = parsed.rep;
				state.container = container;
				resetStepRuntime(state);
				state.phase = "top";
				state.belief = undefined;
				state.objective = undefined;
				state.adaptiveSkill = undefined;
				state.stableSkill = undefined;
				state.instruction = loadInstruction(parsed.task);
				state.store = loadStore();
				state.store.task = parsed.task;
				state.envConstraints = [];
				state.hardFailures = [];
				state.deadFamilies = {};
				state.phaseLog = [];
				state.phaseStartStep = 0;
				state.sameTestStreak = 0;
				state.lastFailSet = [];
				state.failureHistory = [];
				state.readonlyCounts = {};
				ctx.ui.notify(`Container ready: ${container}. Collecting recon evidence…`, "info");
				// If github.com is unreachable, sidecar may have seeded host preclones.
				try {
					const ls = await exec("ls -la /app 2>/dev/null | head -20", 15);
					if (ls.output.includes("pyknotid")) {
						ctx.ui.notify("Preseeded /app content present (network workaround).", "info");
					}
				} catch {
					// recon will surface missing /app
				}
				state.envObs = await probeEnv(exec);
				state.reconEvidence = await reconEvidence(exec, state.instruction);
				state.envConstraints = await probeEnvConstraintsWith(ctx, state.envObs);
				const expText = renderAllExperiences(state.store, state.instruction);
				dumpInjected(state.store, state.task, state.rep, "top");
				const prompt = buildTopPrompt({
					instruction: state.instruction,
					reconEvidence: state.reconEvidence,
					experienceText: expText,
					envObs: state.envObs,
					envConstraints: state.envConstraints,
				});
				pi.setActiveTools([...TOP_TOOLS]);
				pi.setSessionName(`vfe:${parsed.task}-r${parsed.rep}`);
				ctx.ui.notify(
					`Top phase ready (recon=${state.reconEvidence.length}B exp=${state.store.experiences.size} envc=${state.envConstraints.length}). Tools: propose_plan`,
					"info",
				);
				pi.sendUserMessage(prompt);
			} catch (err) {
				const msg = err instanceof Error ? err.message : String(err);
				ctx.ui.notify(`vfe-start failed: ${msg}`, "error");
			}
		},
	});

	pi.registerCommand("vfe-stop", {
		description: "Stop the VFE task container",
		handler: async (_args, ctx) => {
			if (!state.container) {
				ctx.ui.notify("No container running.", "warning");
				return;
			}
			const name = state.container;
			state.container = undefined;
			resetStepRuntime(state);
			state.phase = "idle";
			try {
				await docker.stopContainer(name);
				ctx.ui.notify(`Stopped ${name}`, "info");
			} catch (err) {
				const msg = err instanceof Error ? err.message : String(err);
				ctx.ui.notify(`stop error: ${msg}`, "error");
			}
		},
	});

	pi.registerCommand("vfe-status", {
		description: "Show VFE run state",
		handler: async (_args, ctx) => {
			ctx.ui.notify(
				[
					`phase=${state.phase}`,
					`task=${state.task ?? "-"} rep=${state.rep ?? "-"}`,
					`container=${state.container ?? "-"}`,
					`objective=${state.objective ?? "-"}`,
					`todos=${state.belief?.todos.length ?? 0}`,
					`steps=${state.steps}`,
					`awaiting=${state.awaitingStepTurn} pending_settle=${state.pendingSettle}`,
					`same_test=${state.sameTestStreak} fail_hist=${state.failureHistory.length}`,
					`skill=${state.adaptiveSkill ? `${state.adaptiveSkill.length}B` : "-"}`,
					`envc=${state.envConstraints.length} hard_fail=${state.hardFailures.length} store=${state.store?.experiences.size ?? 0}`,
				].join("\n"),
				"info",
			);
		},
	});

	pi.on("session_shutdown", () => {
		docker.dispose();
	});
}
