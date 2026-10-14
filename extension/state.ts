import { readFileSync } from "node:fs";
import { join } from "node:path";
import type { WorldState } from "./world.ts";

export type TodoStatus = "pending" | "in_progress" | "completed";

export interface TodoItem {
	task: string;
	status: TodoStatus;
}

export interface VfeBelief {
	objective: string;
	todos: TodoItem[];
	target_files: string[];
	history: string[];
	skill_summary: string;
	skill_corrections: string[];
}

/** Agent self-assessment of one settle's outcome; the harness never infers it. */
export type StepVerdict = "better" | "worse" | "same";

export interface StepLogEntry {
	step: number;
	command: string;
	prediction: Record<string, unknown>;
	observation: Record<string, unknown>;
	signal: string;
	summary: string;
	improved: boolean;
	new_fails: string[];
	fixed_fails: string[];
	note?: string;
	/** Raw tool output of the executed sequence (exit + key lines), not the reflection summary. */
	rawObservation?: string;
	/** Self-assessed progress that drove this settle (absent = agent did not assess). */
	verdict?: StepVerdict;
	verdict_evidence?: string;
}

export interface TurnToolRecord {
	tool: string;
	input: string;
	output: string;
	isError: boolean;
	exitCode?: number;
	/** Infrastructure-domain failure (container dead), not a task command failure. */
	infra?: boolean;
}

export interface EnvConstraint {
	title: string;
	text: string;
}

export interface PhaseLogEntry {
	step: number;
	objective: string;
}

export interface SequenceCandidate {
	id: string;
	label: string;
	commands: string[];
	/** Preconditions the candidate declares (path, bin:<name>, or port:<n>). */
	assumes?: string[];
	/** Artifacts the candidate promises to create (paths). */
	provides?: string[];
}

export interface SelectedSequence {
	id: string;
	label: string;
	commands: string[];
	why?: string;
}

export type VfePhase = "idle" | "top" | "step" | "end";

export interface VfeState {
	phase: VfePhase;
	task?: string;
	rep?: number;
	container?: string;
	objective?: string;
	steps: number;
	instruction?: string;
	reconEvidence?: string;
	envObs?: string;
	adaptiveSkill?: string;
	stableSkill?: string;
	belief?: VfeBelief;
	/** One step-message has been sent; settle on the next agent_start for that run. */
	awaitingStepTurn: boolean;
	/** Current agent run is a step run and should settle on agent_end. */
	pendingSettle: boolean;
	/** Tool activity collected during the current step run. */
	turnTools: TurnToolRecord[];
	/** Sequence chosen by the agent for the current step (cleared each step). */
	selectedSequence?: SelectedSequence;
	/** Observe-tool calls after selection in the current step (budget → steer to assess_step). */
	observeCount: number;
	/** Consecutive budget-gate blocks this turn; force-terminate only after it ignores repeated steers. */
	budgetBlocks: number;
	stepLog: StepLogEntry[];
	prevFails: string[];
	bestFails: string[];
	missingPrev: string[];
	fileStatePrev: Record<string, string>;
	envNoTests: number;
	/** Consecutive steps with an infra fault (even if recovered); abort at ENV_INFRA_MAX. */
	infraStreak: number;
	lastReviseStep: number;
	noImproveStreak: number;
	nImproved: number;
	rollbackCount: number;
	stallAbort: boolean;
	earlyStop: boolean;
	abortNote?: string;
	/** Cross-task experience store (loaded at /vfe-start). */
	store?: import("./memory.ts").ExperienceStore;
	envConstraints: EnvConstraint[];
	/** Hard failure ledger (DEAD-FAMILY, ERR-LOC-STALL) — promote at end. */
	hardFailures: string[];
	deadFamilies: Record<string, number>;
	phaseLog: PhaseLogEntry[];
	phaseStartStep: number;
	prevErrLines: Record<string, string>;
	errUnchanged: number;
	/** Consecutive settles with an unchanged failing-test set (raw23 _same_test_streak). */
	sameTestStreak: number;
	/** Last pytest assertion line, its measured value and its number-free shape. */
	prevAssertionText?: string;
	prevAssertionValue?: number;
	prevAssertionSignature?: string;
	/** Consecutive settles with a byte-identical assertion line. */
	assertionStreak: number;
	/** Verdict the agent submitted for THIS step via assess_step (cleared at settle). */
	turnVerdict?: StepVerdict;
	turnVerdictEvidence?: string;
	/** Step + evidence of the last self-assessed "better" (/app_qbest reference). */
	bestVerdict?: { step: number; evidence: string };
	/** Consecutive settles self-assessed as same/worse → STALL guidance line. */
	verdictSameStreak: number;
	/** Consecutive settles with an identical fail set AND unchanged missing outputs → STALL-ABORT. */
	factStagnationStreak: number;
	/** Consecutive turns that executed no tool call → degenerate-loop abort. */
	emptyTurnStreak: number;
	/**
	 * The agent turn just settled ended in a model/provider error (HTTP 503,
	 * no_available_workers, …) before any tool call could run. Not a degenerate
	 * agent turn: counters must freeze exactly as on an infra-recovered step.
	 */
	turnProviderFault: boolean;
	/** Official-verifier contradictions forgiven: in-loop pass + official 0 → feedback rounds. */
	officialResumes: number;
	/** Step at which the current official-resume window expires; undefined = no window. */
	resumeWindowEnd?: number;
	/** Official-failure reminder echoed into feedback while a resume window is open. */
	resumeHint?: string;
	lastFailSet: string[];
	/** Recent failure one-liners for FAILURE HISTORY (raw23 _failure_history). */
	failureHistory: string[];
	/** Read-only tool invocation counts for the redundant-read select-gate. */
	readonlyCounts: Record<string, number>;
	/** Latest deterministic container snapshot, refreshed each settle. */
	worldState?: WorldState;
}

export const MAX_STEPS_DEFAULT = 70;
export const REVISE_EVERY = 2;
export const REPLAN_STALL = 6;
export const STALL_ABORT = 30;
export const ENV_NO_TESTS_MAX = 6;
/** Consecutive infra-fault steps (recovered or not) before ENV-IRRECOVERABLE abort. */
export const ENV_INFRA_MAX = 3;
export const DEAD_THRESHOLD = 4;
export const ERR_LOC_STALL = 3;
/** Consecutive turns with no executed tool call (degenerate/text-only) before abort. */
export const EMPTY_TURN_ABORT = 4;
/**
 * Official 0 vs in-loop all-pass contradiction: hand the official failures to
 * the agent and let it retry this many times before the 0 stands.
 */
export const OFFICIAL_RESUME_MAX = 3;
/** Extra steps granted per official resume (used when the contradiction hits at maxSteps). */
export const OFFICIAL_RESUME_STEPS = 10;
export const MEM_SAVE_EVERY = 5;
/** Streak of identical fail sets → DIAGNOSTIC note (raw23 SAME_TEST_STALL). */
export const SAME_TEST_STALL = 5;
/** Streak ≥ this + consecutive target rewrites → REPETITION ALERT. */
export const REPEAT_GATE = 3;
/** Same read-only tool invocation already executed this many times → block. */
export const READONLY_GATE = 2;
/** Observe calls allowed after the chosen sequence runs; beyond this force end-of-turn. */
export function envObserveBudget(): number {
	const v = Number(process.env.VFE_OBSERVE_BUDGET);
	return Number.isFinite(v) && v >= 1 ? Math.floor(v) : 6;
}

export function createState(): VfeState {
	return {
		phase: "idle",
		steps: 0,
		awaitingStepTurn: false,
		pendingSettle: false,
		turnTools: [],
		selectedSequence: undefined,
		observeCount: 0,
		budgetBlocks: 0,
		stepLog: [],
		prevFails: [],
		bestFails: [],
		missingPrev: [],
		fileStatePrev: {},
		envNoTests: 0,
		infraStreak: 0,
		lastReviseStep: 0,
		noImproveStreak: 0,
		nImproved: 0,
		rollbackCount: 0,
		stallAbort: false,
		earlyStop: false,
		envConstraints: [],
		hardFailures: [],
		deadFamilies: {},
		phaseLog: [],
		phaseStartStep: 0,
		prevErrLines: {},
		errUnchanged: 0,
		sameTestStreak: 0,
		assertionStreak: 0,
		verdictSameStreak: 0,
		factStagnationStreak: 0,
		emptyTurnStreak: 0,
		turnProviderFault: false,
		officialResumes: 0,
		lastFailSet: [],
		failureHistory: [],
		readonlyCounts: {},
	};
}

export function requireContainer(state: VfeState): string {
	if (!state.container) {
		throw new Error("No container. Run /vfe-start <task> [--rep N] first.");
	}
	return state.container;
}

export function normTodos(raw: unknown): TodoItem[] {
	const out: TodoItem[] = [];
	if (!Array.isArray(raw)) return out;
	for (const it of raw) {
		let task = "";
		let status: TodoStatus = "pending";
		if (it && typeof it === "object") {
			const obj = it as { task?: unknown; status?: unknown };
			task = String(obj.task ?? "").trim();
			const s = String(obj.status ?? "pending")
				.trim()
				.toLowerCase();
			status = s === "in_progress" || s === "completed" ? s : "pending";
		} else {
			task = String(it).trim();
		}
		if (!task) continue;
		out.push({ task, status });
	}
	let seenIp = false;
	for (const item of out) {
		if (item.status === "in_progress") {
			if (seenIp) item.status = "pending";
			else seenIp = true;
		}
	}
	return out;
}

export function runDirFor(task: string, rep: number): string {
	return `/tmp/vfe/${task}_rep${rep}`;
}

export function envMaxSteps(): number {
	const v = Number(process.env.VFE_MAX_STEPS);
	return Number.isFinite(v) && v > 0 ? Math.floor(v) : MAX_STEPS_DEFAULT;
}

const TASKS_DIR = process.env.VFE_TASKS_DIR?.trim() || "tb21/tasks";

/**
 * Verifier wall clock from task.toml (`[verifier] timeout_sec`), clamped to
 * [300, 3600]: the official budget must cover real suites (reshard fixture
 * ~7min, verifier 3600) but cannot stall a single settle step indefinitely
 * (sam declares 7200).
 */
export function verifierTimeoutSec(task?: string): number {
	if (!task) return 300;
	let spec: number;
	try {
		const toml = readFileSync(join(TASKS_DIR, task, "task.toml"), "utf8");
		const m = /^\[verifier\][^[]*?timeout_sec\s*=\s*(\d+(?:\.\d+)?)/m.exec(toml);
		spec = m ? Number(m[1]) : Number.NaN;
	} catch {
		return 300;
	}
	if (!Number.isFinite(spec) || spec <= 0) return 300;
	return Math.min(3600, Math.max(300, Math.floor(spec)));
}

export function resetStepRuntime(state: VfeState): void {
	state.awaitingStepTurn = false;
	state.pendingSettle = false;
	state.turnTools = [];
	state.selectedSequence = undefined;
	state.observeCount = 0;
	state.budgetBlocks = 0;
	state.stepLog = [];
	state.prevFails = [];
	state.bestFails = [];
	state.missingPrev = [];
	state.fileStatePrev = {};
	state.envNoTests = 0;
	state.infraStreak = 0;
	state.lastReviseStep = 0;
	state.noImproveStreak = 0;
	state.nImproved = 0;
	state.rollbackCount = 0;
	state.stallAbort = false;
	state.earlyStop = false;
	state.abortNote = undefined;
	state.steps = 0;
	state.hardFailures = [];
	state.deadFamilies = {};
	state.phaseLog = [];
	state.phaseStartStep = 0;
	state.prevErrLines = {};
	state.errUnchanged = 0;
	state.sameTestStreak = 0;
	state.prevAssertionText = undefined;
	state.prevAssertionValue = undefined;
	state.prevAssertionSignature = undefined;
	state.assertionStreak = 0;
	state.turnVerdict = undefined;
	state.turnVerdictEvidence = undefined;
	state.bestVerdict = undefined;
	state.verdictSameStreak = 0;
	state.factStagnationStreak = 0;
	state.emptyTurnStreak = 0;
	state.turnProviderFault = false;
	state.officialResumes = 0;
	state.resumeWindowEnd = undefined;
	state.resumeHint = undefined;
	state.lastFailSet = [];
	state.failureHistory = [];
	state.readonlyCounts = {};
	state.worldState = undefined;
}
