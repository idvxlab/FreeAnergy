import { appendFileSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { completeText } from "./llm.ts";
import {
	dumpInjected,
	type ExperienceStore,
	evalConstraintsPhase,
	evalMethodsPhase,
	extractBestPractice,
	extractLessons,
	type LocalMemoryInput,
	markConf,
	maturePracticeHits,
	maybePeriodicSave,
	promoteLocalFull,
	saveExperiences,
	updateToolPractice,
} from "./memory.ts";
import {
	buildFeedbackPrompt,
	REFLECT_SYSTEM,
	REVISE_STATES_SYSTEM,
	reflectUserPrompt,
	reviseUserPrompt,
} from "./prompts.ts";
import {
	DEAD_THRESHOLD,
	EMPTY_TURN_ABORT,
	ENV_INFRA_MAX,
	ENV_NO_TESTS_MAX,
	ERR_LOC_STALL,
	MAX_STEPS_DEFAULT,
	normTodos,
	OFFICIAL_RESUME_MAX,
	OFFICIAL_RESUME_STEPS,
	REPEAT_GATE,
	REPLAN_STALL,
	REVISE_EVERY,
	runDirFor,
	SAME_TEST_STALL,
	STALL_ABORT,
	type StepLogEntry,
	type StepVerdict,
	type VfeState,
	verifierTimeoutSec,
} from "./state.ts";
import type { ExecResult } from "./types.ts";
import { collectWorldState, inferConsumed, renderWorldState } from "./world.ts";

/**
 * Official test.sh system packages the verifier shells out to, per task. A task
 * absent from this map does no apt work at all (the bootstrap is on the hot path
 * of every settle). Missing markers are `command -v` for binaries and an
 * `ldconfig` probe for libGL (sam's cv2 import needs it). The marker file keeps
 * the apt attempt one-shot per container so an offline image does not pay
 * `apt-get update` on every settle, and each package installs on its own so an
 * unavailable name cannot block the others.
 */
const VERIFIER_APT_PKGS: Record<string, string[]> = {
	"qemu-alpine-ssh": ["sshpass"],
	"sam-cell-seg": ["libgl1"],
	"compile-compcert": ["curl", "binutils"],
	"dna-assembly": ["curl", "primer3"],
	// headless test.sh apt-installs vim; without it test_send_interactive can
	// never pass in-loop (vim.txt is never created) — a permanently unfixable
	// feedback signal for a code-fixable test.
	"headless-terminal": ["vim"],
	mailman: ["curl"],
	"make-doom-for-mips": ["curl"],
	"make-mips-interpreter": ["curl"],
	"merge-diff-arc-agi-task": ["git"],
	"torch-tensor-parallelism": ["python3", "python3-pip"],
	"torch-pipeline-parallelism": ["python3", "python3-pip"],
};

/**
 * Python packages the verifier imports that the bare image does not ship
 * (installed once from the offline seed; a no-op once the right version is
 * importable). A `pkg==pin` spec is version-checked, not just imported:
 * reshard pins hub 0.36.x because the image ships 2.0.0, whose httpx2 client
 * ignores the sitecustomize HF_ENDPOINT rewrite and hangs on huggingface.co.
 */
const SEED_PY_PKGS: Record<string, string[]> = {
	"reshard-c4-data": ["datasets==3.6.0", "tqdm==4.67.1", "huggingface_hub==0.36.2"],
	"protein-assembly": ["biopython==1.85"],
	"financial-document-processor": ["pandas==2.3.2"],
	"torch-pipeline-parallelism": ["transformers"],
};

/**
 * Per-task shell setup mirroring the official tests/test.sh provisioning that
 * is not expressible as an apt/pip package (e.g. building a CLI from source).
 * Runs once per container before pytest; marker avoids rework.
 */
const VERIFIER_SETUP: Record<string, string> = {
	"train-fasttext":
		"command -v fasttext >/dev/null 2>&1 || [ -f /tmp/.vfe_fasttext ] || { " +
		"export DEBIAN_FRONTEND=noninteractive; " +
		"apt-get update -qq >/dev/null 2>&1 || true; " +
		"apt-get install -y -qq git build-essential >/dev/null 2>&1 || true; " +
		"rm -rf /opt/fastText; " +
		"git clone --depth 1 https://github.com/facebookresearch/fastText.git /opt/fastText >/dev/null 2>&1 && " +
		"make -C /opt/fastText -j2 >/dev/null 2>&1 && cp /opt/fastText/fasttext /usr/local/bin/ && touch /tmp/.vfe_fasttext; };",
	"torch-tensor-parallelism":
		"python3 -c 'import torch,numpy' 2>/dev/null || { " +
		"python3 -m pip install --quiet --break-system-packages --index-url https://download.pytorch.org/whl/cpu torch numpy; };",
	"torch-pipeline-parallelism":
		"python3 -c 'import torch,numpy' 2>/dev/null || { " +
		"python3 -m pip install --quiet --break-system-packages --index-url https://download.pytorch.org/whl/cpu torch numpy; };",
};

function verifierSetup(task?: string): string {
	return VERIFIER_SETUP[task ?? ""] ?? "true;";
}

function seedPySteps(task?: string): string[] {
	return (SEED_PY_PKGS[task ?? ""] ?? []).map((spec) => {
		const [pkg, pin] = spec.split("==");
		const ready = pin
			? `python3 -c 'import importlib.metadata as m,sys; sys.exit(0 if m.version("${pkg}") == "${pin}" else 1)' 2>/dev/null`
			: `python3 -c 'import ${pkg}' 2>/dev/null`;
		return `${ready} || { ${seedPip(pkg, pin)}; };`;
	});
}

function verifierDepsApt(task?: string): string {
	const pkgs = VERIFIER_APT_PKGS[task ?? ""] ?? [];
	if (pkgs.length === 0) return "true;";
	const present = pkgs
		.map((pkg) =>
			pkg === "libgl1" ? "ldconfig -p 2>/dev/null | grep -q 'libGL\\.so\\.1'" : `command -v ${pkg} >/dev/null 2>&1`,
		)
		.join(" && ");
	return (
		`${present} || [ -f /tmp/.vfe_verifier_deps ] || { ` +
		"export DEBIAN_FRONTEND=noninteractive; " +
		"apt-get update -qq >/dev/null 2>&1 || true; " +
		`for p in ${pkgs.join(" ")}; do apt-get install -y -qq "$p" >/dev/null 2>&1 || true; done; ` +
		"touch /tmp/.vfe_verifier_deps; };"
	);
}

/**
 * Minimal, task-agnostic process/net toolbox. Stripped task images ship without
 * procps/iproute2/psmisc, so the agent's kill/port commands silently no-op and
 * it burns steps discovering that. Provisioned uniformly for every task,
 * one-shot via /tmp/.vfe_toolbox. NOTE: this augments the task container
 * environment (not just the harness); keep it uniform and disclose it.
 */
const TOOLBOX_APT_PKGS = ["ca-certificates", "procps", "iproute2", "psmisc"];

function toolboxApt(): string {
	return (
		"if [ ! -f /tmp/.vfe_toolbox ]; then " +
		"if ! command -v pkill >/dev/null 2>&1 || [ ! -e /etc/ssl/certs/ca-certificates.crt ]; then " +
		"export DEBIAN_FRONTEND=noninteractive; " +
		"apt-get update -qq >/dev/null 2>&1 || true; " +
		`for p in ${TOOLBOX_APT_PKGS.join(" ")}; do apt-get install -y -qq "$p" >/dev/null 2>&1 || true; done; ` +
		"fi; touch /tmp/.vfe_toolbox; fi;"
	);
}

/**
 * Provision pytest for settle/run_test. Prefer offline seed wheels, then pip/apt.
 * The verifier deps step sits outside the pytest block: pytest may already be
 * importable while sshpass/libgl1 are still missing. Ends with
 * `python3 -c 'import pytest'` so exit code is the ground truth.
 *
 * PYTHONSAFEPATH=1 keeps /app off sys.path for every probe below: an /app file
 * the agent names after a stdlib module (inspect.py) otherwise shadows it and
 * makes `import pytest` fail even though pytest is installed correctly.
 */
export function pytestBootstrap(task?: string): string {
	return [
		"export PYTHONSAFEPATH=1;",
		"python3 -c 'import pytest' 2>/dev/null || {",
		"python3 -m pip --version >/dev/null 2>&1 || {",
		"apt-get update -qq >/dev/null 2>&1 || true;",
		"apt-get install -y -qq python3-pip >/dev/null 2>&1 || true;",
		"};",
		"ok=0;",
		"for d in /opt/vfe_seed/wheels /tmp/wheels /opt/vfe_seed /tmp; do",
		'if ls "$d"/pytest-*.whl >/dev/null 2>&1; then',
		'python3 -m pip install --quiet --break-system-packages --no-index --find-links "$d" pytest==8.4.1 >/dev/null 2>&1 && ok=1 && break;',
		'python3 -m pip install --quiet --no-index --find-links "$d" pytest==8.4.1 >/dev/null 2>&1 && ok=1 && break;',
		"fi;",
		"done;",
		'if [ "$ok" != 1 ]; then',
		"python3 -m pip install --quiet --break-system-packages pytest==8.4.1 >/dev/null 2>&1 || " +
			"python3 -m pip install --quiet pytest==8.4.1 >/dev/null 2>&1 || true;",
		"fi;",
		'if ! python3 -c "import pytest" 2>/dev/null; then',
		"apt-get update -qq >/dev/null 2>&1 || true;",
		"apt-get install -y -qq python3-pytest >/dev/null 2>&1 || true;",
		"fi;",
		"};",
		...seedPySteps(task),
		toolboxApt(),
		verifierSetup(task),
		verifierDepsApt(task),
		// no trailing `;`: run_test appends `&& <pytest>` to this string
		'python3 -c "import pytest"',
	].join(" ");
}
/** Official test.sh system packages for OpenCV/X11 collection (pytorch, install-windows). */
export const SO_RECOVERY_APT =
	"export DEBIAN_FRONTEND=noninteractive; " +
	"apt-get update -qq >/dev/null 2>&1 || true; " +
	"apt-get install -y -qq libsm6 libxext6 libxcb1 libgl1 libglib2.0-0 ffmpeg >/dev/null 2>&1 || true; " +
	"echo SO_RECOVER_DONE";

/** Offline seed install helper: find-links search, no network required. */
const SEED_FIND_LINKS = "/opt/vfe_seed/wheels /tmp/wheels /opt/vfe_seed /tmp";

/**
 * pip install from seed wheels when present; else online pip. Package pins
 * match official test.sh where it matters (opencv/numpy/pytest).
 */
function seedPip(pkg: string, pin?: string): string {
	const spec = pin ? `${pkg}==${pin}` : pkg;
	return (
		`ok=0; for d in ${SEED_FIND_LINKS}; do ` +
		`if ls "$d"/*.whl >/dev/null 2>&1; then ` +
		`python3 -m pip install --quiet --break-system-packages --no-index --find-links "$d" '${spec}' >/dev/null 2>&1 && ok=1 && break; ` +
		`python3 -m pip install --quiet --no-index --find-links "$d" '${spec}' >/dev/null 2>&1 && ok=1 && break; ` +
		"fi; done; " +
		`if [ "$ok" != 1 ]; then python3 -m pip install --quiet --break-system-packages '${spec}' >/dev/null 2>&1 || ` +
		`python3 -m pip install --quiet '${spec}' >/dev/null 2>&1 || true; fi; ` +
		"echo SEED_PIP_DONE"
	);
}

const BUILD_FAIL_RE =
	/Failed to compile|collect2: error|ld returned \d+ exit status|fatal error:|error: (?:redefinition|expected|no matching|cannot|'[^']+' was not declared)|error\[E\d+\]|linker command failed|cannot find -l|No such file or directory.*\.(?:h|hpp)\b/i;

const BUILD_ERR_LINE_RE =
	/collect2: error|ld returned \d+ exit status|fatal error:|error: (?:redefinition|expected|no matching|cannot|'[^']+' was not declared)|error\[E\d+\]|linker command failed|cannot find -l/i;

const ERR_LOC_RE =
	/(?:File\s+"([^"]+\.\w+)"\s*,\s*line\s+(\d+))|(?:(?:^|[\s:(])([A-Za-z0-9_./+-]+\.\w+):(\d+)(?::\d+)?(?=\s*(?::\s*)?(?:in\s+\S+|error|Error|AssertionError|\||-|$)))/gm;

/**
 * Docker client/daemon/OCI errors → infrastructure domain.
 * Deliberately narrow: bare substrings like "is not running" appear in task
 * test files (e.g. install-windows assertions) and must not flag turn tools.
 */
const INFRA_OUT_RE =
	/OCI runtime exec failed|Error response from daemon|Cannot connect to the Docker daemon|No such container:/i;

function looksInfra(output: string): boolean {
	return INFRA_OUT_RE.test(output);
}

/** Container workdir/exec broken before the shell starts (docker OCI chdir). */
export function isWorkdirBroken(output: string): boolean {
	return /OCI runtime exec failed|chdir to cwd/i.test(output);
}

function toolInfra(t: { infra?: boolean; output: string }): boolean {
	return t.infra === true || looksInfra(t.output);
}

type Exec = (cmd: string, timeoutSec?: number) => Promise<ExecResult>;

/**
 * In-loop pytest invocation.
 *
 * `python -m pytest` puts the cwd on sys.path, which the official verifier does
 * not: 82/89 official suites invoke pytest through `uvx` (a script, so /app is
 * never importable) and only two run `python -m pytest` from /app. A file the
 * agent drops in /app that shadows a stdlib module (inspect.py) therefore kills
 * the whole collection in-loop while the official suite still passes, and the
 * ImportError traceback echoes harness command lines — which contain the VFE_*
 * infra markers — back into pytest's stdout. PYTHONSAFEPATH=1 keeps the cwd off
 * sys.path; /app is appended so task modules stay importable after
 * stdlib/site-packages.
 */
export function pytestCmd(py: string): string {
	return (
		`cd /app && PYTHONSAFEPATH=1 ${py} -c ` +
		`"import sys;sys.path.append('/app');import pytest;` +
		`sys.exit(pytest.main(['/tests/test_outputs.py','-q','--tb=line']))" 2>&1`
	);
}

export interface PytestOutcome {
	passed: number;
	failed: number;
	fails: string[];
	output: string;
	missing: string[];
	raw: string;
	/** True when pytest could not run because the container is dead (infra domain). */
	infra: boolean;
}

/** Host-side recovery: docker start (preserve /app) or recreate. Returns mode string. */
export type InfraRecovery = () => Promise<string>;

function extractJson(text: string): Record<string, unknown> | undefined {
	const start = text.indexOf("{");
	const end = text.lastIndexOf("}");
	if (start < 0 || end <= start) return undefined;
	try {
		const parsed: unknown = JSON.parse(text.slice(start, end + 1));
		if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
			return parsed as Record<string, unknown>;
		}
	} catch {
		return undefined;
	}
	return undefined;
}

function headTail(s: string, max = 4000): string {
	const buf = Buffer.from(s, "utf8");
	if (buf.length <= max) return s;
	const head = buf.subarray(0, Math.floor(max * 0.65));
	const tail = buf.subarray(buf.length - Math.floor(max * 0.3));
	return `${head.toString("utf8")}\n…[truncated]…\n${tail.toString("utf8")}`;
}

function signalOf(
	improved: boolean,
	newFails: string[],
	fixedFails: string[],
	verdict?: StepVerdict,
	verdictEvidence?: string,
): string {
	if (newFails.length > 0) {
		return `REGRESSION: ${newFails.length} new test(s) failed: ${newFails.slice(0, 5).join(", ")}`;
	}
	if (fixedFails.length > 0) {
		return `IMPROVEMENT: ${fixedFails.length} test(s) now pass: ${fixedFails.slice(0, 5).join(", ")}`;
	}
	// No test-list facts: report the agent's own assessment verbatim. The harness
	// never infers direction — a missing verdict is stated, never judged.
	const ev = verdictEvidence ? `: ${verdictEvidence}` : "";
	if (verdict === "better") return `IMPROVED (self-assessed)${ev}`;
	if (verdict === "worse") return `REGRESSED (self-assessed)${ev}`;
	if (verdict === "same") return `SAME (self-assessed)${ev}`;
	return improved ? "IMPROVED" : "no fail-set change (no step assessment recorded)";
}

const ASSERT_LINE_RE = /(?:^|\n)\S+:\d+:\s*AssertionError:\s*([^\n]+)/;
const ASSERT_VALUE_RE = /(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*[<>]/i;

export interface AssertionInfo {
	text: string;
	value?: number;
	signature: string;
}

function parseAssertion(raw: string): AssertionInfo | undefined {
	const m = ASSERT_LINE_RE.exec(raw);
	if (!m) return undefined;
	const text = m[1].trim();
	const thresholdForm = text.includes(">") || text.includes("<");
	const num = thresholdForm ? ASSERT_VALUE_RE.exec(text) : null;
	const value = num ? Number(num[1]) : Number.NaN;
	return {
		text,
		value: Number.isFinite(value) ? value : undefined,
		signature: text.replace(/-?\d+(?:\.\d+)?(?:e[+-]?\d+)?/gi, "#"),
	};
}

function runDirOf(state: VfeState): string {
	if (state.task) return runDirFor(state.task, state.rep ?? 0);
	return process.env.VFE_RUN_DIR || "/tmp/vfe/_current";
}

function infraOutcome(raw: string): PytestOutcome {
	return {
		passed: 0,
		failed: 0,
		fails: [],
		output: headTail(raw, 4000),
		missing: [],
		raw,
		infra: true,
	};
}

function isInfraResult(r: { output: string; infra?: boolean }): boolean {
	return r.infra === true || looksInfra(r.output);
}

async function runPytestOnce(
	execTests: Exec,
	py: string,
	timeoutSec: number,
): Promise<{ output: string; infra: boolean }> {
	const r = await execTests(pytestCmd(py), timeoutSec).catch(
		() => ({
			output: "",
			exitCode: 1,
			infra: false,
		}),
	);
	return { output: r.output ?? "", infra: isInfraResult(r) };
}

/**
 * Retry pytest provisioning once before declaring infra. Under concurrency the
 * first bootstrap can fail transiently (apt/pip contention, slow proxy); that is
 * a missing test dep, not a dead container. Returns null when the retry fixes it.
 */
async function retryPytestBootstrap(exec: Exec, task: string | undefined, firstOutput: string): Promise<string | null> {
	await new Promise((resolve) => setTimeout(resolve, 5000));
	const boot = await exec(pytestBootstrap(task), 600).catch(() => ({
		output: "",
		exitCode: 1,
		infra: false,
	}));
	if (isInfraResult(boot)) return boot.output || "[infra] bootstrap failed on retry";
	const verify = await exec("PYTHONSAFEPATH=1 python3 -c 'import pytest'", 30).catch(() => ({
		output: "",
		exitCode: 1,
		infra: false,
	}));
	if (isInfraResult(verify)) return verify.output || "[infra] pytest verify failed on retry";
	if ((verify.exitCode ?? 1) !== 0) {
		const vraw = verify.output || boot.output || firstOutput;
		return `[infra] pytest unavailable after bootstrap (harness provisioning failed): ${vraw.slice(0, 800)}`;
	}
	return null;
}

async function runPytest(
	exec: Exec,
	execTests: Exec,
	runDir: string,
	timeoutSec: number,
	task?: string,
): Promise<PytestOutcome> {
	// 600s: the verifier deps step can pull ~60MB (libgl1) on first settle.
	const boot = await exec(pytestBootstrap(task), 600).catch(() => ({
		output: "",
		exitCode: 1,
		infra: false,
	}));
	if (isInfraResult(boot)) return infraOutcome(boot.output || "[infra] bootstrap failed");
	// Bootstrap must leave pytest importable; otherwise harness failed to provision tests.
	const verify = await exec("PYTHONSAFEPATH=1 python3 -c 'import pytest'", 30).catch(() => ({
		output: "",
		exitCode: 1,
		infra: false,
	}));
	if (isInfraResult(verify)) return infraOutcome(verify.output || "[infra] pytest verify failed");
	if ((verify.exitCode ?? 1) !== 0) {
		const vraw = verify.output || boot.output || "";
		// Workdir/OCI breakage is container infra (recover/recreate), not "pytest missing".
		if (isWorkdirBroken(vraw) || looksInfra(vraw)) {
			return infraOutcome(`[infra] container exec/workdir broken: ${vraw.slice(0, 400)}`);
		}
		const retryFailure = await retryPytestBootstrap(exec, task, vraw);
		if (retryFailure) return infraOutcome(retryFailure);
	}
	const snap = await exec("rm -rf /tmp/.app_snap && cp -a /app /tmp/.app_snap && echo SNAP_OK", 180).catch(() => ({
		output: "",
		exitCode: 1,
		infra: false,
	}));
	if (isInfraResult(snap)) return infraOutcome(snap.output || "[infra] snapshot failed");
	const snapOk = snap.output.includes("SNAP_OK");
	let vout = "";
	let sawInfra = false;
	for (const py of ["python3", "python"]) {
		const r = await runPytestOnce(execTests, py, timeoutSec);
		if (r.infra) {
			sawInfra = true;
			vout = r.output;
			break;
		}
		vout = r.output;
		if (!vout.includes("command not found")) break;
	}
	if (sawInfra) return infraOutcome(vout || "[infra] pytest exec failed");

	const collecting = /ERROR collecting|ModuleNotFoundError|ImportError/i.test(vout);
	const soMissing = /cannot open shared object file|error while loading shared libraries/i.test(vout);
	const busError = /Bus error/i.test(vout);
	const pyModMissing = /No module named '([A-Za-z0-9_.-]+)'/.exec(vout);

	// Official test.sh installs these system libs before pytest; bare image does not.
	// Also covers Bus error during cv2 bootstrap (missing/incompatible native libs).
	if ((collecting && soMissing) || busError) {
		const rec = await exec(SO_RECOVERY_APT, 300).catch(() => ({ output: "", exitCode: 1, infra: false }));
		if (isInfraResult(rec)) return infraOutcome(rec.output || "[infra] system-deps install failed");
		// Prefer official-pinned opencv from seed when cv2 is involved.
		if (busError || /cv2|opencv/i.test(vout)) {
			const cv = await exec(seedPip("opencv-python", "4.11.0.86"), 300).catch(() => ({
				output: "",
				exitCode: 1,
				infra: false,
			}));
			if (isInfraResult(cv)) return infraOutcome(cv.output || "[infra] opencv install failed");
		}
		for (const py of ["python3", "python"]) {
			const r = await runPytestOnce(execTests, py, timeoutSec);
			if (r.infra) return infraOutcome(r.output || "[infra] pytest retry after so-recover failed");
			vout = r.output;
			if (!vout.includes("command not found")) break;
		}
		if (
			/cannot open shared object file|error while loading shared libraries/i.test(vout) ||
			/Bus error/i.test(vout)
		) {
			// Last resort: headless wheel (no GUI libs) still exports cv2 API used by tests.
			const hl = await exec(seedPip("opencv-python-headless", "4.11.0.86"), 300).catch(() => ({
				output: "",
				exitCode: 1,
				infra: false,
			}));
			if (isInfraResult(hl)) return infraOutcome(hl.output || "[infra] opencv-headless install failed");
			for (const py of ["python3", "python"]) {
				const r = await runPytestOnce(execTests, py, timeoutSec);
				if (r.infra) return infraOutcome(r.output || "[infra] pytest retry failed");
				vout = r.output;
				if (!vout.includes("command not found")) break;
			}
			if (
				/cannot open shared object file|error while loading shared libraries/i.test(vout) ||
				/Bus error/i.test(vout)
			) {
				return infraOutcome(
					`[infra] test collection blocked by missing system libraries after recovery: ${vout.slice(0, 800)}`,
				);
			}
		}
	}

	if (pyModMissing && collecting) {
		const name = pyModMissing[1];
		const pip =
			(
				{
					cv2: "opencv-python",
					Bio: "biopython",
					sklearn: "scikit-learn",
					PIL: "Pillow",
					yaml: "PyYAML",
				} as Record<string, string>
			)[name] ?? name;
		const pin = pip === "opencv-python" ? "4.11.0.86" : undefined;
		const rec = await exec(`${seedPip(pip, pin)} 2>&1 | tail -3; echo RECOVER_DONE`, 300).catch(() => ({
			output: "",
			exitCode: 1,
			infra: false,
		}));
		if (isInfraResult(rec)) return infraOutcome(rec.output || "[infra] pip install failed");
		for (const py of ["python3", "python"]) {
			const r = await runPytestOnce(execTests, py, timeoutSec);
			if (r.infra) return infraOutcome(r.output || "[infra] pytest retry failed");
			vout = r.output;
			if (!vout.includes("command not found")) break;
		}
		// opencv wheel can still fail on .so — one more system-deps pass
		if (/cannot open shared object file|error while loading shared libraries/i.test(vout)) {
			const rec2 = await exec(SO_RECOVERY_APT, 300).catch(() => ({ output: "", exitCode: 1, infra: false }));
			if (isInfraResult(rec2)) return infraOutcome(rec2.output || "[infra] system-deps install failed");
			for (const py of ["python3", "python"]) {
				const r = await runPytestOnce(execTests, py, timeoutSec);
				if (r.infra) return infraOutcome(r.output || "[infra] pytest retry failed");
				vout = r.output;
				if (!vout.includes("command not found")) break;
			}
			if (/cannot open shared object file|error while loading shared libraries/i.test(vout)) {
				return infraOutcome(
					`[infra] test collection blocked by missing system libraries after recovery: ${vout.slice(0, 800)}`,
				);
			}
		}
	}

	if (/No module named 'pytest'|No module named pytest/.test(vout)) {
		return infraOutcome(`[infra] pytest disappeared after successful verify: ${vout.slice(0, 400)}`);
	}

	if (snapOk) {
		// Stage then swap: never leave /app missing if mv/cp fails mid-way.
		const restore = await exec(
			"rm -rf /app.restore && cp -a /tmp/.app_snap /app.restore && rm -rf /app && mv /app.restore /app && echo RESTORE_OK",
			180,
		).catch(() => ({
			output: "",
			exitCode: 1,
			infra: false,
		}));
		if (isInfraResult(restore)) return infraOutcome(restore.output || "[infra] restore failed");
		if (!restore.output.includes("RESTORE_OK")) {
			// Best-effort rescue from best snapshot so Workdir never stays absent.
			const rescue = await exec(
				"if [ ! -d /app ] && [ -d /tmp/.app_best ]; then rm -rf /app.restore && cp -a /tmp/.app_best /app.restore && mv /app.restore /app && echo RESCUE_OK; elif [ -d /app ]; then echo APP_OK; fi",
				180,
			).catch(() => ({ output: "", exitCode: 1, infra: false }));
			if (isInfraResult(rescue)) return infraOutcome(rescue.output || "[infra] restore rescue failed");
		}
	}
	const passed = Number(/(\d+)\s+passed/.exec(vout)?.[1] ?? 0);
	const failed = Number(/(\d+)\s+failed/.exec(vout)?.[1] ?? 0);
	const fails = [...vout.matchAll(/FAILED\s+\S*::(\w+)/g)].map((m) => m[1]);
	const missing: string[] = [];
	for (const line of vout.split("\n")) {
		if (line.startsWith("/") && line.includes("exists=False")) {
			const p = line.split(" exists=")[0];
			if (p.startsWith("/app")) missing.push(p);
		}
	}
	try {
		mkdirSync(runDir, { recursive: true });
		appendFileSync(join(runDir, "last_pytest_full.out"), `${vout}\n`, "utf8");
	} catch {
		// optional host mirror
	}
	return { passed, failed, fails, output: headTail(vout, 4000), missing, raw: vout, infra: false };
}

export type OfficialOutcome = { reward: number; passed: number; failed: number; fails: string[]; raw: string };
type OfficialRun = { ok: true; outcome: OfficialOutcome } | { ok: false; reason: string };

/**
 * Authoritative verifier: run the task's official /tests/test.sh and read
 * /logs/verifier/reward.txt. Reward must come from the official suite, not the
 * harness's bespoke pytest. Tests and the reward path are injected into a private
 * mount namespace (see DockerEnv.exec_tests), so the agent never sees /tests and
 * cannot pre-write reward.txt. A skip (test.sh could not actually run) returns a
 * reason instead of an outcome; the reason and the raw output tail are appended
 * to `<logFile>.skip.log` so an undefined verdict is diagnosable, and the caller
 * records it in result.json (`official_skip`).
 */
async function runOfficialVerifier(
	exec: Exec,
	execTests: Exec,
	task: string | undefined,
	runDir: string,
	timeoutSec: number,
	logFile = "official_verifier.out",
): Promise<OfficialRun> {
	const skip = (reason: string, raw: string): OfficialRun => {
		try {
			mkdirSync(runDir, { recursive: true });
			appendFileSync(
				join(runDir, `${logFile}.skip.log`),
				`${new Date().toISOString()} reason=${reason}\n${raw.slice(-3000)}\n---\n`,
				"utf8",
			);
		} catch {
			// optional host mirror
		}
		return { ok: false, reason };
	};
	if (!task) return skip("no_task", "");

	// /logs/verifier is where test.sh writes reward.txt; seed uv when possible so
	// test.sh does not depend on astral.sh being reachable. Persistent installs
	// (agent-legit) are preserved; only the tests/reward paths are namespaced.
	const setup = await exec(
		"mkdir -p /root/.local/bin; command -v uvx >/dev/null 2>&1 || " +
			'for d in /opt/vfe_seed /root/.local/bin /usr/local/bin; do [ -x "$d/uvx" ] && cp "$d/uvx" "$d/uv" /usr/local/bin/ 2>/dev/null && cp "$d/uvx" "$d/uv" /root/.local/bin/ 2>/dev/null && break; done; ' +
			'[ -f /root/.local/bin/env ] || printf "%s\\n" "#!/bin/sh" "export PATH=/root/.local/bin:/usr/local/bin:$PATH" > /root/.local/bin/env; ' +
			"chmod +x /root/.local/bin/env 2>/dev/null; echo OFFICIAL_READY",
		120,
	).catch(() => ({ output: "", exitCode: 1, infra: false }));
	if (isInfraResult(setup)) return skip("setup_infra", setup.output ?? "");

	// Keep the container environment (Dockerfile ENV + runtime -e) so task suites
	// that depend on image env still work; the wrapper unsets LD_PRELOAD and masks
	// /etc/ld.so.preload, which are the env-independent preload vectors.
	const runner =
		"cd /app && " +
		"if [ ! -f /tests/test.sh ]; then echo VFE_NO_TESTSH; " +
		"else bash /tests/test.sh 2>&1; fi; " +
		"echo '---VFE-REWARD---'; cat /logs/verifier/reward.txt 2>/dev/null; echo; echo '---VFE-REWARD-END---'";
	const r = await execTests(runner, timeoutSec).catch(() => ({
		output: "",
		exitCode: 1,
		infra: false,
	}));
	if (isInfraResult(r)) return skip("exec_infra", r.output ?? "");
	const raw = r.output ?? "";
	if (raw.includes("VFE_NO_TESTSH")) return skip("no_testsh", raw);
	// If test.sh could not actually execute the suite (uv/network failure), do NOT
	// trust its fallback reward.txt=0 — skip so the caller falls back to the
	// in-loop pytest result instead of turning a real pass into a 0.
	const executed =
		/test session starts|\d+\s+(?:passed|failed|error)/i.test(raw) &&
		!/uvx: command not found|failed to download .*uv-/i.test(raw);
	if (!executed) return skip("not_executed", raw);
	const passed = Number(/(\d+)\s+passed/.exec(raw)?.[1] ?? 0);
	const failed = Number(/(\d+)\s+failed/.exec(raw)?.[1] ?? 0);
	const fails = [...raw.matchAll(/FAILED\s+\S*::(\w+)/g)].map((m) => m[1]);
	try {
		mkdirSync(runDir, { recursive: true });
		appendFileSync(join(runDir, logFile), `${raw}\n`, "utf8");
	} catch {
		// optional host mirror
	}
	let reward: number | undefined;
	const m = /---VFE-REWARD---\s*\n?\s*([01])\s*\n?\s*---VFE-REWARD-END---/.exec(raw);
	if (m) reward = Number(m[1]);
	if (reward === undefined) reward = r.exitCode === 0 && failed === 0 ? 1 : 0;
	return { ok: true, outcome: { reward, passed, failed, fails, raw } };
}

/**
 * Pristine-container sandbox for the fresh official re-verify (index.ts wires it
 * to a tb2verify-* container). start() provisions it with the agent's /app;
 * exec/execTests run against it until stop() removes it.
 */
export interface VerifySandbox {
	/** Pre-official snapshot of the agent environment as the verify image. */
	prepare?: () => Promise<void>;
	start: () => Promise<void>;
	stop: () => Promise<void>;
	exec: (cmd: string, timeoutSec?: number) => Promise<ExecResult>;
	execTests: (cmd: string, timeoutSec?: number) => Promise<ExecResult>;
}

/**
 * Re-run the official suite in a fresh container (pristine image + agent /app).
 * The in-loop pytest leaves residue outside /app (headless-terminal's tests
 * mkdir /server), which fails the official run in the same container right after
 * end — a harness-made false 0, impossible in official TB where the agent never
 * sees /tests. Fresh failure/infra returns undefined and the caller keeps the
 * in-container verdict (conservative: only a fresh pass can upgrade a 0).
 */
async function runFreshVerifier(
	sandbox: VerifySandbox,
	task: string | undefined,
	runDir: string,
	timeoutSec: number,
): Promise<OfficialOutcome | undefined> {
	try {
		await sandbox.start();
	} catch {
		return undefined;
	}
	try {
		const r = await runOfficialVerifier(
			sandbox.exec,
			sandbox.execTests,
			task,
			runDir,
			timeoutSec,
			"official_verifier_fresh.out",
		);
		return r.ok ? r.outcome : undefined;
	} catch {
		return undefined;
	} finally {
		await sandbox.stop().catch(() => undefined);
	}
}

async function snapshotTargets(exec: Exec, paths: string[]): Promise<Record<string, string>> {
	const unique = [...new Set(paths.filter((p) => p.startsWith("/app")))].slice(0, 24);
	if (unique.length === 0) return {};
	const script =
		"import hashlib, json\n" +
		`paths = ${JSON.stringify(unique)}\n` +
		"out = {}\n" +
		"for p in paths:\n" +
		"    try:\n" +
		"        out[p] = hashlib.md5(open(p, 'rb').read()).hexdigest()\n" +
		"    except Exception:\n" +
		"        out[p] = 'MISSING'\n" +
		"print('=== FILE_SNAPSHOT ===')\n" +
		"print(json.dumps(out))\n";
	const b64 = Buffer.from(script, "utf8").toString("base64");
	const r = await exec(`cd /app && python3 -c "import base64;exec(base64.b64decode('${b64}').decode())"`, 30).catch(
		() => ({ output: "", exitCode: 1, infra: false }),
	);
	if (r.infra || !r.output.includes("=== FILE_SNAPSHOT ===")) return {};
	const line = r.output.split("=== FILE_SNAPSHOT ===").pop()?.trim().split("\n")[0] ?? "";
	try {
		const parsed: unknown = JSON.parse(line);
		if (parsed && typeof parsed === "object") {
			return Object.fromEntries(Object.entries(parsed as Record<string, unknown>).map(([k, v]) => [k, String(v)]));
		}
	} catch {
		return {};
	}
	return {};
}

function taskImproved(sig: {
	failsNow: string[];
	failsPrev: string[];
	missingNow: string[];
	missingPrev: string[];
}): boolean {
	return sig.failsNow.length < sig.failsPrev.length || (sig.missingPrev.length > 0 && sig.missingNow.length === 0);
}

function fileTouched(fileState: Record<string, string>, fileStatePrev: Record<string, string>): boolean {
	return Object.keys(fileState).length > 0 && JSON.stringify(fileState) !== JSON.stringify(fileStatePrev);
}

function feedbackHint(state: VfeState): { signal: string; summary: string; note?: string } {
	const e = state.stepLog.at(-1);
	if (!e) return { signal: "NO DATA", summary: "" };
	return { signal: e.signal, summary: e.summary, note: e.note };
}

function sameFailSet(a: string[], b: string[]): boolean {
	if (a.length === 0 && b.length === 0) return false;
	if (a.length !== b.length) return false;
	const sa = [...a].sort().join("\n");
	const sb = [...b].sort().join("\n");
	return sa === sb;
}

/** raw23 _same_test_streak: consecutive settles with an unchanged fail set. */
function trackSameTestStreak(state: VfeState, failsNow: string[]): void {
	const prev = state.prevFails;
	if (prev.length > 0 && sameFailSet(failsNow, prev)) state.sameTestStreak += 1;
	else if (failsNow.length === 0 && prev.length > 0) state.sameTestStreak += 1;
	else state.sameTestStreak = 0;
	if (prev.length > 0) state.lastFailSet = [...prev];
}

function repetitionAlert(state: VfeState): string | undefined {
	const tfs = state.belief?.target_files ?? [];
	if (tfs.length === 0 || state.stepLog.length === 0) return undefined;
	const W_RE = />|sed\s+-i|\btee\b|\bcp\b|\bmv\b|\bpython3?\b|\bgcc\b|\bclang\b|\bmake\b/;
	let rw = 0;
	for (const e of [...state.stepLog].reverse()) {
		if (tfs.some((t) => e.command.includes(t)) && W_RE.test(e.command)) rw++;
		else break;
	}
	if (rw >= 2 || state.sameTestStreak >= REPEAT_GATE) {
		return `[REPETITION ALERT] consecutive target rewrites=${rw}, same-test streak=${state.sameTestStreak}. Stop repeating the same fix; run a diagnostic that yields new information (isolate the first divergence / read the exact error line).`;
	}
	return undefined;
}

function diagnosticNote(state: VfeState): string | undefined {
	if (state.sameTestStreak < SAME_TEST_STALL || state.lastFailSet.length === 0) return undefined;
	return `[DIAGNOSTIC] Same failing tests for ${state.sameTestStreak} steps: ${state.lastFailSet.slice(0, 5).join(", ")}. The current rewrite is not changing outcomes. Localize the first divergence (compare artifact vs reference, read the exact assertion/error line) before another edit.`;
}

function commandLabel(turnTools: VfeState["turnTools"], selected?: VfeState["selectedSequence"]): string {
	if (selected) {
		return `chosen ${selected.id}(${selected.label}): ${selected.commands.join(" && ")}`.slice(0, 4000);
	}
	if (turnTools.length === 0) return "(no container tools this turn)";
	return turnTools
		.map((t) => {
			if (t.tool === "run") {
				try {
					const parsed: unknown = JSON.parse(t.input);
					if (parsed && typeof parsed === "object" && Array.isArray((parsed as { commands?: unknown }).commands)) {
						return (parsed as { commands: unknown[] }).commands.map(String).join(" && ");
					}
				} catch {
					// truncated input — fall through
				}
				return "run";
			}
			if (t.tool === "run_test") return "run_test";
			return `${t.tool} ${t.input.slice(0, 80)}`;
		})
		.join(", ")
		.slice(0, 4000);
}

function extractErrLocs(text: string): Array<[string, number]> {
	const out: Array<[string, number]> = [];
	const seen = new Set<string>();
	for (const m of text.matchAll(ERR_LOC_RE)) {
		const p = m[1] || m[3];
		const ln = m[2] || m[4];
		if (!p || !ln) continue;
		let path = p;
		if (path.startsWith("./")) path = path.slice(2);
		if (!path.startsWith("/")) path = `/${path}`;
		const key = `${path}:${ln}`;
		if (seen.has(key)) continue;
		seen.add(key);
		out.push([path, Number(ln)]);
	}
	return out;
}

async function readErrLines(exec: Exec, locs: Array<[string, number]>): Promise<Record<string, string>> {
	if (locs.length === 0) return {};
	const reads = locs.slice(0, 6).map(([ep, el]) => {
		const rp = ep.startsWith("/app") || ep.startsWith("/tests") ? ep : `/app${ep}`;
		return `echo '### ${rp}:${el} ###'; sed -n '${el},${el + 2}p' -- '${rp}' 2>/dev/null`;
	});
	let raw = "";
	try {
		const r = await exec(reads.join("\n"), 30);
		raw = r.output;
	} catch {
		return {};
	}
	const map: Record<string, string> = {};
	let cur: string | undefined;
	let lines: string[] = [];
	const flush = () => {
		if (cur) map[cur] = lines.join("\n").trim();
	};
	for (const ln of raw.split("\n")) {
		const m = /^### (.+):(\d+) ###$/.exec(ln.trim());
		if (m) {
			flush();
			cur = `${m[1]}:${m[2]}`;
			lines = [];
		} else if (cur !== undefined) {
			lines.push(ln.trim());
		}
	}
	flush();
	return map;
}

function buildErrBlock(pyout: string): string {
	const seen = new Set<string>();
	const lines: string[] = [];
	for (const bl of pyout.split("\n")) {
		if (!BUILD_ERR_LINE_RE.test(bl)) continue;
		const bs = bl.trim().slice(0, 220);
		if (bs && !seen.has(bs)) {
			seen.add(bs);
			lines.push(bs);
		}
		if (lines.length >= 8) break;
	}
	if (lines.length === 0) return "";
	return (
		`\n[BUILD-ERRORS] compile/link errors — FIX THESE FIRST (they are why all tests fail):\n` +
		lines.map((x) => `- ${x}`).join("\n")
	);
}

function trackErrLocStall(state: VfeState, errLines: Record<string, string>): string | undefined {
	const appNow: Record<string, string> = {};
	for (const [k, v] of Object.entries(errLines)) {
		if (k.startsWith("/app/")) appNow[k] = v;
	}
	if (Object.keys(appNow).length === 0) {
		state.prevErrLines = {};
		state.errUnchanged = 0;
		return undefined;
	}
	if (JSON.stringify(appNow) === JSON.stringify(state.prevErrLines)) {
		state.errUnchanged += 1;
	} else {
		state.errUnchanged = 0;
	}
	let note: string | undefined;
	if (state.errUnchanged >= ERR_LOC_STALL) {
		const stall =
			"[ERR-LOC-STALL] 报错位置行内容连续未变：你反复重写但从未修改报错所在行（见上方 [ERR-LOC]）。必须直接改报错行，而不是在别处微调。";
		if (!state.hardFailures.includes(stall)) state.hardFailures.push(stall);
		note = stall;
	}
	state.prevErrLines = appNow;
	return note;
}

function creditToolPractices(store: ExperienceStore | undefined, turnTools: VfeState["turnTools"]): void {
	if (!store) return;
	for (const t of turnTools) {
		// Infra failures are not the tool's fault — do not train practice credit on them.
		if (toolInfra(t)) continue;
		let cmd = t.input;
		if (t.tool === "run") {
			try {
				const parsed: unknown = JSON.parse(t.input);
				if (parsed && typeof parsed === "object" && Array.isArray((parsed as { commands?: unknown }).commands)) {
					for (const c of (parsed as { commands: unknown[] }).commands) {
						updateToolPractice(store, String(c), (t.exitCode ?? 0) === 0 && !t.isError);
					}
					continue;
				}
			} catch {
				// fall through
			}
		}
		if (t.tool === "run_test") cmd = "pytest /tests/test_outputs.py";
		else if (t.tool !== "run") cmd = `${t.tool} ${t.input.slice(0, 120)}`;
		updateToolPractice(store, cmd, (t.exitCode ?? 0) === 0 && !t.isError);
	}
}

/** Recon/read-only tools whose repetition can become a genuine dead end. */
const RECON_TOOLS = new Set(["read", "ls", "find", "grep"]);

function trackDeadFamilies(
	state: VfeState,
	turnTools: VfeState["turnTools"],
	improved: boolean,
	newFails: string[],
	missing: string[],
): void {
	if (!state.store) return;
	for (const t of turnTools) {
		if (toolInfra(t)) continue;
		// Only reconnaissance/read-only families are dead-end candidates. Repeating a
		// compile/edit/run is normal iteration (a necessary step in edit→build→test
		// tasks), not a dead end — branding it so is misleading.
		if (!RECON_TOOLS.has(t.tool)) continue;
		let target = "";
		try {
			const parsed: unknown = JSON.parse(t.input);
			if (parsed && typeof parsed === "object") {
				const o = parsed as { path?: unknown; pattern?: unknown };
				target = String(o.path ?? o.pattern ?? "");
			}
		} catch {
			// ignore
		}
		const fp = `${t.tool} ${target}`.replace(/\s+/g, " ").trim().slice(0, 40);
		if (!fp) continue;
		if (improved) state.deadFamilies[fp] = 0;
		else state.deadFamilies[fp] = (state.deadFamilies[fp] ?? 0) + 1;
		const runs = state.deadFamilies[fp] ?? 0;
		if (runs >= DEAD_THRESHOLD) {
			let cause = "推理死路（命令族反复无改善）";
			if (t.exitCode !== undefined && t.exitCode !== 0) cause = "工具执行失败（非零退出）";
			else if (newFails.length > 0) cause = "意外观测（出现新失败测试）";
			else if (missing.length > 0) cause = "环境不符（产物缺失）";
			const strong = `[DEAD-FAMILY] 【现象】命令族 \`${fp}\` 连续 ${runs} 次无改善 /【原因】${cause}/【建议】换一条路径或先读报错行，不要再重复同一命令族。`;
			if (!state.hardFailures.includes(strong)) state.hardFailures.push(strong);
		}
	}
}

async function reflect(
	ctx: ExtensionContext,
	state: VfeState,
	prediction: Record<string, unknown>,
	observation: Record<string, unknown>,
	obsTxt: string,
	improved: boolean,
	newFails: string[],
	fixedFails: string[],
	command: string,
	verdict?: StepVerdict,
	verdictEvidence?: string,
): Promise<StepLogEntry> {
	const signal = signalOf(improved, newFails, fixedFails, verdict, verdictEvidence);
	const resultLabel = improved
		? verdict === "better"
			? "改善(自评)"
			: "改善(事实)"
		: verdict === "worse"
			? "退步(自评)"
			: verdict === "same"
				? "持平(自评)"
				: "无变化(未自评)";
	const text = await completeText(
		ctx,
		REFLECT_SYSTEM,
		reflectUserPrompt({
			predicted: JSON.stringify(prediction).slice(0, 400),
			actual: JSON.stringify(observation).slice(0, 400),
			observation: obsTxt.slice(0, 800),
			resultLabel,
			verdictEvidence: verdictEvidence ? verdictEvidence.slice(0, 240) : undefined,
			newFails,
			fixedFails,
		}),
		200,
	);
	let summary = `评价=${improved ? "命中" : "偏离"}; 信号=${signal}`;
	const j = extractJson(text);
	if (j) {
		summary = `评价=${String(j.评价 ?? "?")}; 原因=${String(j.原因 ?? "")}; 经验=${String(j.经验 ?? "")}`;
	}
	return {
		step: state.steps,
		command: command.slice(0, 4000),
		prediction,
		observation,
		signal,
		summary,
		improved,
		new_fails: newFails,
		fixed_fails: fixedFails,
		...(verdict ? { verdict, ...(verdictEvidence ? { verdict_evidence: verdictEvidence } : {}) } : {}),
	};
}

async function reviseStates(ctx: ExtensionContext, state: VfeState, evidence: string): Promise<boolean> {
	const belief = state.belief;
	if (!belief || (!belief.objective && belief.todos.length === 0)) return false;
	if (state.steps - state.lastReviseStep < REVISE_EVERY) return false;
	const text = await completeText(
		ctx,
		REVISE_STATES_SYSTEM,
		reviseUserPrompt({
			skillMd: state.adaptiveSkill ?? "",
			todosJson: JSON.stringify(belief.todos).slice(0, 1500),
			objective: belief.objective,
			evidence: evidence.slice(0, 4000),
			fails: state.prevFails,
			fileState: state.fileStatePrev,
		}),
		1200,
	);
	const j = extractJson(text);
	if (!j) return false;
	const todos = normTodos(j.todos);
	if (todos.length === 0) return false;
	let objective = String(j.objective ?? "").trim();
	const ip = todos.find((t) => t.status === "in_progress");
	if (ip) objective = ip.task;
	else if (objective) {
		const hit = todos.find((t) => t.task === objective);
		if (hit) hit.status = "in_progress";
	}
	const changed =
		JSON.stringify(todos) !== JSON.stringify(belief.todos) || (objective && objective !== belief.objective);
	if (!changed) return false;
	if (objective && objective !== belief.objective) {
		belief.history.push(belief.objective);
	}
	belief.todos = todos;
	if (objective) belief.objective = objective;
	if (Array.isArray(j.target_files)) {
		const tfs = j.target_files.map(String).filter((p) => p.startsWith("/"));
		if (tfs.length > 0) belief.target_files = tfs;
	}
	const ss = String(j.skill_summary ?? "").trim();
	if (ss) belief.skill_summary = ss;
	state.objective = belief.objective;
	state.lastReviseStep = state.steps;
	return true;
}

async function dumpStateFiles(exec: Exec, state: VfeState): Promise<void> {
	const belief = state.belief;
	const files: Record<string, string> = {
		"/tmp/vfe_local_memory.json": JSON.stringify(
			{
				objective: belief?.objective ?? "",
				todos: belief?.todos ?? [],
				target_files: belief?.target_files ?? [],
				history: belief?.history ?? [],
				env_constraints: state.envConstraints,
				failures: state.hardFailures,
				recent_steps: state.stepLog.slice(-10).map((e) => ({
					step: e.step,
					command: e.command.slice(0, 160),
					improved: e.improved,
					signal: e.signal,
				})),
			},
			null,
			1,
		).slice(0, 20000),
		"/tmp/vfe_objectives.txt": (belief?.history ?? [])
			.slice(-8)
			.map((h) => `- ${h}`)
			.join("\n"),
		"/tmp/vfe_failures.txt": state.hardFailures
			.slice(-8)
			.map((f) => `- ${f}`)
			.join("\n"),
		"/tmp/vfe_env_constraints.txt": state.envConstraints.map((c) => `- [${c.title}] ${c.text}`).join("\n"),
		"/tmp/vfe_adaptive_skill.md": (state.adaptiveSkill ?? "").slice(0, 12000),
		"/tmp/vfe_diary.txt": state.stepLog
			.slice(-12)
			.map((e) => `Step ${e.step}: ${e.command.slice(0, 160)} | ${e.signal}${e.improved ? " [improved]" : ""}`)
			.join("\n")
			.slice(0, 8000),
		"/tmp/vfe_environment.txt": (state.envObs ?? "").slice(0, 4000),
		"/tmp/vfe_last_output.txt": state.turnTools
			.map((t) => `$ ${t.tool}\n${t.output}`)
			.join("\n")
			.slice(0, 20000),
	};
	const payload = Buffer.from(JSON.stringify(files), "utf8").toString("base64");
	try {
		await exec(
			`python3 -c "import base64,json; d=json.loads(base64.b64decode('${payload}')); [open(p,'w').write(c) for p,c in d.items()]"`,
			30,
		);
	} catch {
		// dump is best-effort
	}
}

function appendTrajectory(state: VfeState, row: Record<string, unknown>): void {
	if (!state.task) return;
	const dir = runDirFor(state.task, state.rep ?? 0);
	try {
		mkdirSync(dir, { recursive: true });
		appendFileSync(join(dir, "trajectory.jsonl"), `${JSON.stringify(row)}\n`, "utf8");
	} catch {
		// best-effort
	}
}

function writeResult(state: VfeState, result: Record<string, unknown>): void {
	if (!state.task) return;
	const dir = runDirFor(state.task, state.rep ?? 0);
	try {
		mkdirSync(dir, { recursive: true });
		writeFileSync(join(dir, "result.json"), JSON.stringify(result, null, 2), "utf8");
	} catch {
		// best-effort
	}
}

function readTrajectory(runDir: string): Array<Record<string, unknown>> {
	try {
		const text = readFileSync(join(runDir, "trajectory.jsonl"), "utf8");
		const rows: Array<Record<string, unknown>> = [];
		for (const line of text.split("\n")) {
			if (!line.trim()) continue;
			try {
				const parsed: unknown = JSON.parse(line);
				if (parsed && typeof parsed === "object") rows.push(parsed as Record<string, unknown>);
			} catch {
				// skip bad line
			}
		}
		return rows;
	} catch {
		return [];
	}
}

async function endOfRunSediment(ctx: ExtensionContext, state: VfeState, runDir: string, reward: number): Promise<void> {
	const store = state.store;
	if (!store) return;
	const rows = readTrajectory(runDir);
	const phaseEnd = state.steps + 1;
	const objective = state.belief?.objective ?? state.objective ?? "";
	try {
		await evalMethodsPhase(ctx, store, rows, state.phaseStartStep, phaseEnd, objective);
		await evalConstraintsPhase(ctx, store, rows, state.phaseStartStep, phaseEnd, objective);
		await extractLessons(ctx, store, state.instruction ?? state.task ?? "", rows, reward);
		extractBestPractice(store, rows, state.phaseLog);
		const local: LocalMemoryInput = {
			envConstraints: state.envConstraints,
			failures: state.hardFailures,
			stepLog: state.stepLog,
		};
		await promoteLocalFull(ctx, store, local);
		markConf(store);
		dumpInjected(store, state.task, state.rep, "final");
		saveExperiences(store.experiences, store.nextId, store.path);
	} catch (err) {
		console.error(`[WARN] end-of-run sediment failed: ${String(err)}`);
		try {
			saveExperiences(store.experiences, store.nextId, store.path);
		} catch {
			// ignore
		}
	}
}

/**
 * One settle after a step agent turn: pytest truth, improved signal, reflect,
 * rollback, revise, trajectory, memory sediment, optional next-step followUp.
 *
 * Domain split: when turn tools or pytest report infra (container dead), call
 * `recoverInfra` once. Success → freeze counters/objective this step. Failure →
 * immediate ENV-IRRECOVERABLE abort (no 6-step 0/0 wait).
 */
export async function settleStep(
	pi: ExtensionAPI,
	ctx: ExtensionContext,
	state: VfeState,
	exec: Exec,
	execTests: Exec,
	maxSteps = MAX_STEPS_DEFAULT,
	recoverInfra?: InfraRecovery,
	verifySandbox?: VerifySandbox,
	execOfficialTests?: Exec,
): Promise<void> {
	if (state.phase !== "step") return;
	state.steps += 1;
	const step = state.steps;
	const turnTools = [...state.turnTools];
	state.turnTools = [];
	const selected = state.selectedSequence;
	state.selectedSequence = undefined;
	const command = commandLabel(turnTools, selected);
	const obsTxt = turnTools
		.map((t) => `$ ${t.tool}\n${t.output}`)
		.join("\n")
		.slice(0, 4000);
	const runDir = runDirOf(state);

	// Domain split: infra faults from this turn's tool records and/or pytest.
	const turnInfra = turnTools.some(toolInfra);
	const pyTimeout = verifierTimeoutSec(state.task);
	let py = await runPytest(exec, execTests, runDir, pyTimeout, state.task);
	let infraUnresolved = turnInfra || py.infra;
	let recovered = false;

	if (infraUnresolved) {
		state.infraStreak += 1;
		if (recoverInfra) {
			try {
				const mode = await recoverInfra();
				recovered = true;
				ctx.ui.notify(`Infra fault at step ${step}; recovery mode=${mode}`, "warning");
				py = await runPytest(exec, execTests, runDir, pyTimeout, state.task);
				infraUnresolved = py.infra;
			} catch (err) {
				const msg = err instanceof Error ? err.message : String(err);
				ctx.ui.notify(`Infra recovery failed at step ${step}: ${msg}`, "error");
				infraUnresolved = true;
				recovered = false;
			}
		} else {
			infraUnresolved = true;
		}
	} else {
		state.infraStreak = 0;
	}

	// Unrecoverable infra (recovery failed or still dead) or repeated faults → abort this step.
	// Successful recovery continues with frozen counters (infraStep below).
	const infraAbort = infraUnresolved || state.infraStreak >= ENV_INFRA_MAX;
	if (infraAbort) {
		const reason = infraUnresolved
			? recovered
				? `[ENV-IRRECOVERABLE] container still dead after recovery at step ${step}.`
				: `[ENV-IRRECOVERABLE] container infra fault; recovery failed at step ${step}.`
			: `[ENV-IRRECOVERABLE] ${state.infraStreak} consecutive steps with container infra faults.`;
		state.abortNote = reason;
		state.phase = "end";
		state.awaitingStepTurn = false;
		state.pendingSettle = false;
		const observation = {
			execution: ["infra_fault"],
			tests: "env_unavailable",
			failed_tests: [],
			missing_artifacts: [],
		};
		appendTrajectory(state, {
			task: state.task,
			rep: state.rep,
			step,
			command,
			batch: turnTools.map((t) => t.tool),
			prediction: {},
			observation,
			improved: false,
			fails: [],
			objective: state.belief?.objective ?? "",
			target_files: state.belief?.target_files ?? [],
			outputs: turnTools.map((t) => t.output.slice(0, 40000)),
			verdict: null,
			verdict_evidence: null,
			rollback: null,
			infra: true,
			infra_mode: infraUnresolved ? (recovered ? "recovered_but_still_dead" : "unrecovered") : "streak",
			ts: new Date().toISOString(),
		});
		const result = {
			task: state.task,
			rep: state.rep,
			steps: step,
			rollback_count: state.rollbackCount,
			passed: py.passed,
			failed: py.failed,
			early_stop: false,
			stall_abort: false,
			abort: reason,
			objective: state.belief?.objective ?? "",
			verified_by: "none",
			reward: 0,
		};
		writeResult(state, result);
		await endOfRunSediment(ctx, state, runDir, 0);
		pi.setActiveTools([]);
		ctx.ui.notify(`VFE end at step ${step}: ${reason}`, "warning");
		pi.sendUserMessage(
			[
				`VFE run finished (phase=end, step ${step}/${maxSteps}).`,
				reason,
				"Container infrastructure failed and could not be recovered. Do not continue mutating /app.",
			].join("\n"),
			{ deliverAs: "followUp" },
		);
		return;
	}

	// Recovered path: real pytest after restart — freeze counters, skip revise this step.
	const infraStep = recovered && !infraUnresolved;
	// A provider error (503 / no_available_workers) ends the loop before the agent
	// could act. /app is unchanged, so every streak counter must hold still — the
	// same freeze as an infra step, otherwise four 503s read as EMPTY-TURN-ABORT.
	const providerFault = state.turnProviderFault && turnTools.length === 0;
	const faultStep = infraStep || providerFault;
	if (infraStep) {
		ctx.ui.notify(`Container recovered at step ${step}; objective frozen for this settle.`, "info");
	}

	const targets = state.belief?.target_files ?? [];
	const fileState = await snapshotTargets(exec, targets);
	if (!faultStep) trackSameTestStreak(state, py.fails);
	const newFails = [...py.fails.filter((f) => !state.prevFails.includes(f))].sort();
	const fixedFails = [...state.prevFails.filter((f) => !py.fails.includes(f))].sort();
	const touched = fileTouched(fileState, state.fileStatePrev);
	const assertion = parseAssertion(py.raw);
	let assertionDelta: number | undefined;
	if (!faultStep) {
		if (assertion) {
			state.assertionStreak = state.prevAssertionText === assertion.text ? state.assertionStreak + 1 : 1;
			if (
				state.prevAssertionSignature === assertion.signature &&
				state.prevAssertionValue !== undefined &&
				assertion.value !== undefined
			) {
				// Display-only numeric delta. Direction is never inferred from the
				// message shape — the agent reads the assertion and self-assesses.
				assertionDelta = assertion.value - state.prevAssertionValue;
			}
			state.prevAssertionText = assertion.text;
			state.prevAssertionValue = assertion.value;
			state.prevAssertionSignature = assertion.signature;
		} else {
			state.assertionStreak = 0;
			state.prevAssertionText = undefined;
			state.prevAssertionValue = undefined;
			state.prevAssertionSignature = undefined;
		}
	}
	// Self-assessed verdict for THIS step, submitted via assess_step at the end
	// of the same turn. Consume and clear it here so a later settle can never
	// inherit a stale one.
	const verdict = state.turnVerdict;
	const verdictEvidence = state.turnVerdictEvidence;
	state.turnVerdict = undefined;
	state.turnVerdictEvidence = undefined;
	// Pure-objective stagnation: identical failing-test set AND unchanged missing
	// outputs across consecutive tool-active settles. Computed before prevFails/
	// missingPrev are overwritten below.
	const factStagnant =
		!faultStep &&
		turnTools.length > 0 &&
		sameFailSet(state.prevFails, py.fails) &&
		py.missing.length === state.missingPrev.length &&
		[...py.missing].sort().join("\n") === [...state.missingPrev].sort().join("\n");
	let improved =
		taskImproved({
			failsNow: py.fails,
			failsPrev: state.prevFails,
			missingNow: py.missing,
			missingPrev: state.missingPrev,
		}) || verdict === "better";
	const buildFailed = BUILD_FAIL_RE.test(py.raw);
	// Test-list facts override a self-assessed "better" (fact lines take precedence).
	if (improved && newFails.length > 0) improved = false;
	if (improved && buildFailed) improved = false;
	if (improved && turnTools.some((t) => (t.exitCode ?? 0) >= 128)) improved = false;
	if (improved && py.fails.length === 0 && py.passed === 0) improved = false;
	if (faultStep) improved = false;
	state.fileStatePrev = fileState;
	state.missingPrev = [...py.missing];
	if (faultStep) {
		// Freeze improvement counters on infra-recovered/provider-fault steps
		// (neither is agent progress).
	} else {
		if (improved) {
			state.nImproved += 1;
			state.noImproveStreak = 0;
		} else if (turnTools.length > 0) {
			state.noImproveStreak += 1;
		}
		// STALL guidance streak: reset whenever THIS settle improved (facts or
		// assessment), accumulate on same/worse, and stay unchanged when no
		// verdict was submitted (facts-only turn applies no change-approach
		// pressure).
		if (improved) state.verdictSameStreak = 0;
		else if (verdict) state.verdictSameStreak += 1;
		// STALL-ABORT basis: objective stagnation. An evidence-backed "better"
		// assessment is not stagnation and rescues the run; same/worse/missing
		// verdicts let the factual counter run toward STALL_ABORT.
		state.factStagnationStreak = factStagnant && verdict !== "better" ? state.factStagnationStreak + 1 : 0;
	}
	if (py.fails.length > 0) {
		state.failureHistory.push(`Step ${step}: ${py.fails.slice(0, 3).join(", ")}`);
		if (state.failureHistory.length > 10) {
			state.failureHistory.splice(0, state.failureHistory.length - 10);
		}
	}
	state.prevFails = [...py.fails];

	// Tool-practice credit + dead-family hard facts + ERR-LOC stall (skip on infra step).
	if (!infraStep) {
		creditToolPractices(state.store, turnTools);
		trackDeadFamilies(state, turnTools, improved, newFails, py.missing);
	}
	const errLocs = extractErrLocs(py.raw);
	const errLineTxt =
		errLocs.length > 0
			? errLocs
					.slice(0, 6)
					.map(([p, l]) => `[ERR-LOC] ${p}:${l}`)
					.join("\n")
			: "";
	const errLines = await readErrLines(exec, errLocs);
	const errStallNote = trackErrLocStall(state, errLines);
	const errRender =
		Object.keys(errLines).length > 0
			? Object.entries(errLines)
					.map(([k, v]) => `[ERR-LOC] ${k} -> ${v || "(empty)"}`)
					.join("\n")
			: errLineTxt;
	const buildErrTxt = buildFailed ? buildErrBlock(py.raw) : "";

	const prediction: Record<string, unknown> = {};
	// After successful recovery, py is real post-restart results — use them;
	// only mark tests=env_unavailable when the run is aborting with dead container
	// (handled above). Here py is healthy on the continue path.
	const testsLabel =
		!py.fails.length && py.passed > 0
			? "all_passed"
			: py.passed === 0 && py.failed === 0
				? "no_tests"
				: "failures_present";
	const observation = {
		execution: [
			infraStep
				? "infra_recovered"
				: providerFault
					? "provider_error"
					: turnTools.every((t) => !t.isError)
						? "completed"
						: "command_failure",
		],
		tests: testsLabel,
		failed_tests: py.fails,
		missing_artifacts: py.missing,
		...(infraStep ? { infra: true } : {}),
	};

	// Snapshot / rollback policy. Two independent trees with separate jobs:
	//   /tmp/.app_best  — rollback baseline (facts only): refreshed whenever the
	//                     fail set did not grow, protective rollback when it did.
	//   /tmp/.app_qbest — qualitative best: written ONLY when the agent declares
	//                     better (and facts do not override it), so a same/missing
	//                     refresh of .app_best can never clobber it.
	let note: string | undefined;
	let qNote: string | undefined;
	const failsGrew = py.fails.length > 0 && py.fails.some((f) => !state.bestFails.includes(f));
	if (infraStep) {
		note =
			"\n[INFRA RECOVERED] the container was restarted mid-settle; progress counters and objective were frozen for this step. Re-verify tests before the next edit.";
	} else if (providerFault) {
		// No tool call ran, so /app is untouched: skip snapshot/rollback entirely
		// (a no-op copy would still bump bestFails and mask a later real regression).
		note = "\n[PROVIDER ERROR] the previous turn ended in a model/provider error before any tool call executed. Nothing was changed; just proceed with the same objective.";
	} else if (buildFailed) {
		note =
			"\n[BUILD FAILED] your edit did NOT compile/link and was NOT reverted (kept so you can fix it in place). Fix the compile/link error FIRST, then re-check the tests.";
	} else if (state.bestFails.length === 0 || !failsGrew) {
		state.bestFails = [...py.fails];
		await exec("rm -rf /tmp/.app_best && cp -a /app /tmp/.app_best && echo BEST_OK", 180).catch(() => undefined);
	} else {
		// Stage then swap so a failed copy cannot delete /app.
		const r = await exec(
			"rm -rf /app.restore && cp -a /tmp/.app_best /app.restore && rm -rf /app && mv /app.restore /app && echo RB_OK",
			180,
		).catch(() => ({
			output: "",
			exitCode: 1,
			infra: false,
		}));
		if (r.output.includes("RB_OK")) {
			state.rollbackCount += 1;
			note =
				"\n[SYSTEM ROLLBACK] your previous action was REVERTED because it introduced new failing tests; the previous /app state was restored.";
		}
	}
	if (!infraStep && verdict === "better" && improved) {
		state.bestVerdict = { step, evidence: verdictEvidence ?? "" };
		await exec("rm -rf /tmp/.app_qbest && cp -a /app /tmp/.app_qbest && echo QBEST_OK", 180).catch(() => undefined);
	} else if (!infraStep && verdict === "worse" && state.bestVerdict) {
		// Never refresh the qualitative best on a self-assessed regression: the
		// previous best tree must stay restorable. Point the agent back at it.
		qNote = `\n[BELOW-BEST] step ${state.bestVerdict.step} you assessed better ("${state.bestVerdict.evidence}"); this step you assessed worse ("${verdictEvidence ?? ""}"). Diff the current artifact against /tmp/.app_qbest (your best tree) and restore what worked before further edits.`;
	}

	const reflected = await reflect(
		ctx,
		state,
		prediction,
		observation,
		`${obsTxt}\nvalidator=${py.output}`,
		improved,
		newFails,
		fixedFails,
		command,
		verdict,
		verdictEvidence,
	);
	const strategyNotes = [diagnosticNote(state), repetitionAlert(state)].filter(Boolean);
	const notes = [reflected.note, note, qNote, errStallNote, ...strategyNotes].filter(Boolean).join("\n");
	if (notes) reflected.note = notes.trim();
	reflected.rawObservation = obsTxt.slice(0, 1500);
	state.stepLog.push(reflected);
	if (state.stepLog.length > 20) state.stepLog.splice(0, state.stepLog.length - 20);

	// Objective / todo revise (every REVISE_EVERY steps, or stall re-plan).
	// Freeze objective on infra-recovered steps: fault evidence must not mutate belief.
	const diary = turnTools
		.filter((t) => !toolInfra(t))
		.map((t) => `$ ${t.tool}\n${t.output}`)
		.join("\n")
		.slice(0, 1500);
	const fb = feedbackHint(state);
	const evidence = `${diary}\n${fb.signal ? `STEP OUTCOME: ${fb.signal}\n${fb.summary}` : ""}`;
	const oldObjective = state.belief?.objective ?? "";
	if (!faultStep && state.steps - state.lastReviseStep >= REVISE_EVERY) {
		await reviseStates(ctx, state, evidence);
	}
	const stallNow = Math.max(0, state.steps - state.nImproved);
	if (
		!faultStep &&
		stallNow > 0 &&
		stallNow % REPLAN_STALL === 0 &&
		state.steps - state.lastReviseStep >= REVISE_EVERY
	) {
		await reviseStates(ctx, state, evidence);
	}
	// Phase log: record objective boundaries for best-practice extraction + mid-run eval.
	const newObj = state.belief?.objective ?? "";
	if (newObj !== oldObjective) {
		state.phaseLog.push({ step: state.phaseStartStep, objective: oldObjective });
		state.phaseStartStep = step;
		if (state.store) {
			try {
				await evalMethodsPhase(
					ctx,
					state.store,
					readTrajectory(runDir),
					state.phaseLog.at(-1)?.step ?? 0,
					step,
					oldObjective,
				);
			} catch {
				// best-effort mid-run eval
			}
		}
	}
	if (!state.phaseLog.some((p) => p.step === 0 && p.objective === (state.belief?.objective ?? ""))) {
		// ensure phaseLog has a root entry for first objective
		if (state.phaseLog.length === 0) {
			state.phaseLog.push({ step: 0, objective: newObj });
		}
	}

	await dumpStateFiles(exec, state);

	appendTrajectory(state, {
		task: state.task,
		rep: state.rep,
		step,
		command,
		batch: turnTools.map((t) => t.tool),
		prediction,
		observation,
		improved,
		touched,
		assertion: assertion?.text ?? null,
		fails: py.fails,
		objective: newObj,
		target_files: state.belief?.target_files ?? [],
		outputs: turnTools.map((t) => t.output.slice(0, 40000)),
		verdict: verdict ?? null,
		verdict_evidence: verdictEvidence ?? null,
		rollback: note ?? null,
		...(infraStep ? { infra: true } : {}),
		ts: new Date().toISOString(),
	});

	// Periodic memory sediment (crash loses ≤ MEM_SAVE_EVERY steps).
	if (state.store) maybePeriodicSave(state.store, step);

	// Hard-fact diary extras → feedback note path (BUILD/ERR-LOC shown to agent).
	const hardExtra = [buildErrTxt, errRender].filter(Boolean).join("\n");

	// Degenerate-turn detection: a turn that executed no tool call (e.g. the model
	// wrote the tool invocation as plain text). Two in a row → stop burning budget.
	// A provider fault never reached tool execution, so it must not advance this.
	if (!faultStep) state.emptyTurnStreak = turnTools.length === 0 ? state.emptyTurnStreak + 1 : 0;

	// Early stop / aborts. faultStep freezes envNoTests/noImprove (the fault is not agent progress).
	// A resume window (official-0 contradiction) keeps the run alive: in-loop pass
	// must not end the run until the window expires, or the official failures fed
	// back at resume would get exactly one turn before the run terminates.
	const inResumeWindow = state.resumeWindowEnd !== undefined && step < state.resumeWindowEnd;
	if (!py.fails.length && py.passed > 0 && !inResumeWindow) {
		state.earlyStop = true;
		state.phase = "end";
	} else if (faultStep) {
		// infra-recovered or provider-fault step: leave streaks as-is
	} else if (state.emptyTurnStreak >= EMPTY_TURN_ABORT) {
		state.abortNote = `[EMPTY-TURN-ABORT] ${state.emptyTurnStreak} consecutive turns with no executed tool call (tool calls written as text do not execute).`;
		state.phase = "end";
	} else if (state.factStagnationStreak >= STALL_ABORT) {
		state.stallAbort = true;
		state.abortNote = `[STALL-ABORT] ${state.factStagnationStreak} consecutive steps with an identical failing-test set and unchanged outputs → early stop.`;
		state.phase = "end";
	} else if (py.passed === 0 && py.failed === 0) {
		// Pure exec/pytest timeout is not a collection fault — do not count toward ENV_NO_TESTS.
		const pureTimeout = /^\[TIMEOUT after \d+s\]/.test(py.raw.trim()) && !/\d+\s+(passed|failed|error)/.test(py.raw);
		if (pureTimeout) {
			// leave envNoTests unchanged; noImproveStreak already advanced above
		} else {
			const next = state.envNoTests + 1;
			state.envNoTests = next;
			if (next >= ENV_NO_TESTS_MAX) {
				state.abortNote = `[ENV-FAULT-ABORT] ${next} consecutive steps with 0/0 tests (collection fault).`;
				state.phase = "end";
			}
		}
	} else {
		state.envNoTests = 0;
	}

	// Step budget grows with each official resume so a contradiction at maxSteps
	// still gets OFFICIAL_RESUME_STEPS more turns. Invariant: resumeWindowEnd
	// <= stepBudget at all times (a window starts at the end step of the previous
	// round, which is <= its budget), so window expiry always fires first.
	const stepBudget = maxSteps + OFFICIAL_RESUME_STEPS * state.officialResumes;
	if (step >= stepBudget && state.phase === "step") {
		state.phase = "end";
		state.abortNote = state.abortNote ?? `[MAX-STEPS] reached ${stepBudget} steps.`;
	}

	state.awaitingStepTurn = false;
	state.pendingSettle = false;

	if (state.phase === "end") {
		// Pre-official snapshot of the agent-final /app: official test.sh writes
		// into /app (headless vim.txt, compiled binaries), so a fresh re-verify
		// must capture the tree before the official run mutates it. Take it
		// whenever a fresh re-verify is possible, regardless of in-loop results —
		// the official 0-vs-inloop contradiction case needs it most.
		const inloopPass = !py.fails.length && py.passed > 0;
		if (verifySandbox) {
			await exec("rm -rf /tmp/.app_final && cp -a /app /tmp/.app_final && echo APP_FINAL_OK", 300).catch(
				() => undefined,
			);
			// Snapshot the agent environment as the verify image BEFORE the official
			// run: the official suite writes real-fs state (/server, /app/vim.txt)
			// that must not leak into a fresh re-verify (its mkdir /server is not
			// idempotent), while the agent's system-python packages (pexpect — not
			// installed by test.sh) must survive there. Best-effort: start_verify
			// falls back to committing at start or to the pristine image.
			if (verifySandbox.prepare) {
				await verifySandbox.prepare().catch(() => undefined);
			}
		}
		// Authoritative reward comes from the official verifier, not the in-loop
		// pytest. The official suite runs on the real rootfs (no overlay); a skip
		// (network/uv failure) is logged and surfaced via official_skip.
		const officialExec = execOfficialTests ?? execTests;
		const officialRun = await runOfficialVerifier(
			exec,
			officialExec,
			state.task,
			runDir,
			verifierTimeoutSec(state.task),
		);
		const official = officialRun.ok ? officialRun.outcome : undefined;
		const officialSkip = officialRun.ok ? undefined : officialRun.reason;
		let reward = inloopPass ? 1 : 0;
		let finalPassed = py.passed;
		let finalFailed = py.failed;
		let verifiedBy = official ? "official" : "inloop";
		if (official) {
			reward = official.reward;
			finalPassed = official.passed;
			finalFailed = official.failed;
			// The official suite disagrees with an in-loop all-pass: not a real early stop.
			if (official.reward === 0) state.earlyStop = false;
		}
		// Official 0: re-verify in a fresh container (committed env, agent-final
		// /app) in case the in-container run was polluted; only a fresh pass
		// overrides the 0. Gate on the official verdict alone — the in-loop pytest
		// is a feedback signal, not a verifier precondition.
		let freshVerify: { reward: number; passed: number; failed: number } | undefined;
		if (official && official.reward === 0 && verifySandbox) {
			const fresh = await runFreshVerifier(verifySandbox, state.task, runDir, verifierTimeoutSec(state.task));
			if (fresh) {
				freshVerify = { reward: fresh.reward, passed: fresh.passed, failed: fresh.failed };
				if (fresh.reward === 1) {
					reward = 1;
					finalPassed = fresh.passed;
					finalFailed = fresh.failed;
					verifiedBy = "official_fresh";
					state.earlyStop = true;
				}
			}
		}
		// Official 0 vs in-loop all-pass: a contradiction, not a verdict. Hand the
		// official failures back to the agent and open a bounded resume window
		// instead of writing result.json. Gated on a clean early stop (no abort
		// note): a degenerate/STALL run during the window ends for real, and the
		// fresh pass above already short-circuits a genuine success.
		const canResume =
			official !== undefined &&
			official.reward === 0 &&
			reward === 0 &&
			inloopPass &&
			state.abortNote === undefined &&
			state.officialResumes < OFFICIAL_RESUME_MAX;
		if (canResume && official) {
			state.officialResumes += 1;
			state.resumeWindowEnd = step + OFFICIAL_RESUME_STEPS;
			state.earlyStop = false;
			state.phase = "step";
			// test.sh wrote into /app (real fs): hand the agent its pre-official tree back.
			await exec(
				"if [ -d /tmp/.app_final ]; then rm -rf /app.restore && cp -a /tmp/.app_final /app.restore && rm -rf /app && mv /app.restore /app && echo RESTORE_OK; else echo NO_SNAPSHOT; fi",
				300,
			).catch(() => undefined);
			state.failureHistory.push(`Step ${step}: OFFICIAL failed ${official.fails.slice(0, 5).join(", ") || "(raw)"}`);
			if (state.failureHistory.length > 10) {
				state.failureHistory.splice(0, state.failureHistory.length - 10);
			}
			state.resumeHint =
				`[OFFICIAL VERIFIER FAILED] in-loop passes but official reports reward=0 ` +
				`(resume ${state.officialResumes}/${OFFICIAL_RESUME_MAX}); official failed: ${
					official.fails.slice(0, 5).join(", ") || "(raw)"
				}.`;
			ctx.ui.notify(
				`Official 0 vs in-loop pass at step ${step}; resume ${state.officialResumes}/${OFFICIAL_RESUME_MAX} (window ends step ${state.resumeWindowEnd}).`,
				"warning",
			);
			state.awaitingStepTurn = true;
			pi.sendUserMessage(
				[
					`[OFFICIAL VERIFIER FAILED — resume ${state.officialResumes}/${OFFICIAL_RESUME_MAX}] Your in-loop suite passes, but the official /tests/test.sh reports reward=0. The official verdict is the only one that counts.`,
					`Official failed tests: ${official.fails.slice(0, 20).join(", ") || "(not parsed from output)"}`,
					`Official output tail:\n${official.raw.slice(-3000)}`,
					`You have ${OFFICIAL_RESUME_STEPS} more steps (until step ${state.resumeWindowEnd}) before this window closes. /app was restored to your pre-official state. Fix what the official suite rejects; do not stop at in-loop green.`,
				].join("\n"),
				{ deliverAs: "followUp" },
			);
			return;
		}
		const result = {
			task: state.task,
			rep: state.rep,
			steps: step,
			rollback_count: state.rollbackCount,
			passed: finalPassed,
			failed: finalFailed,
			early_stop: state.earlyStop,
			stall_abort: state.stallAbort,
			abort: state.abortNote ?? null,
			objective: state.belief?.objective ?? "",
			// Provenance: an official pass, a fresh-container pass and a harness
			// fallback pass must not be indistinguishable in the result record.
			verified_by: verifiedBy,
			reward,
			...(state.officialResumes > 0 ? { official_resumes: state.officialResumes } : {}),
			...(freshVerify ? { fresh_verify: freshVerify } : {}),
			...(officialSkip ? { official_skip: officialSkip } : {}),
		};
		writeResult(state, result);
		await endOfRunSediment(ctx, state, runDir, reward);
		pi.setActiveTools([]);
		ctx.ui.notify(
			`VFE end at step ${step}: reward=${result.reward} passed=${finalPassed} failed=${finalFailed} official=${official ? "yes" : "no"} fresh=${freshVerify ? freshVerify.reward : "-"} rollbacks=${state.rollbackCount} store=${state.store?.experiences.size ?? 0}`,
			state.earlyStop ? "info" : "warning",
		);
		pi.sendUserMessage(
			[
				`VFE run finished (phase=end, step ${step}/${stepBudget}).`,
				state.abortNote ?? "",
				`Final tests: passed=${finalPassed} failed=${finalFailed}`,
				state.earlyStop ? "All tests passed — early stop." : "",
				"You may inspect /tmp/vfe_* artifacts or the host run dir. Do not continue mutating /app unless asked.",
			]
				.filter(Boolean)
				.join("\n"),
			{ deliverAs: "followUp" },
		);
		return;
	}

	// Next step kickoff.
	const { signal, summary, note: n2 } = feedbackHint(state);
	const stall = state.verdictSameStreak;
	const practiceQuery = `${state.belief?.objective ?? ""} ${(state.belief?.target_files ?? []).join(" ")}`;
	const practiceHits = state.store ? maturePracticeHits(state.store, practiceQuery) : [];
	const emptyTurnNudge =
		turnTools.length === 0 && !providerFault
			? "[NO TOOL CALL] Your last message produced no executed tool call — a tool call written as plain text does NOT execute. Submit tool calls as real function calls and start this turn with propose_sequences (2-3 candidates)."
			: "";
	const assessNudge =
		!infraStep && turnTools.length > 0 && !verdict
			? "[NO ASSESSMENT] you ended the last turn without calling assess_step, so STEP OUTCOME above is facts-only. This turn: make assess_step {verdict, evidence} your final tool call, after the last observation and before ending."
			: "";
	const resumeNudge =
		state.resumeWindowEnd !== undefined && step < state.resumeWindowEnd && state.resumeHint
			? `${state.resumeHint} Window closes at step ${state.resumeWindowEnd}.`
			: "";
	const worldPaths = [...new Set([...(state.belief?.target_files ?? []), ...inferConsumed([command])])];
	state.worldState = await collectWorldState(exec, worldPaths);
	const worldText = renderWorldState(state.worldState);
	const feedback = buildFeedbackPrompt({
		step,
		maxSteps: stepBudget,
		signal,
		summary,
		note: n2,
		objective: state.belief?.objective ?? "",
		targetFiles: state.belief?.target_files ?? [],
		fails: py.fails,
		passed: py.passed,
		failed: py.failed,
		improved,
		assertionText: assertion?.text,
		assertionDelta,
		assertionStreak: state.assertionStreak,
		stall,
		hint: [note?.startsWith("\n[SYSTEM ROLLBACK]") ? note : "", emptyTurnNudge, assessNudge, resumeNudge]
			.filter(Boolean)
			.join("\n"),
		skillCorrections: state.belief?.skill_corrections ?? [],
		requiredOutputs: py.missing,
		failureHistory: state.failureHistory,
		practiceHits,
		worldState: worldText,
	});
	// Attach BUILD/ERR-LOC hard facts so the agent sees them next step.
	const nextMsg = hardExtra ? `${feedback}\n\n${hardExtra}` : feedback;
	state.awaitingStepTurn = true;
	pi.sendUserMessage(nextMsg, { deliverAs: "followUp" });
}
