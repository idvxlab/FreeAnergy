import { type ChildProcessWithoutNullStreams, spawn } from "node:child_process";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import type { ExecResult } from "./types.ts";

interface SidecarReply {
	id: number;
	ok: boolean;
	container?: string;
	task?: string;
	output?: string;
	exit_code?: number;
	infra?: boolean;
	mode?: string;
	error?: string;
	traceback?: string;
}

const here = dirname(fileURLToPath(import.meta.url));
const SIDECAR = join(here, "docker_sidecar.py");

/**
 * Docker client/daemon/OCI errors → infra. Narrow on purpose: task output may
 * contain phrases like "is not running" without the container being dead.
 */
const INFRA_OUT_RE =
	/OCI runtime exec failed|Error response from daemon|Cannot connect to the Docker daemon|No such container:/i;

function classifyInfra(output: string): boolean {
	return INFRA_OUT_RE.test(output);
}

/** Long-lived JSONL client for docker_sidecar.py (thin shell over DockerEnv). */
export class DockerClient {
	private proc?: ChildProcessWithoutNullStreams;
	private nextId = 1;
	private pending = new Map<number, { resolve: (r: SidecarReply) => void; reject: (e: Error) => void }>();
	private buf = "";
	private stderrBuf = "";

	start(): void {
		if (this.proc) return;
		const python = process.env.VFE_PYTHON ?? "python3";
		this.proc = spawn(python, [SIDECAR], { stdio: ["pipe", "pipe", "pipe"] });
		this.proc.stdout.setEncoding("utf8");
		this.proc.stdout.on("data", (chunk: string) => this.onStdout(chunk));
		this.proc.stderr.setEncoding("utf8");
		this.proc.stderr.on("data", (chunk: string) => {
			this.stderrBuf = (this.stderrBuf + chunk).slice(-4000);
		});
		this.proc.on("exit", (code) => {
			const err = new Error(`docker sidecar exited (code ${code ?? "?"}): ${this.stderrBuf || "no stderr"}`);
			for (const [, p] of this.pending) p.reject(err);
			this.pending.clear();
			this.proc = undefined;
			this.buf = "";
		});
	}

	private onStdout(chunk: string): void {
		this.buf += chunk;
		for (;;) {
			const nl = this.buf.indexOf("\n");
			if (nl < 0) break;
			const line = this.buf.slice(0, nl);
			this.buf = this.buf.slice(nl + 1);
			if (!line.trim()) continue;
			let reply: SidecarReply;
			try {
				reply = JSON.parse(line) as SidecarReply;
			} catch {
				continue;
			}
			const waiter = this.pending.get(reply.id);
			if (waiter) {
				this.pending.delete(reply.id);
				waiter.resolve(reply);
			}
		}
	}

	private request(payload: Record<string, unknown>, timeoutMs: number): Promise<SidecarReply> {
		this.start();
		const proc = this.proc;
		if (!proc || !proc.stdin) {
			return Promise.reject(new Error("docker sidecar not running"));
		}
		const id = this.nextId++;
		return new Promise<SidecarReply>((resolve, reject) => {
			const timer = setTimeout(() => {
				this.pending.delete(id);
				reject(new Error(`sidecar request timeout after ${timeoutMs}ms`));
			}, timeoutMs);
			this.pending.set(id, {
				resolve: (r) => {
					clearTimeout(timer);
					resolve(r);
				},
				reject: (e) => {
					clearTimeout(timer);
					reject(e);
				},
			});
			const line = `${JSON.stringify({ id, ...payload })}\n`;
			proc.stdin.write(line, (err) => {
				if (err) {
					this.pending.delete(id);
					clearTimeout(timer);
					reject(err);
				}
			});
		});
	}

	async startContainer(task: string, rep: number): Promise<string> {
		const r = await this.request({ op: "start", task, rep }, 660_000);
		if (!r.ok || !r.container) {
			throw new Error(r.error ?? "start failed");
		}
		return r.container;
	}

	async exec(container: string, cmd: string, timeoutSec = 120): Promise<ExecResult> {
		try {
			const r = await this.request({ op: "exec", container, cmd, timeout: timeoutSec }, (timeoutSec + 60) * 1000);
			if (!r.ok) {
				const output = r.error ?? "exec failed";
				return { output, exitCode: -1, infra: classifyInfra(output) };
			}
			const output = r.output ?? "";
			const exitCode = r.exit_code ?? -1;
			return { output, exitCode, infra: r.infra === true || classifyInfra(output) };
		} catch (err) {
			// Sidecar busy/timeout is not automatically container-dead infra.
			// Only docker-daemon style errors count; otherwise surface as a domain failure.
			const msg = err instanceof Error ? err.message : String(err);
			return { output: `[sidecar] ${msg}`, exitCode: -1, infra: classifyInfra(msg) };
		}
	}

	/**
	 * Run a test/verification command with the task's tests injected into a private
	 * mount namespace (tmpfs). The tests never appear in the agent-visible filesystem.
	 * isolate=true (default): the command runs in an overlay whose writes are discarded
	 * after the run (in-loop feedback leaves zero residue). isolate=false: the official
	 * suite runs on the real rootfs, matching official TB semantics.
	 */
	async execTests(
		container: string,
		task: string,
		cmd: string,
		timeoutSec = 120,
		isolate = true,
	): Promise<ExecResult> {
		try {
			const r = await this.request(
				{ op: "exec_tests", container, task, cmd, timeout: timeoutSec, isolate },
				(timeoutSec + 120) * 1000,
			);
			if (!r.ok) {
				const output = r.error ?? "exec_tests failed";
				return { output, exitCode: -1, infra: classifyInfra(output) };
			}
			const output = r.output ?? "";
			const exitCode = r.exit_code ?? -1;
			return { output, exitCode, infra: r.infra === true || classifyInfra(output) };
		} catch (err) {
			const msg = err instanceof Error ? err.message : String(err);
			return { output: `[sidecar] ${msg}`, exitCode: -1, infra: classifyInfra(msg) };
		}
	}

	/**
	 * Snapshot the agent container as the verify image BEFORE the official run
	 * (see sidecar commit_verify). Consumed by startVerifyContainer.
	 */
	async commitVerifyImage(task: string, rep: number, src: string): Promise<void> {
		const r = await this.request({ op: "commit_verify", task, rep, src }, 900_000);
		if (!r.ok) {
			throw new Error(r.error ?? "commit_verify failed");
		}
	}

	/**
	 * Start a pristine verifier container: fresh task image seeded with mirror
	 * configs/caches, /app replaced by the agent container's final /app.
	 * Returns the verify container name (tb2verify-<task>[-rN]).
	 */
	async startVerifyContainer(task: string, rep: number, src: string): Promise<string> {
		const r = await this.request({ op: "start_verify", task, rep, src }, 900_000);
		if (!r.ok || !r.container) {
			throw new Error(r.error ?? "start_verify failed");
		}
		return r.container;
	}

	/**
	 * Recover a dead container: prefer `docker start` (preserves /app), else recreate.
	 * Returns the sidecar mode: running | started | recreated.
	 */
	async restartContainer(container: string, task: string, rep: number): Promise<string> {
		const r = await this.request({ op: "restart", container, task, rep }, 120_000);
		if (!r.ok) {
			throw new Error(r.error ?? "restart failed");
		}
		return r.mode ?? "started";
	}

	async stopContainer(container: string): Promise<void> {
		const r = await this.request({ op: "stop", container }, 90_000);
		if (!r.ok) {
			throw new Error(r.error ?? "stop failed");
		}
	}

	dispose(): void {
		if (!this.proc) return;
		try {
			this.proc.stdin.end();
		} catch {
			// ignore
		}
		this.proc.kill();
		this.proc = undefined;
	}
}
