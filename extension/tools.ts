import { defineTool, type ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { assertWritable, resolveContainerPath, truncateOutput } from "./paths.ts";
import { pytestBootstrap, pytestCmd } from "./settle.ts";
import { requireContainer, type VfeState, verifierTimeoutSec } from "./state.ts";
import type { ExecResult } from "./types.ts";

function shellQuote(s: string): string {
	return `'${s.replaceAll("'", `'\\''`)}'`;
}

function b64(s: string): string {
	return Buffer.from(s, "utf8").toString("base64");
}

function textResult(output: string, details: Record<string, unknown>) {
	return {
		content: [{ type: "text" as const, text: truncateOutput(output) }],
		details,
	};
}

function errorResult(message: string) {
	return {
		content: [{ type: "text" as const, text: message }],
		details: { error: true },
	};
}

/** Register container-backed tools (override built-ins by name where they collide). */
export function registerContainerTools(
	pi: ExtensionAPI,
	state: VfeState,
	exec: (cmd: string, timeoutSec?: number) => Promise<ExecResult>,
	execTests: (cmd: string, timeoutSec?: number) => Promise<ExecResult>,
): void {
	const runShell = async (cmd: string, timeoutSec?: number): Promise<ExecResult> => {
		requireContainer(state);
		return exec(cmd, timeoutSec);
	};

	const runTests = async (cmd: string, timeoutSec?: number): Promise<ExecResult> => {
		requireContainer(state);
		return execTests(cmd, timeoutSec);
	};

	pi.registerTool(
		defineTool({
			name: "read",
			label: "read (container)",
			description: "Read a file inside the task container (paths under /app). Relative paths resolve against /app.",
			promptSnippet: "Read file contents from the task container",
			promptGuidelines: ["Use read for files in the container; host files are not visible."],
			parameters: Type.Object({
				path: Type.String({ description: "Path relative to /app or absolute under /app" }),
				offset: Type.Optional(Type.Number({ description: "1-indexed start line" })),
				limit: Type.Optional(Type.Number({ description: "Max lines to return" })),
			}),
			async execute(_id, params) {
				const path = resolveContainerPath(params.path);
				const start = params.offset && params.offset > 0 ? params.offset : 1;
				const end = params.limit && params.limit > 0 ? start + params.limit - 1 : 10_000_000;
				const r = await runShell(`sed -n ${start},${end}p -- ${shellQuote(path)} 2>&1`);
				if (r.exitCode !== 0 && !r.output.trim()) {
					return errorResult(`read failed (exit ${r.exitCode}): ${path}`);
				}
				return textResult(r.output, { path, exitCode: r.exitCode });
			},
		}),
	);

	pi.registerTool(
		defineTool({
			name: "write",
			label: "write (container)",
			description: "Create or overwrite a file inside the task container under /app (base64-safe).",
			promptSnippet: "Write a file in the task container",
			promptGuidelines: ["Use write for full file contents; prefer edit for small changes."],
			parameters: Type.Object({
				path: Type.String({ description: "Path relative to /app (must stay under /app)" }),
				content: Type.String({ description: "Full file content to write" }),
			}),
			async execute(_id, params) {
				const path = resolveContainerPath(params.path);
				assertWritable(path);
				const payload = b64(params.content);
				const dir = path.includes("/") ? path.slice(0, path.lastIndexOf("/")) : "/app";
				const r = await runShell(
					`mkdir -p ${shellQuote(dir)} && printf %s ${shellQuote(payload)} | base64 -d > ${shellQuote(path)} && wc -c < ${shellQuote(path)}`,
				);
				if (r.exitCode !== 0) {
					return errorResult(`write failed (exit ${r.exitCode}): ${r.output}`);
				}
				return textResult(`Wrote ${path} (${r.output.trim()} bytes)`, { path, bytes: r.output.trim() });
			},
		}),
	);

	pi.registerTool(
		defineTool({
			name: "edit",
			label: "edit (container)",
			description:
				"Exact text replacement in a container file. Each oldText must match exactly once in the original file.",
			promptSnippet: "Apply exact text replacements in a container file",
			promptGuidelines: ["Each edits[].oldText must be unique in the file and non-overlapping."],
			parameters: Type.Object({
				path: Type.String({ description: "File path under /app" }),
				edits: Type.Array(
					Type.Object({
						oldText: Type.String({ description: "Exact text to replace (must appear exactly once)" }),
						newText: Type.String({ description: "Replacement text" }),
					}),
					{ description: "Non-overlapping replacements matched against the original file" },
				),
			}),
			async execute(_id, params) {
				const path = resolveContainerPath(params.path);
				assertWritable(path);
				const req = b64(JSON.stringify({ path, edits: params.edits }));
				const script = [
					"import base64,json,sys",
					`d=json.loads(base64.b64decode(sys.argv[1]).decode("utf-8"))`,
					`p=d["path"]`,
					`text=open(p,encoding="utf-8").read()`,
					`for i,e in enumerate(d["edits"]):`,
					`    old,new=e["oldText"],e["newText"]`,
					`    n=text.count(old)`,
					`    if n!=1: sys.stderr.write(f"edit {i}: oldText count={n} (need 1)\\n"); sys.exit(2)`,
					`    if old==new: sys.stderr.write(f"edit {i}: oldText==newText\\n"); sys.exit(3)`,
					`    text=text.replace(old,new,1)`,
					`open(p,"w",encoding="utf-8").write(text)`,
					`print(f"applied {len(d['edits'])} edit(s) to {p}")`,
				].join("\n");
				const r = await runShell(`python3 -c ${shellQuote(script)} ${shellQuote(req)} 2>&1`);
				if (r.exitCode !== 0) {
					return errorResult(`edit failed (exit ${r.exitCode}): ${r.output}`);
				}
				return textResult(r.output, { path, edits: params.edits.length });
			},
		}),
	);

	pi.registerTool(
		defineTool({
			name: "grep",
			label: "grep (container)",
			description: "Search file contents inside the container with a regex pattern.",
			promptSnippet: "Regex-search files in the task container",
			parameters: Type.Object({
				pattern: Type.String({ description: "Regex pattern" }),
				path: Type.Optional(Type.String({ description: "Directory or file under /app (default /app)" })),
				ignoreCase: Type.Optional(Type.Boolean({ description: "Case-insensitive (default false)" })),
				limit: Type.Optional(Type.Number({ description: "Max matches (default 100)" })),
			}),
			async execute(_id, params) {
				const path = resolveContainerPath(params.path ?? "/app");
				const limit = params.limit && params.limit > 0 ? params.limit : 100;
				const flags = params.ignoreCase ? "-i" : "";
				const r = await runShell(
					`grep -R -n ${flags} -e ${shellQuote(params.pattern)} -- ${shellQuote(path)} 2>/dev/null | head -n ${limit}`,
				);
				const out = r.output.trim() || "(no matches)";
				return textResult(out, { path, pattern: params.pattern });
			},
		}),
	);

	pi.registerTool(
		defineTool({
			name: "find",
			label: "find (container)",
			description: "Find files by glob pattern in the container (python glob, recursive=True).",
			promptSnippet: "Glob-find files in the task container",
			parameters: Type.Object({
				pattern: Type.String({ description: "Glob pattern, e.g. '**/*.py'" }),
				path: Type.Optional(Type.String({ description: "Root directory (default /app)" })),
				limit: Type.Optional(Type.Number({ description: "Max results (default 200)" })),
			}),
			async execute(_id, params) {
				const path = resolveContainerPath(params.path ?? "/app");
				const limit = params.limit && params.limit > 0 ? params.limit : 200;
				const payload = b64(JSON.stringify({ pattern: params.pattern, path, limit }));
				const script = [
					"import base64,glob,json,os,sys",
					`d=json.loads(base64.b64decode(sys.argv[1]).decode("utf-8"))`,
					`os.chdir(d["path"])`,
					`hits=sorted(glob.glob(d["pattern"], recursive=True))[:d["limit"]]`,
					`print("\\n".join(hits) if hits else "(no matches)")`,
				].join("\n");
				const r = await runShell(`python3 -c ${shellQuote(script)} ${shellQuote(payload)} 2>&1`);
				if (r.exitCode !== 0) {
					return errorResult(`find failed (exit ${r.exitCode}): ${r.output}`);
				}
				return textResult(r.output, { pattern: params.pattern, path });
			},
		}),
	);

	pi.registerTool(
		defineTool({
			name: "ls",
			label: "ls (container)",
			description: "List a directory inside the task container.",
			promptSnippet: "List a container directory",
			parameters: Type.Object({
				path: Type.Optional(Type.String({ description: "Directory (default /app)" })),
				limit: Type.Optional(Type.Number({ description: "Max entries (default 500)" })),
			}),
			async execute(_id, params) {
				const path = resolveContainerPath(params.path ?? "/app");
				const limit = params.limit && params.limit > 0 ? params.limit : 500;
				const r = await runShell(`ls -la -- ${shellQuote(path)} 2>&1 | head -n ${limit + 1}`);
				return textResult(r.output, { path, exitCode: r.exitCode });
			},
		}),
	);

	pi.registerTool(
		defineTool({
			name: "run",
			label: "run (container)",
			description:
				"Run a sequence of shell commands inside the container, in order. Commands see each other's filesystem effects; shell variables do not persist across entries (each entry is a separate sh -lc).",
			promptSnippet: "Run a command sequence in the task container",
			promptGuidelines: [
				"Prefer a short sequence (1–8 commands) with a clear goal.",
				"Compile/run loops: keep write steps explicit before run steps.",
			],
			parameters: Type.Object({
				commands: Type.Array(Type.String(), { description: "Shell commands to run in order" }),
				stopOnFailure: Type.Optional(Type.Boolean({ description: "Stop at first non-zero exit (default false)" })),
				timeoutSec: Type.Optional(Type.Number({ description: "Per-command timeout (default 120)" })),
			}),
			async execute(_id, params) {
				if (params.commands.length === 0) {
					return errorResult("commands must be non-empty");
				}
				if (params.commands.length > 8) {
					return errorResult(`too many commands (${params.commands.length}); max 8 per run`);
				}
				// /tests is hidden: the acceptance suite must not be read as an answer key.
				const testsRe = /(?:^|[^A-Za-z0-9_./-])\/tests(?:\/|\b)|test_outputs\.py|\btest\.sh\b/;
				const blockedCmd = params.commands.find((c) => testsRe.test(c));
				if (blockedCmd) {
					return errorResult(
						"Access to /tests is blocked. Work only from /app and the task instruction; use run_test for suite feedback.",
					);
				}
				const timeout = params.timeoutSec && params.timeoutSec > 0 ? params.timeoutSec : 240;
				const parts: string[] = [];
				let lastExit = 0;
				for (let i = 0; i < params.commands.length; i++) {
					const cmd = params.commands[i];
					const r = await runShell(cmd, timeout);
					parts.push(`$ ${cmd}\n${r.output}(exit ${r.exitCode})`);
					lastExit = r.exitCode;
					if (r.exitCode !== 0 && params.stopOnFailure) {
						parts.push(`[stopped at command ${i + 1}]`);
						break;
					}
				}
				return textResult(parts.join("\n"), {
					n: params.commands.length,
					exitCode: lastExit,
				});
			},
		}),
	);

	pi.registerTool(
		defineTool({
			name: "run_test",
			label: "run_test (container)",
			description:
				"Run the task's pytest suite (/tests/test_outputs.py) against /app. Returns pass/fail summary. Does not modify /app except pytest cache (ignored).",
			promptSnippet: "Run the container test suite",
			promptGuidelines: [
				"Call run_test after meaningful writes or when diagnosing failure.",
				"Tests are the ground truth: use FAILED names and assertions as feedback.",
			],
			parameters: Type.Object({
				timeoutSec: Type.Optional(
					Type.Number({
						description: "Timeout seconds (default: task.toml [verifier] timeout_sec, clamped to 300-3600)",
					}),
				),
			}),
			async execute(_id, params) {
				const timeout =
					params.timeoutSec && params.timeoutSec > 0 ? params.timeoutSec : verifierTimeoutSec(state.task);
				await runShell(pytestBootstrap(state.task), 600);
				const cmd = `(${pytestCmd("python3")} || true) | tail -c 6000`;
				const r = await runTests(cmd, Math.max(timeout, 300));
				const output = r.output;
				const passed = /(\d+) passed/.exec(output)?.[1];
				const failed = /(\d+) failed/.exec(output)?.[1];
				return textResult(output, {
					passed: passed ? Number(passed) : 0,
					failed: failed ? Number(failed) : 0,
					exitCode: r.exitCode,
				});
			},
		}),
	);
}
