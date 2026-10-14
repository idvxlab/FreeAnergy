import type { ExtensionContext } from "@earendil-works/pi-coding-agent";
import { completeText } from "./llm.ts";
import type { EnvConstraint } from "./state.ts";
import type { ExecResult } from "./types.ts";

const PATH_RE =
	/(?:\/[\w.-]+)+\/[\w.-]+|[\w.-]+\.(?:c|h|cc|cpp|py|rs|go|java|js|ts|sh|txt|json|yaml|yml|toml|md|mk)\b/g;

function shellQuote(s: string): string {
	return `'${s.replaceAll("'", `'\\''`)}'`;
}

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

/**
 * Opening evidence collection before skill gen: /app listing, task-named
 * files, small /app sources. Read-only; budget-capped. /tests is intentionally
 * never read so the acceptance suite cannot leak into the skill/plan.
 */
export async function reconEvidence(
	exec: (cmd: string, timeoutSec?: number) => Promise<ExecResult>,
	instruction: string,
	budget = 14000,
): Promise<string> {
	const paths: string[] = [];
	for (const m of instruction.matchAll(PATH_RE)) {
		const p = m[0].replace(/[.,;:\])}"']+$/, "");
		if (p && !paths.includes(p)) paths.push(p);
	}
	const parts = ["echo '### /app ###'; ls -la /app 2>/dev/null"];
	if (paths.length > 0) {
		parts.push("echo '### TASK-NAMED FILES ###'");
		for (const p of paths.slice(0, 8)) {
			parts.push(
				`for f in ${shellQuote(p)}; do [ -f "$f" ] && echo "----- $f -----" && head -c 2500 "$f" | tr -cd '\\11\\12\\15\\40-\\176'; done`,
			);
		}
	}
	parts.push(
		'echo \'### /app SOURCES ###\'; for f in /app/*; do case "$f" in *.c|*.h|*.cc|*.cpp|*.py|*.rs|*.go|*.java|*.js|*.ts|*.sh|Makefile|*.mk|*.txt|*.json|*.toml|*.yaml|*.yml) echo "----- $f -----"; head -c 2500 "$f";; esac; done',
	);
	try {
		const res = await exec(parts.join("; "), 60);
		return res.output.slice(0, budget);
	} catch {
		return "";
	}
}

/** Opening env probe: installed pip packages + PATH binaries (for env constraints). */
export async function probeEnv(exec: (cmd: string, timeoutSec?: number) => Promise<ExecResult>): Promise<string> {
	try {
		const r = await exec(
			"pip3 list 2>/dev/null | head -80; echo '---TOOLS---'; " +
				'for d in $(echo $PATH | tr ":" " "); do [ -d "$d" ] && ls "$d" 2>/dev/null; done | sort -u | head -200',
			60,
		);
		return r.output.slice(0, 1500);
	} catch {
		return "";
	}
}

const ENV_C_SYSTEM = `You are an environment-constraint extractor. Given a brief ENV probe (pip packages + PATH tools), extract at most 2 STATIC environment facts that constrain a coding agent (e.g. "pip needs --break-system-packages", "no GPU tools on PATH"). Facts must be conditions to AWARE of, not procedures. Return ONLY JSON {"items": [{"title": "...", "text": "..."}]}; empty if none.`;

/** LLM probe of env_obs → env_constraints [{title, text}] (max 2). Needs session ctx for completeText. */
export async function probeEnvConstraintsWith(ctx: ExtensionContext, envObs: string): Promise<EnvConstraint[]> {
	if (!envObs?.trim()) return [];
	const text = await completeText(ctx, ENV_C_SYSTEM, envObs.slice(0, 1500), 300);
	const j = extractJson(text);
	if (!Array.isArray(j?.items)) return [];
	const out: EnvConstraint[] = [];
	for (const raw of (j.items as unknown[]).slice(0, 2)) {
		if (!raw || typeof raw !== "object") continue;
		const o = raw as { title?: unknown; text?: unknown };
		const t = String(o.text ?? "").trim();
		if (!t) continue;
		out.push({ title: String(o.title ?? "").slice(0, 80), text: t.slice(0, 400) });
	}
	return out;
}
