import type { ExecResult } from "./types.ts";

export type WorldExec = (cmd: string, timeoutSec?: number) => Promise<ExecResult>;

export interface WorldArtifact {
	exists: boolean;
	size?: number;
}

/**
 * Deterministic, tool-agnostic snapshot of the container state at a point in
 * time. Gathered from /proc + python stdlib only (no pkill/ps/ss dependency),
 * so it works on stripped images.
 */
export interface WorldState {
	ports: number[];
	processes: string[];
	binaries: Record<string, boolean>;
	artifacts: Record<string, WorldArtifact>;
	appListing: Record<string, string>;
	capturedAt: string;
}

/** Minimal toolbox probed by name; absence is surfaced instead of assumed. */
export const TOOLBOX = [
	"sh",
	"bash",
	"python3",
	"pip3",
	"git",
	"make",
	"gcc",
	"cc",
	"curl",
	"wget",
	"ssh",
	"sshpass",
	"expect",
	"qemu-system-x86_64",
	"pkill",
	"killall",
	"ps",
	"ss",
	"netstat",
	"fuser",
	"lsof",
	"ip",
	"ping",
	"nc",
	"tmux",
	"screen",
	"setsid",
	"stdbuf",
	"timeout",
	"nohup",
	"mount",
	"isoinfo",
	"apk",
	"apt-get",
	"chpasswd",
];

function b64(s: string): string {
	return Buffer.from(s, "utf8").toString("base64");
}

function probeCommand(paths: string[]): string {
	const payload = b64(JSON.stringify({ paths, tools: TOOLBOX }));
	const py = [
		"import base64,json,os,glob,sys",
		"d=json.loads(base64.b64decode(sys.argv[1]).decode('utf-8'))",
		"out={'ports':[],'processes':[],'binaries':{},'artifacts':{},'appListing':{}}",
		"for fn in ('/proc/net/tcp','/proc/net/tcp6'):",
		"    try:",
		"        for i,line in enumerate(open(fn)):",
		"            if i==0: continue",
		"            p=line.split()",
		"            if len(p)>=4 and p[3]=='0A':",
		"                try: out['ports'].append(int(p[1].split(':')[1],16))",
		"                except Exception: pass",
		"    except Exception: pass",
		"out['ports']=sorted(set(out['ports']))",
		"for dd in glob.glob('/proc/[0-9]*'):",
		"    try:",
		"        c=open(dd+'/cmdline','rb').read().decode('utf-8','replace').replace(chr(0),' ').strip()",
		"        if c and 'b64decode(sys.argv' not in c: out['processes'].append(c[:160])",
		"    except Exception: pass",
		"out['processes']=out['processes'][:40]",
		"pathdirs=os.environ.get('PATH','/usr/bin:/bin').split(':')",
		"for b in d['tools']:",
		"    out['binaries'][b]=any(os.path.isdir(x) and os.access(os.path.join(x,b),os.X_OK) for x in pathdirs)",
		"for p in d['paths']:",
		"    try: out['artifacts'][p]={'exists':os.path.exists(p),'size':(os.path.getsize(p) if os.path.isfile(p) else None)}",
		"    except Exception: out['artifacts'][p]={'exists':False}",
		"try:",
		"    for n in sorted(os.listdir('/app'))[:60]:",
		"        fp='/app/'+n",
		"        try: out['appListing'][n]='dir' if os.path.isdir(fp) else str(os.path.getsize(fp))",
		"        except Exception: out['appListing'][n]='?'",
		"except Exception: pass",
		"print('===WORLD===')",
		"print(json.dumps(out))",
	].join("\n");
	return `python3 -c "import base64;exec(base64.b64decode('${b64(py)}').decode())" '${payload}'`;
}

function emptyWorld(): WorldState {
	return {
		ports: [],
		processes: [],
		binaries: {},
		artifacts: {},
		appListing: {},
		capturedAt: new Date().toISOString(),
	};
}

/** Gather the world snapshot. Never throws; on any failure returns an empty snapshot. */
export async function collectWorldState(exec: WorldExec, paths: string[]): Promise<WorldState> {
	const uniq = [...new Set(paths.filter((p) => typeof p === "string" && p.startsWith("/"))).values()].slice(0, 24);
	try {
		const r = await exec(probeCommand(uniq), 30);
		const marker = "===WORLD===";
		if (!r.output.includes(marker)) return emptyWorld();
		const line = r.output.split(marker).pop()?.trim().split("\n")[0] ?? "";
		const parsed: unknown = JSON.parse(line);
		if (!parsed || typeof parsed !== "object") return emptyWorld();
		const o = parsed as Partial<WorldState>;
		return {
			ports: Array.isArray(o.ports) ? o.ports.map(Number).filter((n) => Number.isFinite(n)) : [],
			processes: Array.isArray(o.processes) ? o.processes.map(String) : [],
			binaries: o.binaries && typeof o.binaries === "object" ? (o.binaries as Record<string, boolean>) : {},
			artifacts:
				o.artifacts && typeof o.artifacts === "object" ? (o.artifacts as Record<string, WorldArtifact>) : {},
			appListing: o.appListing && typeof o.appListing === "object" ? (o.appListing as Record<string, string>) : {},
			capturedAt: new Date().toISOString(),
		};
	} catch {
		return emptyWorld();
	}
}

/** Render the snapshot as a compact ground-truth block for prompts/feedback. */
export function renderWorldState(ws: WorldState | undefined, maxLen = 1600): string {
	if (!ws) return "";
	const ports = ws.ports.length ? ws.ports.join(", ") : "none";
	const procs = ws.processes.length ? ws.processes.slice(0, 5).join(" ; ") : "none";
	const present = Object.entries(ws.binaries)
		.filter(([, v]) => v)
		.map(([k]) => k);
	const missing = Object.entries(ws.binaries)
		.filter(([, v]) => !v)
		.map(([k]) => k);
	const app = Object.entries(ws.appListing)
		.slice(0, 40)
		.map(([n, s]) => `${n}=${s}`);
	const art = Object.entries(ws.artifacts).map(
		([p, a]) =>
			`${p}=${a.exists ? `exists${a.size !== undefined && a.size !== null ? `(${a.size}B)` : ""}` : "MISSING"}`,
	);
	const lines = [
		"WORLD STATE (deterministic ground truth captured from the container; treat as fact and override any memory):",
		`- listening TCP ports: ${ports}`,
		`- live processes: ${procs}`,
		`- tools present: ${present.join(" ") || "(none probed)"}`,
		`- tools MISSING (do NOT call these): ${missing.join(" ") || "none"}`,
		app.length ? `- /app entries: ${app.join(", ")}` : "",
		art.length ? `- referenced artifacts: ${art.join(", ")}` : "",
	].filter(Boolean);
	return lines.join("\n").slice(0, maxLen);
}

const PRODUCE_RE = /(?:>>?|tee\s+|mkdir\s+-p\s+|touch\s+|install\s+-[A-Za-z0-9_-]+\s+)\s*(\/[A-Za-z0-9._/-]+)/g;
// Only interpreter/exec consumers imply "this file must already exist". Read-only
// probes (cat/wc/stat/grep/test) are intentionally excluded: checking for absence
// is legitimate and must not make a candidate infeasible.
const CONSUME_RE =
	/(?:^|[\s;|&(])(?:expect\s+-[A-Za-z]*f|bash|sh|source|\.|python3?|perl|ruby|node)\s+(?:-[A-Za-z0-9_-]+\s+)*(\/[A-Za-z0-9._/-]+)/g;

function matchPaths(re: RegExp, text: string): string[] {
	const out: string[] = [];
	for (const m of text.matchAll(re)) {
		if (m[1]) out.push(m[1].replace(/[.,;:]+$/, ""));
	}
	return out;
}

/** Paths a command sequence creates (redirections, mkdir, touch, tee). */
export function inferProduced(commands: string[]): string[] {
	return [...new Set(commands.flatMap((c) => matchPaths(PRODUCE_RE, c)))];
}

/** Paths a command sequence reads/executes (expect -f, bash, cat, chmod, ...). */
export function inferConsumed(commands: string[]): string[] {
	const produced = new Set(inferProduced(commands));
	return [...new Set(commands.flatMap((c) => matchPaths(CONSUME_RE, c)).filter((p) => !produced.has(p)))];
}

export interface CandidateFeasibility {
	id: string;
	label: string;
	feasible: boolean;
	reasons: string[];
}

/**
 * Deterministic precondition check: a candidate may only consume artifacts,
 * binaries, or ports that already exist (or that it produces itself).
 * `assumes` entries may be a path, `bin:<name>`, or `port:<n>`.
 */
export function validateCandidate(
	cand: { id: string; label: string; commands: string[]; assumes?: string[]; provides?: string[] },
	ws: WorldState,
): CandidateFeasibility {
	const reasons: string[] = [];
	const produced = new Set<string>([...(cand.provides ?? []), ...inferProduced(cand.commands)]);
	const required = new Set<string>([...(cand.assumes ?? []), ...inferConsumed(cand.commands)]);
	for (const req of required) {
		if (req.startsWith("bin:")) {
			const b = req.slice(4);
			if (ws.binaries[b] === false) reasons.push(`requires '${b}' but it is MISSING in the container`);
			continue;
		}
		if (req.startsWith("port:")) {
			const p = Number(req.slice(5));
			if (Number.isFinite(p) && !ws.ports.includes(p)) {
				reasons.push(`requires port ${p} to be listening but nothing is listening there`);
			}
			continue;
		}
		if (!req.startsWith("/")) continue;
		if (produced.has(req)) continue;
		const art = ws.artifacts[req];
		if (art && !art.exists)
			reasons.push(`consumes '${req}' which does not exist and is not created by this candidate`);
	}
	return { id: cand.id, label: cand.label, feasible: reasons.length === 0, reasons };
}
