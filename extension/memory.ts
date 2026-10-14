import { appendFileSync, existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import type { ExtensionContext } from "@earendil-works/pi-coding-agent";
import { completeText } from "./llm.ts";

export type ExperienceCategory = "methods" | "failures" | "constraints" | "tool" | "practices";
export type ExperienceScope = "task" | "pending" | "global";

export interface Experience {
	category: ExperienceCategory;
	name?: string;
	text?: string;
	steps?: string[];
	score?: number;
	conf?: number;
	last_ts?: string;
	scope?: ExperienceScope;
	kind?: string;
	/** Distinct task names this experience has been observed/validated in; global needs >=2. */
	tasks?: string[];
}

export interface MemoryFile {
	schema_version: number;
	next_id: number;
	experiences: Record<string, Experience>;
}

export interface ExperienceStore {
	experiences: Map<string, Experience>;
	nextId: number;
	injected: Set<string>;
	path: string;
	/** Current task name; stamped onto entries and used for same-task pending injection. */
	task?: string;
}

const MEM_CATEGORIES: ReadonlySet<string> = new Set(["methods", "failures", "constraints", "tool", "practices"]);
const NAT_CATS = ["methods", "constraints", "practices", "failures"] as const;
const TOOL_REL_MIN = 0.5;
const NAT_REL_MIN = 0.3;
const MEM_SAVE_EVERY = 5;
const MEM_DUP_REL = 0.6;
const PRACTICE_DUP_REL = 0.7;
/** Mature sequence-practice thresholds (raw23): inject only score/conf ≥ 3 globals. */
const PRACTICE_MIN_SCORE = 3;
const PRACTICE_MIN_CONF = 3;
/** Max mature practices injected per step prompt (raw23 PRACTICE_INJECT_MAX). */
export const PRACTICE_INJECT_MAX = 3;
const MEM_CAPS: Record<string, number> = {
	tool: 150,
	failures: 100,
	methods: 150,
	practices: 100,
	constraints: 200,
};

const DEFAULT_MEM_PATH = "vfe_memory.json";
const PROC_ID = `${process.pid}.${Date.now() % 100000000}`;

export function memoryPath(): string {
	return process.env.VFE_MEMORY_PATH?.trim() || DEFAULT_MEM_PATH;
}

/** ASCII code words (+ paths) + CJK bigrams (jieba-free). */
export function tokenize(text: string): Set<string> {
	const toks = new Set<string>();
	const t = String(text ?? "").toLowerCase();
	for (const m of t.matchAll(/[a-z][a-z0-9_./+-]{1,}/g)) toks.add(m[0]);
	for (const m of t.matchAll(/[一-鿿]+/g)) {
		const s = m[0];
		if (s.length <= 2) toks.add(s);
		else for (let i = 0; i < s.length - 1; i++) toks.add(s.slice(i, i + 2));
	}
	return toks;
}

/** Word-overlap relevance: |q∩d| / sqrt(|q|) + substring bonus. 0 = unrelated. */
export function relScore(query: string, doc: string): number {
	const qt = tokenize(query);
	const dt = tokenize(doc);
	if (qt.size === 0 || dt.size === 0) return 0;
	let base = 0;
	for (const w of qt) {
		if (dt.has(w)) base++;
		else if (w.length >= 3 && doc.toLowerCase().includes(w)) base += 0.5;
	}
	return base / Math.sqrt(qt.size);
}

function nowIso(): string {
	return new Date().toISOString();
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

function headTail(s: string, max = 600): string {
	const buf = Buffer.from(s, "utf8");
	if (buf.length <= max) return s;
	const head = buf.subarray(0, Math.floor(max * 0.65));
	const tail = buf.subarray(buf.length - Math.floor(max * 0.3));
	return `${head.toString("utf8")}\n…[truncated]…\n${tail.toString("utf8")}`;
}

function expDisplay(exp: Experience): string {
	const t = String(exp.text ?? "").trim();
	if (t) return t;
	if (exp.category === "practices" && exp.kind === "sequence") {
		const steps = (exp.steps ?? []).slice(0, 6).map(String).join(" | ");
		return `when ${String(exp.name ?? "").slice(0, 120)} do steps: ${steps}`;
	}
	return t;
}

function isTaskSpecific(text: string): boolean {
	const t = text || "";
	if (/\/app\/|\/tests\/|\/home\/|\/opt\/|\/usr\/local/.test(t)) return true;
	if (/\b\S+\.(tex|html|js|py|json|c|h|cpp|go|rs|md)\b/.test(t)) return true;
	return false;
}

function scopeOfNew(text: string): ExperienceScope {
	return isTaskSpecific(text) ? "task" : "pending";
}

/** Record that this experience was encountered/validated in a task (distinct-task evidence). */
function markTask(exp: Experience, task?: string): void {
	if (!task) return;
	const list = exp.tasks ?? [];
	if (!list.includes(task)) {
		list.push(task);
		exp.tasks = list;
	}
}

/** Cross-task evidence: an experience is "global knowledge" only after >=2 distinct tasks. */
function crossTaskValidated(exp: Experience): boolean {
	return (exp.tasks?.length ?? 0) >= 2;
}

/** Path/digit-normalized command family key (Python `_ema_key`). */
export function emaKey(cmd: string): string {
	let t = String(cmd || "");
	t = t.replace(/\/app\/[A-Za-z0-9_./+-]+/g, "/app/F");
	t = t.replace(/\/tests\/[A-Za-z0-9_./+-]+/g, "/tests/F");
	t = t.replace(/\b\d+\b/g, "N");
	t = t.replace(/\d+>|&\d+|&&|\|\||2>&1|2>\/dev\/null|>&/g, "");
	t = t.replace(/\s+/g, " ").trim();
	return t.slice(0, 60);
}

function polSig(name: string): string {
	let t = String(name || "").replace(/\/app\/[A-Za-z0-9_./+-]+/g, "/app/F");
	t = t.replace(/\/tests\/[A-Za-z0-9_./+-]+/g, "/tests/F");
	t = t.replace(/\b\d+\b/g, "N");
	t = t.replace(/\s+/g, " ").trim().toLowerCase();
	return t.slice(0, 120);
}

function seqOverlap(a: string[], b: string[]): number {
	const sa = new Set(a);
	const sb = new Set(b);
	if (sa.size === 0 || sb.size === 0) return 0;
	let inter = 0;
	for (const x of sa) if (sb.has(x)) inter++;
	return inter / (sa.size + sb.size - inter);
}

function failureFp(text: string): string {
	const t = String(text || "").trim();
	const m = /[`"“「]([^`"”」]{1,80})[`"”」]/.exec(t);
	const frag = (m ? m[1].trim() : t).slice(0, 60);
	if (!frag) return "misc";
	return emaKey(frag) || frag;
}

interface LoadedMemory {
	experiences: Map<string, Experience>;
	nextId: number;
}

function loadFromDisk(path: string): LoadedMemory {
	const experiences = new Map<string, Experience>();
	let nextId = 0;
	try {
		if (!existsSync(path)) return { experiences, nextId };
		const d = JSON.parse(readFileSync(path, "utf8")) as Partial<MemoryFile>;
		const map = d.experiences ?? {};
		for (const [eid, e] of Object.entries(map)) {
			if (!e || typeof e !== "object") continue;
			if (!MEM_CATEGORIES.has(e.category)) continue;
			const scope = e.scope ?? "global";
			if (scope === "pending" || scope === "global") {
				experiences.set(String(eid), e);
			}
		}
		nextId = Number(d.next_id) || 0;
	} catch {
		experiences.clear();
	}
	return { experiences, nextId };
}

function enforceCaps(experiences: Map<string, Experience>): void {
	for (const [cat, cap] of Object.entries(MEM_CAPS)) {
		const items = [...experiences.entries()].filter(([, e]) => e.category === cat);
		if (items.length <= cap) continue;
		items.sort((a, b) => {
			const sa = a[1].score ?? 0;
			const sb = b[1].score ?? 0;
			if (sa !== sb) return sa - sb;
			const ca = a[1].conf ?? 0;
			const cb = b[1].conf ?? 0;
			if (ca !== cb) return ca - cb;
			return String(a[1].last_ts ?? "").localeCompare(String(b[1].last_ts ?? ""));
		});
		for (const [eid] of items.slice(0, items.length - cap)) experiences.delete(eid);
	}
}

/** Merge-write non-task experiences back to disk (cross-process safe merge). */
export function saveExperiences(experiences: Map<string, Experience>, nextId: number, path = memoryPath()): void {
	enforceCaps(experiences);
	try {
		const dir = dirname(path);
		if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
		let disk: Record<string, Experience> = {};
		let diskNext = 0;
		try {
			if (existsSync(path)) {
				const d = JSON.parse(readFileSync(path, "utf8")) as Partial<MemoryFile>;
				disk = d.experiences ?? {};
				diskNext = Number(d.next_id) || 0;
			}
		} catch {
			disk = {};
		}
		const merged: Record<string, Experience> = { ...disk };
		for (const [eid, e] of experiences) {
			if ((e.scope ?? "global") !== "task") merged[eid] = e;
		}
		for (const [eid, e] of Object.entries(disk)) {
			if (!experiences.has(eid) && (e.scope ?? "global") !== "task") {
				experiences.set(eid, e);
			}
		}
		const out: MemoryFile = {
			schema_version: 2,
			next_id: Math.max(nextId, diskNext),
			experiences: merged,
		};
		const tmp = `${path}.tmp`;
		writeFileSync(tmp, JSON.stringify(out, null, 2), "utf8");
		renameSync(tmp, path);
	} catch (err) {
		console.error(`[WARN] saveExperiences failed: ${String(err)}`);
	}
}

export function loadStore(path = memoryPath()): ExperienceStore {
	const { experiences, nextId } = loadFromDisk(path);
	return { experiences, nextId, injected: new Set(), path };
}

function newExp(
	store: ExperienceStore,
	category: ExperienceCategory,
	text: string,
	name = "",
	extra?: Partial<Experience>,
): string {
	store.nextId += 1;
	const eid = `${category}-${PROC_ID}-${store.nextId}`;
	const entry: Experience = {
		category,
		name,
		text: String(text),
		score: 0,
		conf: 0,
		last_ts: nowIso(),
		scope: extra?.scope || scopeOfNew(text),
		...extra,
	};
	store.experiences.set(eid, entry);
	markTask(entry, store.task);
	return eid;
}

function bumpExp(store: ExperienceStore, eid: string, improved: boolean): void {
	const exp = store.experiences.get(eid);
	if (!exp) return;
	exp.score = (exp.score ?? 0) + (improved ? 1 : -1);
	exp.last_ts = nowIso();
	markTask(exp, store.task);
	// Promotion to global is reserved for cross-task knowledge: the same
	// experience validated in >=2 distinct tasks. Repetition inside one task
	// (or one run) is not evidence of generality.
	if (exp.scope === "pending" && crossTaskValidated(exp) && (exp.score ?? 0) > 0) {
		exp.scope = "global";
	}
	if ((exp.score ?? 0) <= -3 && exp.category !== "tool") {
		store.experiences.delete(eid);
	}
}

function findFailureByFp(store: ExperienceStore, fp: string): [string, Experience] | undefined {
	for (const [eid, exp] of store.experiences) {
		if (exp.category === "failures" && exp.name === fp) return [eid, exp];
	}
	return undefined;
}

export function addFailure(store: ExperienceStore, text: string, name = ""): void {
	const body = String(text);
	if (!body.trim()) return;
	const fp = name.trim() || failureFp(body);
	const same = findFailureByFp(store, fp);
	if (same) {
		same[1].text = body;
		same[1].score = (same[1].score ?? 0) + 1;
		same[1].last_ts = nowIso();
		return;
	}
	const prefix = body.slice(0, 40);
	for (const e of store.experiences.values()) {
		if (e.category === "failures" && prefix && String(e.text ?? "").includes(prefix)) return;
	}
	newExp(store, "failures", body, fp);
}

export function addMethod(store: ExperienceStore, text: string, name = ""): void {
	const body = String(text);
	if (!body.trim()) return;
	const prefix = body.slice(0, 40);
	for (const e of store.experiences.values()) {
		if (e.category === "methods" && prefix && String(e.text ?? "").includes(prefix)) {
			e.last_ts = nowIso();
			return;
		}
	}
	newExp(store, "methods", body, name);
}

export function addConstraint(store: ExperienceStore, text: string, name = ""): void {
	const body = String(text);
	if (!body.trim()) return;
	const prefix = body.slice(0, 40);
	for (const e of store.experiences.values()) {
		if (e.category === "constraints" && prefix && String(e.text ?? "").includes(prefix)) {
			e.last_ts = nowIso();
			return;
		}
	}
	newExp(store, "constraints", body, name);
}

function addPractice(store: ExperienceStore, name: string, steps: string[]): string {
	store.nextId += 1;
	const eid = `practices-${PROC_ID}-${store.nextId}`;
	const entry: Experience = {
		category: "practices",
		kind: "sequence",
		name: String(name).slice(0, 200),
		steps: steps.map((s) => String(s).slice(0, 300)),
		score: 0,
		conf: 0,
		last_ts: nowIso(),
		scope: "pending",
	};
	store.experiences.set(eid, entry);
	return eid;
}

function findToolPractice(store: ExperienceStore, fp: string): [string, Experience] | undefined {
	for (const [eid, exp] of store.experiences) {
		if (exp.category === "tool" && exp.name === fp) return [eid, exp];
	}
	return undefined;
}

export function updateToolPractice(store: ExperienceStore, cmd: string, success: boolean): void {
	const fp = emaKey(cmd);
	const hit = findToolPractice(store, fp);
	if (!hit) {
		newExp(store, "tool", "", fp, { scope: "pending" });
		return;
	}
	bumpExp(store, hit[0], success);
}

export function markConf(store: ExperienceStore): void {
	for (const exp of store.experiences.values()) {
		if ((exp.scope ?? "global") === "task") continue;
		exp.conf = (exp.conf ?? 0) + 1;
	}
}

interface TrajectoryRow {
	step?: number;
	command?: string;
	batch?: string[];
	outputs?: string[];
	improved?: boolean;
	fails?: string[] | number;
	objective?: string;
	[key: string]: unknown;
}

function trajFails(r: TrajectoryRow): number | undefined {
	if (typeof r.fails === "number") return r.fails;
	if (Array.isArray(r.fails)) return r.fails.length;
	return undefined;
}

function trajCmds(r: TrajectoryRow): string[] {
	const list = Array.isArray(r.batch) && r.batch.length > 0 ? r.batch : r.command ? [r.command] : [];
	return list
		.map(String)
		.map((s) => s.trim())
		.filter(Boolean);
}

export function extractBestPractice(
	store: ExperienceStore,
	rows: TrajectoryRow[],
	phaseLog: Array<{ step: number; objective: string }>,
): void {
	if (rows.length === 0) return;
	const effective: TrajectoryRow[] = [];
	let prev: number | undefined;
	for (const r of rows) {
		const nf = trajFails(r);
		const good = Boolean(r.improved) || (nf !== undefined && prev !== undefined && nf < prev);
		if (good && trajCmds(r).length > 0) effective.push(r);
		if (nf !== undefined) prev = nf;
	}
	if (effective.length === 0) return;
	const segments: Array<{ ph: string; objStep: number; steps: string[]; last: number }> = [];
	let cur: { ph: string; objStep: number; steps: string[]; last: number } | undefined;
	for (const r of effective) {
		const st = Number(r.step ?? 0);
		const ph = String(r.objective ?? "");
		if (!cur || ph !== cur.ph || st !== cur.last + 1) {
			if (cur) segments.push(cur);
			cur = { ph, objStep: st, steps: [], last: st };
		}
		cur.steps.push(...trajCmds(r));
		cur.last = st;
	}
	if (cur) segments.push(cur);
	const objOf = (step: number): string => {
		let obj = "";
		for (const s of phaseLog) {
			if (step >= s.step) obj = s.objective;
			else break;
		}
		return obj;
	};
	for (const seg of segments.slice(0, 3)) {
		const steps: string[] = [];
		for (const s of seg.steps) if (!steps.includes(s)) steps.push(s);
		if (steps.length === 0) continue;
		const nm = objOf(seg.objStep) || seg.ph || "general";
		const sig = polSig(nm);
		let bestEid: string | undefined;
		let bestOv = 0;
		for (const [eid, exp] of store.experiences) {
			if (exp.category === "practices" && exp.kind === "sequence" && polSig(String(exp.name ?? "")) === sig) {
				const ov = seqOverlap(steps, exp.steps ?? []);
				if (ov > bestOv) {
					bestEid = eid;
					bestOv = ov;
				}
			}
		}
		if (bestEid && bestOv >= PRACTICE_DUP_REL) {
			const exp = store.experiences.get(bestEid);
			if (exp) {
				exp.steps = steps.map((s) => s.slice(0, 300));
				exp.last_ts = nowIso();
			}
		} else {
			addPractice(store, nm, steps);
		}
	}
}

export interface LocalMemoryInput {
	envConstraints: Array<{ title: string; text: string }>;
	failures: string[];
	stepLog: Array<{
		step: number;
		command: string;
		prediction: Record<string, unknown>;
		observation: Record<string, unknown>;
		signal: string;
		summary: string;
	}>;
}

function projectCalib(e: LocalMemoryInput["stepLog"][number]): string {
	const cmd = String(e.command ?? "").slice(0, 120);
	const signal = String(e.signal ?? "").slice(0, 100);
	const summary = String(e.summary ?? "").slice(0, 200);
	const pred = JSON.stringify(e.prediction ?? {}).slice(0, 200);
	const obs = JSON.stringify(e.observation ?? {}).slice(0, 200);
	const parts = [`- STEP ${e.step ?? "?"}: 命令=${cmd}; 信号=${signal}`];
	if (summary) parts.push(`  复盘=${summary}`);
	parts.push(`  预测=${pred}; 实际=${obs}`);
	return parts.join("\n");
}

async function promoteCalibToMemory(
	ctx: ExtensionContext,
	store: ExperienceStore,
	entries: LocalMemoryInput["stepLog"],
): Promise<void> {
	if (entries.length === 0) return;
	const block = entries.map(projectCalib).join("\n");
	if (!block.trim()) return;
	const text = await completeText(ctx, PROMPT_CALIB_SYSTEM, block.slice(0, 8000), 700);
	const j = extractJson(text);
	if (!j || !Array.isArray(j.items)) return;
	const toolPending: Array<[string, string]> = [];
	for (const raw of j.items) {
		if (!raw || typeof raw !== "object") continue;
		const it = raw as { category?: unknown; name?: unknown; text?: unknown };
		const cat = String(it.category ?? "").trim();
		const txt = String(it.text ?? "").trim();
		if (!MEM_CATEGORIES.has(cat) || !txt) continue;
		const name = String(it.name ?? "").slice(0, 120);
		if (cat === "tool") {
			const kn = emaKey(name);
			if (!kn) continue;
			const hit = findToolPractice(store, kn);
			if (hit) {
				hit[1].text = txt;
				hit[1].last_ts = nowIso();
				continue;
			}
			toolPending.push([kn, txt]);
			continue;
		}
		let dup = false;
		for (const e of store.experiences.values()) {
			if (e.category === cat && relScore(txt, String(e.text ?? "")) >= MEM_DUP_REL) {
				dup = true;
				break;
			}
		}
		if (!dup) newExp(store, cat as ExperienceCategory, txt, name);
	}
	await dedupeToolLlm(ctx, store, toolPending);
}

async function dedupeToolLlm(
	ctx: ExtensionContext,
	store: ExperienceStore,
	pending: Array<[string, string]>,
): Promise<void> {
	if (pending.length === 0) return;
	const needJudge: Array<{ idx: number; name: string; txt: string; cands: Array<[number, string, string]> }> = [];
	for (const [name, txt] of pending) {
		const cands: Array<[number, string, string]> = [];
		for (const [eid, e] of store.experiences) {
			if (e.category !== "tool") continue;
			const oldTxt = String(e.text ?? "").trim();
			if (!oldTxt) continue;
			const rel = Math.max(relScore(name, String(e.name ?? "")), relScore(txt, oldTxt));
			if (rel >= MEM_DUP_REL) cands.push([rel, eid, oldTxt]);
		}
		if (cands.length === 0) {
			newExp(store, "tool", txt, name, { scope: "pending" });
			continue;
		}
		cands.sort((a, b) => b[0] - a[0]);
		needJudge.push({ idx: needJudge.length, name, txt, cands: cands.slice(0, 3) });
	}
	if (needJudge.length === 0) return;
	const usr: string[] = [];
	for (const item of needJudge) {
		usr.push(`NEW ITEM ${item.idx}: name=${item.name}\n  text=${item.txt.slice(0, 200)}`);
		for (const [rel, eid, oldTxt] of item.cands) {
			usr.push(`  EXISTING ${eid} (rel=${rel.toFixed(2)}): ${oldTxt.slice(0, 120)}`);
		}
	}
	const resp = await completeText(ctx, PROMPT_DEDUPE_TOOL_SYSTEM, usr.join("\n"), 500);
	const j = extractJson(resp);
	const acts = new Map<number, { action: string; target: string; text: string }>();
	for (const raw of (j?.decisions as unknown[]) ?? []) {
		if (!raw || typeof raw !== "object") continue;
		const v = raw as { idx?: unknown; action?: unknown; target?: unknown; text?: unknown };
		const idx = Number(v.idx);
		if (!Number.isFinite(idx)) continue;
		acts.set(idx, {
			action: String(v.action ?? "new")
				.trim()
				.toLowerCase(),
			target: String(v.target ?? ""),
			text: String(v.text ?? "").trim(),
		});
	}
	for (const item of needJudge) {
		const a = acts.get(item.idx) ?? { action: "new", target: "", text: "" };
		if (a.action === "skip") continue;
		if (a.action === "update" && a.target && store.experiences.has(a.target)) {
			const exp = store.experiences.get(a.target);
			if (exp) {
				exp.text = a.text || item.txt;
				exp.last_ts = nowIso();
			}
			continue;
		}
		newExp(store, "tool", item.txt, item.name, { scope: "pending" });
	}
}

export function promoteLocalToMemory(store: ExperienceStore, local: LocalMemoryInput): void {
	for (const c of local.envConstraints ?? []) {
		const s = String(c?.text ?? "").trim();
		const t = String(c?.title ?? "").slice(0, 80);
		if (!s) continue;
		let dup = false;
		for (const e of store.experiences.values()) {
			if (e.category === "constraints" && s.slice(0, 40) && String(e.text ?? "").includes(s.slice(0, 40))) {
				dup = true;
				break;
			}
		}
		if (!dup) addConstraint(store, s, t);
	}
	for (const f of local.failures ?? []) {
		const s = String(f).trim();
		if (s) addFailure(store, s, "");
	}
}

export async function promoteLocalFull(
	ctx: ExtensionContext,
	store: ExperienceStore,
	local: LocalMemoryInput,
): Promise<void> {
	promoteLocalToMemory(store, local);
	await promoteCalibToMemory(ctx, store, local.stepLog ?? []);
}

export async function extractLessons(
	ctx: ExtensionContext,
	store: ExperienceStore,
	instruction: string,
	rows: TrajectoryRow[],
	reward: number,
): Promise<void> {
	if (rows.length === 0) return;
	const outcome = reward === 1 ? "任务成功（reward=1，全测通过）" : "任务失败（reward=0，末步测试未全过）";
	const trajTxt = rows.slice(-40).map((r, i) => {
		const c = String(r.command ?? "").replace(/\n/g, " ");
		const o = headTail((r.outputs ?? []).map(String).join("\n"), 600);
		return `[${i}] improved=${r.improved ?? false} fails=${JSON.stringify(r.fails ?? [])} | ${c.slice(0, 110)}\n  output=${o}`;
	});
	const text = await completeText(
		ctx,
		PROMPT_LESSONS_SYSTEM,
		`TASK:\n${instruction}\nOUTCOME:\n${outcome}\nTRAJECTORY (steps, status, commands):\n${trajTxt.join("\n")}`,
		900,
	);
	const j = extractJson(text);
	if (!j) return;
	const titleText = (item: unknown): [string, string] => {
		if (item && typeof item === "object") {
			const o = item as { title?: unknown; text?: unknown };
			return [String(o.title ?? "").slice(0, 80), String(o.text ?? "").slice(0, 220)];
		}
		return ["", String(item ?? "").slice(0, 220)];
	};
	for (const it of ((j.insights as unknown[]) ?? []).slice(0, 3)) {
		const [t, x] = titleText(it);
		if (x.trim()) addMethod(store, x, t);
	}
	for (const it of ((j.facts as unknown[]) ?? []).slice(0, 2)) {
		const [t, x] = titleText(it);
		if (x.trim()) addConstraint(store, x, t);
	}
}

export async function evalMethodsPhase(
	ctx: ExtensionContext,
	store: ExperienceStore,
	rows: TrajectoryRow[],
	phaseStart: number,
	phaseEnd: number,
	phaseObj: string,
): Promise<void> {
	const phaseRows = rows.filter((r) => {
		const st = Number(r.step ?? 0);
		return st >= phaseStart && st < phaseEnd;
	});
	if (phaseRows.length < 1) return;
	const items: Array<[string, Experience]> = [];
	for (const [eid, e] of store.experiences) {
		const okCat = e.category === "methods" || (e.category === "practices" && e.kind === "sequence");
		if (!okCat) continue;
		if ((e.scope ?? "global") === "task") continue;
		if (!store.injected.has(eid)) continue;
		items.push([eid, e]);
	}
	if (items.length === 0) return;
	items.sort((a, b) => {
		const ca = a[1].conf ?? 0;
		const cb = b[1].conf ?? 0;
		if (ca !== cb) return cb - ca;
		return (b[1].score ?? 0) - (a[1].score ?? 0);
	});
	const top = items.slice(0, 30);
	const trajBlock = phaseRows
		.slice(-20)
		.map(
			(r) =>
				`[${r.step}] improved=${r.improved ?? false} | ${String(r.command ?? "")
					.replace(/\n/g, " ")
					.slice(0, 110)}`,
		)
		.join("\n");
	const user = [
		`SUBTASK OBJECTIVE:\n${phaseObj.slice(0, 800)}`,
		`TRAJECTORY (phase steps, status):\n${trajBlock}`,
		`ITEMS (id: advice):\n${top.map(([eid, e]) => `${eid}: ${expDisplay(e).slice(0, 220)}`).join("\n")}`,
	].join("\n\n");
	const text = await completeText(ctx, PROMPT_EVAL_METHODS_SYSTEM, user, 1200);
	const j = extractJson(text);
	const list = (j?.items ?? j?.methods) as unknown[] | undefined;
	if (!Array.isArray(list)) return;
	for (const raw of list) {
		if (!raw || typeof raw !== "object") continue;
		const v = raw as { id?: unknown; followed?: unknown; helped?: unknown };
		const eid = String(v.id ?? "");
		const exp = store.experiences.get(eid);
		if (!exp) continue;
		const okCat = exp.category === "methods" || (exp.category === "practices" && exp.kind === "sequence");
		if (!okCat) continue;
		if (v.followed) bumpExp(store, eid, Boolean(v.helped));
	}
}

export async function evalConstraintsPhase(
	ctx: ExtensionContext,
	store: ExperienceStore,
	rows: TrajectoryRow[],
	phaseStart: number,
	phaseEnd: number,
	phaseObj: string,
): Promise<void> {
	const phaseRows = rows.filter((r) => {
		const st = Number(r.step ?? 0);
		return st >= phaseStart && st < phaseEnd;
	});
	if (phaseRows.length === 0) return;
	const items: Array<[string, Experience]> = [];
	for (const [eid, e] of store.experiences) {
		if (e.category !== "constraints") continue;
		if ((e.scope ?? "global") === "task") continue;
		if ((e.conf ?? 0) < 1) continue;
		items.push([eid, e]);
	}
	if (items.length === 0) return;
	items.sort((a, b) => {
		const ca = a[1].conf ?? 0;
		const cb = b[1].conf ?? 0;
		if (ca !== cb) return cb - ca;
		return (b[1].score ?? 0) - (a[1].score ?? 0);
	});
	const top = items.slice(0, 20);
	const trajBlock = phaseRows
		.slice(-20)
		.map(
			(r) =>
				`[${r.step}] improved=${r.improved ?? false} | ${String(r.command ?? "")
					.replace(/\n/g, " ")
					.slice(0, 110)}`,
		)
		.join("\n");
	const user = [
		`PHASE OBJECTIVE:\n${phaseObj.slice(0, 500)}`,
		`TRAJECTORY:\n${trajBlock}`,
		`CONSTRAINTS (id: text):\n${top.map(([eid, e]) => `${eid}: ${String(e.text ?? "").slice(0, 200)}`).join("\n")}`,
	].join("\n\n");
	const text = await completeText(ctx, PROMPT_EVAL_CONSTRAINTS_SYSTEM, user, 1200);
	const j = extractJson(text);
	if (!Array.isArray(j?.items)) return;
	for (const raw of j.items as unknown[]) {
		if (!raw || typeof raw !== "object") continue;
		const v = raw as { id?: unknown; relevant?: unknown; respected?: unknown; outcome?: unknown };
		const eid = String(v.id ?? "");
		const exp = store.experiences.get(eid);
		if (!exp || exp.category !== "constraints") continue;
		if (!v.relevant) continue;
		const respected = Boolean(v.respected);
		const outcome = String(v.outcome ?? "neutral");
		const helped = (respected && outcome === "success") || (!respected && outcome === "failure");
		bumpExp(store, eid, helped);
	}
}

/**
 * Graded render for adaptive skill gen: tool strict (rel≥0.5, score≥1, top-5),
 * natural-language loose (rel≥0.3, top-10), same-task pending pass-through (top-5).
 * Total ≤15 lines. Records injected eids on the store.
 *
 * Injection policy: global entries must be cross-task validated (>=2 tasks);
 * pending entries only surface in their own originating task. This keeps
 * single-task/task-specific junk from leaking into unrelated tasks.
 */
export function renderAllExperiences(store: ExperienceStore, instruction = ""): string {
	if (store.experiences.size === 0) return "";
	const q = String(instruction ?? "");
	const globalsTool: Array<[number, string, Experience]> = [];
	const globalsNat: Array<[number, string, Experience]> = [];
	const pendNat: Array<[number, string, Experience]> = [];
	for (const [eid, exp] of store.experiences) {
		if ((exp.score ?? 0) < 0) continue;
		const cat = exp.category;
		const scope = exp.scope ?? "global";
		const doc = `${String(exp.name ?? "")} ${expDisplay(exp)}`;
		const rel = relScore(q, doc);
		if (scope === "global") {
			if (!crossTaskValidated(exp)) continue;
			if (cat === "tool") {
				if (rel >= TOOL_REL_MIN && (exp.score ?? 0) >= 1) globalsTool.push([rel, eid, exp]);
			} else if ((NAT_CATS as readonly string[]).includes(cat) && rel >= NAT_REL_MIN) {
				globalsNat.push([rel, eid, exp]);
			}
		} else if (scope === "pending" && (NAT_CATS as readonly string[]).includes(cat)) {
			if (!store.task || !(exp.tasks ?? []).includes(store.task)) continue;
			if (rel >= NAT_REL_MIN) pendNat.push([rel, eid, exp]);
		}
	}
	const byScore = (a: [number, string, Experience], b: [number, string, Experience]): number => {
		const ra = a[0] - b[0];
		if (ra !== 0) return ra;
		const sa = (a[2].score ?? 0) - (b[2].score ?? 0);
		if (sa !== 0) return sa;
		return (a[2].conf ?? 0) - (b[2].conf ?? 0);
	};
	globalsTool.sort(byScore);
	globalsNat.sort(byScore);
	pendNat.sort(byScore);
	const injectTool = globalsTool.slice(0, 5);
	const injectNat = globalsNat.slice(0, 10);
	const injectPend = pendNat.slice(0, 5);
	const line = (tag: string, exp: Experience): string =>
		`- [${tag}] ${expDisplay(exp).slice(0, 180)} (score=${exp.score ?? 0}, conf=${exp.conf ?? 0})`;
	const lines: string[] = [];
	for (const [, , exp] of injectTool) lines.push(line(`${exp.category}:global`, exp));
	for (const [, , exp] of injectNat) lines.push(line(`${exp.category}:global`, exp));
	for (const [, , exp] of injectPend) lines.push(line(`${exp.category}:pending(same-task,unverified)`, exp));
	if (lines.length === 0) return "";
	for (const [, eid] of [...injectTool, ...injectNat, ...injectPend]) store.injected.add(eid);
	return `\nEXPERIENCE MEMORY (cross-task validated unless marked pending; verify pending items before relying):\n${lines.join("\n")}`;
}

/** Persist the eids injected this run (top prompt + per-step practice hits) to the run dir. */
export function dumpInjected(store: ExperienceStore, task?: string, rep?: number, phase = "unknown"): void {
	if (!task) return;
	try {
		const dir = `/tmp/vfe/${task}_rep${rep ?? 0}`;
		mkdirSync(dir, { recursive: true });
		const entries = [...store.injected].map((eid) => {
			const e = store.experiences.get(eid);
			return {
				eid,
				category: e?.category,
				scope: e?.scope,
				score: e?.score,
				conf: e?.conf,
				tasks: e?.tasks,
				text: String(e?.text ?? e?.name ?? "").slice(0, 300),
			};
		});
		appendFileSync(
			`${dir}/memory_injected.jsonl`,
			`${JSON.stringify({ ts: nowIso(), task, rep: rep ?? 0, phase, count: entries.length, entries })}\n`,
			"utf8",
		);
	} catch {
		// best-effort observability
	}
}

export function maybePeriodicSave(store: ExperienceStore, step: number): void {
	if (step > 0 && step % MEM_SAVE_EVERY === 0) {
		saveExperiences(store.experiences, store.nextId, store.path);
	}
}

export interface PracticeHit {
	eid: string;
	name: string;
	steps: string[];
}

/**
 * Mature global sequence practices relevant to the current objective
 * (raw23 _mature_practice_hits): score/conf ≥ 3, ranked by rel(score, conf),
 * top PRACTICE_INJECT_MAX; injected eids recorded for eval-methods phase.
 */
export function maturePracticeHits(store: ExperienceStore, query: string, limit = PRACTICE_INJECT_MAX): PracticeHit[] {
	const scored: Array<{ rel: number; score: number; conf: number; hit: PracticeHit }> = [];
	for (const [eid, exp] of store.experiences) {
		if (exp.category !== "practices" || exp.kind !== "sequence") continue;
		if ((exp.scope ?? "global") !== "global") continue;
		if (!crossTaskValidated(exp)) continue;
		if ((exp.score ?? 0) < PRACTICE_MIN_SCORE || (exp.conf ?? 0) < PRACTICE_MIN_CONF) continue;
		const steps = (exp.steps ?? []).map(String).filter(Boolean);
		if (steps.length === 0) continue;
		const rel = relScore(query, `${String(exp.name ?? "")} ${expDisplay(exp)}`);
		scored.push({
			rel,
			score: exp.score ?? 0,
			conf: exp.conf ?? 0,
			hit: { eid, name: String(exp.name ?? ""), steps },
		});
	}
	scored.sort((a, b) => b.rel - a.rel || b.score - a.score || b.conf - a.conf);
	const hits = scored.slice(0, limit).map((s) => s.hit);
	for (const h of hits) store.injected.add(h.eid);
	return hits;
}

const PROMPT_LESSONS_SYSTEM = `You are an experiential-learning module. Given a completed coding task's execution trajectory and its final outcome, extract 2-3 GENERALIZABLE LESSONS that would help solve FUTURE similar tasks. Two types: (1) insights: 'when <situation>, do <action> (because <reason>)' — transferable methods that worked or that failed; (2) facts: STATIC environment/task boundary truths the agent must respect (e.g. 'no GPU available', 'sandbox forbids writing outside /app', 'delivery must land at /app/results/X'). RULE: facts are conditions to be AWARE of, NOT procedures — do NOT output 'use X' / 'install Y' / 'add --flag' (those are methods/tools, not facts). RULE: do NOT output textbook facts or definitions — only output things learned from THIS task's outcome that a competent engineer wouldn't already know. Do NOT give task-specific file names or values. Do NOT give advice that only applies to this exact task. Each lesson has a SHORT TITLE (≤10 words) and the TEXT. Output ONLY JSON {"insights": [{"title": "...", "text": "..."}], "facts": [{"title": "...", "text": "..."}]}.`;

const PROMPT_EVAL_METHODS_SYSTEM = `You are an experiential-learning evaluator. Given a coding subtask's objective, its execution trajectory (commands + test pass-rate trend), and candidate strategy lessons (methods) plus reusable command sequences (practices), decide for each item whether the agent FOLLOWED its advice in this subtask and whether following it HELPED make progress. Judge adherence ONLY from the commands actually executed in the trajectory. Output ONLY JSON {"items": [{"id": "...", "followed": true/false, "helped": true/false}, ...]} — one object per listed item. If an item was not followed, set followed=false and helped=false.`;

const PROMPT_EVAL_CONSTRAINTS_SYSTEM = `You are a constraint evaluator. Given a coding subtask's trajectory and a list of environment constraints (facts like 'pip needs --break-system-packages', 'no GPU'), decide for each constraint: (1) was it RELEVANT to any command in this phase? (2) if relevant, did the agent RESPECT it? (3) what was the OUTCOME? Output ONLY JSON {"items": [{"id": "...", "relevant": true/false, "respected": true/false, "outcome": "success/failure/neutral"}, ...]}.`;

const PROMPT_CALIB_SYSTEM = `下面是某任务每步的【预测 vs 实际】原始数据（含命令、改善/未改善、回归/修复信息）。请提炼对未来任务有价值的、任务无关的通用教训，写成经验条目。每条从 {methods, failures, constraints, tool, practices} 选类别，写简洁教训；跳过一次性噪声。RULE: 类别为 tool 时，name 必须给出该命令族的代表性命令模板（动词+关键参数，路径归一化为 /app/F、数字归一化为 N），例如 "pytest -x"；text 写该命令族的用法知识（怎么做/何时用/边界），不写"好用/不好用"之类的效果结论——效果由 score 统计表达。其他类别 name 可省略或简短。只返回 JSON {"items": [{"category": "...", "name": "...", "text": "..."}]}；无则 {"items": []}。`;

const PROMPT_DEDUPE_TOOL_SYSTEM = `You are a memory deduplication judge. A memory bank stores per-command-family usage knowledge, one entry per distinct usage pattern. Given NEW tool experiences and their closest EXISTING entries, decide for each NEW item exactly one action: "new" (genuinely different usage/insight — create a separate entry), "update" (refines/extends the same usage — merge into the given existing entry), "skip" (duplicates an existing entry or is low-value noise). For "update" you may provide merged_text (deduplicated combined text); omit it to keep existing text and only refresh timestamp. Output ONLY JSON {"decisions": [{"idx": 0, "action": "new"}, {"idx": 1, "action": "update", "target": "<existing eid>", "text": "..."}]}.`;
