import { appendFileSync, mkdirSync } from "node:fs";
import { dirname } from "node:path";
import type { ExperienceStore, LocalMemoryInput } from "./memory.ts";

export const EFE_DEBUG_PATH = process.env.VFE_SEL_DEBUG || "/tmp/vfe_selection_debug.jsonl";

function ensureDebugDir(path: string): void {
	try {
		const dir = dirname(path);
		if (dir && dir !== ".") mkdirSync(dir, { recursive: true });
	} catch {
		// best-effort
	}
}

export function appendEfeDebug(row: Record<string, unknown>): void {
	try {
		ensureDebugDir(EFE_DEBUG_PATH);
		appendFileSync(EFE_DEBUG_PATH, `${JSON.stringify({ ts: new Date().toISOString(), ...row })}\n`, "utf8");
	} catch {
		// debug must never break the run
	}
}

// Re-export for settle wiring (kept here so index does not need extra imports).
export type { ExperienceStore, LocalMemoryInput };
