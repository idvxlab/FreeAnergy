const ALLOWED_ROOTS = ["/app"] as const;

/** Resolve a tool path against the container workdir and enforce the /app sandbox.
 *  /tests is hidden from the agent so the acceptance suite cannot be read as an answer key. */
export function resolveContainerPath(input: string): string {
	const raw = input.trim() || ".";
	const abs = raw.startsWith("/") ? normalizePosix(raw) : normalizePosix(`/app/${raw}`);
	for (const root of ALLOWED_ROOTS) {
		if (abs === root || abs.startsWith(`${root}/`)) {
			return abs;
		}
	}
	throw new Error(`Path not allowed: ${input} → ${abs}. Use paths under /app (workdir).`);
}

export function assertWritable(absPath: string): void {
	if (absPath === "/tests" || absPath.startsWith("/tests/")) {
		throw new Error(`Path is read-only (tests mount): ${absPath}`);
	}
}

function normalizePosix(p: string): string {
	const parts: string[] = [];
	for (const seg of p.split("/")) {
		if (!seg || seg === ".") continue;
		if (seg === "..") {
			parts.pop();
			continue;
		}
		parts.push(seg);
	}
	return `/${parts.join("/")}`;
}

/** Head-tail truncate to maxBytes, marking truncation for the model. */
export function truncateOutput(text: string, maxBytes = 8000): string {
	const buf = Buffer.from(text, "utf8");
	if (buf.length <= maxBytes) return text;
	const head = buf.subarray(0, Math.floor(maxBytes * 0.6));
	const tail = buf.subarray(buf.length - Math.floor(maxBytes * 0.3));
	const dropped = buf.length - head.length - tail.length;
	return `${head.toString("utf8")}\n…[truncated: showing head+tail of ${buf.length} bytes, ${dropped} omitted. Full output was not kept — redirect to a file (e.g. \`> /app/out.txt\`) and page it with read offset/limit, or narrow with grep/head/tail]…\n${tail.toString("utf8")}`;
}
