import type { ExtensionContext } from "@earendil-works/pi-coding-agent";

function textOf(response: { content: Array<{ type: string; text?: string }> }): string {
	return response.content
		.filter((c): c is { type: "text"; text: string } => c.type === "text" && typeof c.text === "string")
		.map((c) => c.text)
		.join("\n");
}

/** Nested single-turn complete via the session model. Returns "" when unavailable or aborted. */
export async function completeText(
	ctx: ExtensionContext,
	system: string,
	user: string,
	maxTokens = 4000,
): Promise<string> {
	const model = ctx.model;
	if (!model) return "";
	try {
		const response = await ctx.modelRegistry.complete(
			model,
			{
				systemPrompt: system,
				messages: [
					{
						role: "user" as const,
						content: [{ type: "text" as const, text: user }],
						timestamp: Date.now(),
					},
				],
			},
			{ maxTokens, cacheRetention: "none", signal: ctx.signal },
		);
		if (response.stopReason === "aborted") return "";
		return textOf(response).trim();
	} catch {
		return "";
	}
}
