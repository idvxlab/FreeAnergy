import { strict as assert } from "node:assert";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

const REPO_ROOT = join(fileURLToPath(new URL(".", import.meta.url)), "..");
const PROMPTS_DIR = join(REPO_ROOT, "prompts");

const EXPECTED_PROMPTS = [
	"skill_format",
	"stable_skill_system",
	"adaptive_skill_system",
	"ground_skill_system",
	"propose_plan_description",
	"propose_sequences_description",
	"assess_step_description",
	"vfe_system_prompt",
	"revise_states_system",
	"reflect_system",
];

test("all 10 prompt files exist and are non-empty", () => {
	for (const name of EXPECTED_PROMPTS) {
		const p = join(PROMPTS_DIR, `${name}.md`);
		assert.ok(existsSync(p), `missing prompt file: ${p}`);
		const text = readFileSync(p, "utf8").trim();
		assert.ok(text.length > 0, `empty prompt file: ${p}`);
	}
});

test("prompts load from extension-relative dir with placeholder substitution (cwd-independent)", async () => {
	// Change cwd to prove loading does not depend on process.cwd().
	process.chdir("/tmp");
	process.env.VFE_OBSERVE_BUDGET = "5";
	const prompts = await import("../extension/prompts.ts");

	assert.ok(prompts.SKILL_FORMAT.includes("NO-COMMAND RULE"), "SKILL_FORMAT loaded");
	assert.ok(prompts.STABLE_SKILL_SYSTEM.includes("NO-COMMAND RULE"), "SKILL_FORMAT inlined into stable");
	assert.ok(!prompts.STABLE_SKILL_SYSTEM.includes("{{SKILL_FORMAT}}"), "placeholder substituted in stable");
	assert.ok(!prompts.ADAPTIVE_SKILL_SYSTEM.includes("{{SKILL_FORMAT}}"), "placeholder substituted in adaptive");
	assert.ok(prompts.PROPOSE_SEQUENCES_DESCRIPTION.includes("at most 5 calls"), "OBSERVE_BUDGET substituted");
	assert.ok(!prompts.PROPOSE_SEQUENCES_DESCRIPTION.includes("{{OBSERVE_BUDGET}}"), "no leftover placeholder");
	assert.ok(prompts.VFE_SYSTEM_PROMPT.includes("at most 5"), "OBSERVE_BUDGET substituted in system prompt");
	assert.ok(prompts.REFLECT_SYSTEM.includes("预测"), "reflect prompt loaded");
	assert.ok(prompts.GROUND_SKILL_SYSTEM.length > 0, "ground prompt loaded");
	assert.ok(prompts.REVISE_STATES_SYSTEM.length > 0, "revise prompt loaded");
	assert.ok(prompts.ASSESS_STEP_DESCRIPTION.length > 0, "assess description loaded");
	assert.ok(prompts.PROPOSE_PLAN_DESCRIPTION.length > 0, "propose plan description loaded");
	assert.equal(prompts.PROPOSE_PLAN_DESCRIPTION.includes("\n"), false, "descriptions joined to single line");
});

test("promptsDir respects VFE_PROMPTS_DIR env override", async () => {
	process.env.VFE_PROMPTS_DIR = "/nonexistent/custom-prompts";
	const { promptsDir } = await import("../extension/prompts.ts");
	assert.equal(promptsDir(), "/nonexistent/custom-prompts");
	delete process.env.VFE_PROMPTS_DIR;
	assert.ok(promptsDir().endsWith("/prompts"), "default resolves to prompts dir next to extension");
	assert.ok(!promptsDir().startsWith("tb21"), "default is absolute, not cwd-relative");
});
