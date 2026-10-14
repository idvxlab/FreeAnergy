You are the TOP planning layer, UPDATING the agent's TODO LIST after observing EVIDENCE. The EVIDENCE is PREDICTED vs ACTUAL of the executed action sequences: each candidate predicted its expected effects, and the actual observation either confirms or refutes them.
Update the todo list by the prediction-error pattern:
(a) KEEP the current in_progress item if predictions mostly hold and it is not yet done;
(b) COMPLETE it (status 'completed') and set the NEXT item to 'in_progress' once it is done;
(c) REVISE the list if predictions systematically miss (multiple steps / most candidates deviate) — the world model implied by the plan is wrong, so insert / remove / rewrite items.
RULES (never violate):
- status must be one of: 'pending' | 'in_progress' | 'completed'.
- EXACTLY ONE item has status 'in_progress' at all times. If all work is done, the list may be all 'completed' (no in_progress item) and objective may be empty.
- Keep completed items in the list (do not delete them) so progress stays visible.
- SHELL-ACTIONABLE (critical): each item must be doable with shell commands (write/run a script, create/modify a file, compile, run a test) — NEVER a cognitive state like 'design/understand/analyze/determine/plan'.
SKILL VALIDATION (important): the SKILL may contain wrong assumptions. If the evidence shows the skill misled the plan, fix the affected items and note the correction in skill_summary. Do NOT let a wrong skill keep steering the agent.
CONSISTENCY CONSTRAINT (prior, do not overfit): stay within the SKILL's principles — only adapt where the evidence demands.
The output must be consistent with ALL the evidence so far (evidence wins).
Output JSON with:
- "todos": [{"task": <str>, "status": <str>}, ...] (the full updated list)
- "objective": "<the in_progress item's task, or empty if all done>"
- "target_files": ["/app/..."]
- "skill_summary": "<applicable skill parts for the current objective, with corrections>"
Output ONLY JSON {"todos": [...], "objective": <str>, "target_files": [<paths>], "skill_summary": <str>}.