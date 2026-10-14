Close the step turn: record YOUR assessment of THIS step — better | worse | same — after your chosen sequence ran and you observed the result (test output, assertion numbers, artifact diffs).
evidence (<=160 chars): the concrete content backing your verdict (numbers, test names, observed diffs).
Fact lines REGRESSION/IMPROVEMENT come from real test-list changes and override your verdict; if they contradict it, follow the facts.
Call once before ending the turn; a later call in the same turn overwrites the earlier one. Settle consumes it the same turn, so the verdict always refers to this step.
Omitting it yields a facts-only (neutral) signal for this step. Requires a selected sequence this turn (run and observe first).