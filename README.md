# VFE — Verified Forward Execution

A [pi](https://github.com/earendil-works/pi) extension that wraps Terminal-Bench 2.1 tasks in a three-phase agent protocol:

| Phase | Purpose |
|---|---|
| **TOP** | Synthesize an adaptive skill + todo plan from the task and environment evidence |
| **STEP** | Propose 2–3 candidate action sequences, execute one, observe, assess |
| **END** | Finish with a summary; experience is sedimented into cross-task memory |

## Features

- **Candidate-sequence planning** — the agent must propose and rank multiple action sequences per step before acting
- **Isolated in-loop verification** — task tests are injected into a private mount namespace so the agent never sees them
- **Cross-task experience memory** — validated lessons accumulate across runs (`VFE_MEMORY_PATH`)
- **Editable prompts** — all model-facing prompt text lives in plain `.md` files; no code changes needed to retune behavior

## Repository layout

```
extension/    pi extension source (TypeScript + Docker sidecar Python)
prompts/      editable prompt templates (.md)
tests/        unit tests for prompt loading
```

## Requirements

- [pi](https://github.com/earendil-works/pi) checkout with dependencies installed
- Docker daemon
- Terminal-Bench 2.1 task directories (each task has `task.toml`, `instruction.md`, `tests/`)
- A configured pi provider (API key or OAuth)

## Installation

```bash
# clone this repository
git clone https://github.com/idvxlab/FreeAnergy.git
cd FreeAnergy

# load the extension from your pi checkout
export VFE_TASKS_DIR=/path/to/tb21/tasks
export VFE_MEMORY_PATH=/tmp/vfe_memory.json   # optional; keeps runs isolated
/path/to/pi/pi-test.sh -e "$PWD/extension"
```

## Usage

Inside a pi session:

```
/vfe-start <task-dir-name> --rep <N>
```

## Prompts

All model-facing prompt text lives in [`prompts/`](prompts/). Edit these files to change agent behavior — no TypeScript changes required.

| File | Used for |
|---|---|
| `skill_format.md` | Skill document format rules (inlined into stable/adaptive) |
| `stable_skill_system.md` | TOP system prompt, no-memory variant |
| `adaptive_skill_system.md` | TOP system prompt, with experience memory |
| `ground_skill_system.md` | Grounding verifier (strip unsupported claims) |
| `propose_plan_description.md` | `propose_plan` tool description |
| `propose_sequences_description.md` | `propose_sequences` tool description |
| `assess_step_description.md` | `assess_step` tool description |
| `vfe_system_prompt.md` | Full-replacement system prompt during VFE |
| `revise_states_system.md` | Todo-list revision after settle feedback |
| `reflect_system.md` | Per-step reflection JSON prompt |

Placeholders `{{SKILL_FORMAT}}` and `{{OBSERVE_BUDGET}}` are substituted at load time. Prompt files are resolved relative to the extension directory (`extension/../prompts/`), never from the process working directory. Override with `VFE_PROMPTS_DIR`.

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `VFE_TASKS_DIR` | `tb21/tasks` | Root of TB2.1 task directories |
| `VFE_MEMORY_PATH` | `vfe_memory.json` | Cross-task experience memory file |
| `VFE_PROMPTS_DIR` | `extension/../prompts` | Override prompt file directory |
| `VFE_MAX_STEPS` | `25` | Max settle steps per task |
| `VFE_OBSERVE_BUDGET` | `4` | Observation calls allowed after the chosen sequence runs |
| `VFE_PYTHON` | `python3` | Python used to launch the Docker sidecar |
| `VFE_SEQ_TIMEOUT` | `600` | Per-sequence exec timeout (seconds) |

## Tests

```bash
# using tsx from a pi checkout
/path/to/pi/node_modules/.bin/tsx --test tests/prompts.test.ts

# or with Node >= 22.6
node --experimental-strip-types --test tests/prompts.test.ts
```

## License

MIT — see [LICENSE](LICENSE).
