# VFE

VFE (Verified Forward Execution) is a [pi](https://github.com/earendil-works/pi) extension for solving Terminal-Bench 2.1 tasks with an explicit plan–act–assess loop.

Instead of free-form tool calling, the agent works in three phases:

1. **TOP** — read the task and environment evidence, then produce an adaptive skill and a short todo plan
2. **STEP** — each turn: propose 2–3 candidate command sequences, run one, observe, assess better/worse/same
3. **END** — finish with a summary; validated lessons are written to cross-task memory

## Features

- Candidate-sequence planning with explicit weighing of expected effect, exploration, and deviation
- Isolated in-loop verification (task tests never appear in the agent-visible filesystem)
- Cross-task experience memory shared across runs
- All model prompts live in plain Markdown under `prompts/` — edit behavior without touching TypeScript

## Requirements

- [pi](https://github.com/earendil-works/pi) with dependencies installed
- Docker
- Terminal-Bench 2.1 task directories (`task.toml`, `instruction.md`, `tests/` per task)
- A configured pi provider (API key or OAuth)

## Getting started

```bash
git clone https://github.com/idvxlab/FreeAnergy.git
cd FreeAnergy

export VFE_TASKS_DIR=/path/to/tb21/tasks
/path/to/pi/pi-test.sh -e "$PWD/extension"
```

Optional:

```bash
export VFE_MEMORY_PATH=/tmp/vfe_memory.json   # isolate experience memory per experiment
```

## Usage

Inside a pi session:

```text
/vfe-start <task-dir-name> --rep <N>
```

## Project structure

```text
extension/   pi extension (TypeScript + Docker sidecar)
prompts/     prompt templates loaded by extension/prompts.ts
tests/       unit tests
```

## Prompts

Edit the Markdown files in [`prompts/`](prompts/) to change how the agent plans, proposes sequences, or reflects. Placeholders `{{SKILL_FORMAT}}` and `{{OBSERVE_BUDGET}}` are filled in at load time. Files are resolved relative to the extension directory (or `VFE_PROMPTS_DIR`), never from the current working directory.

## Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `VFE_TASKS_DIR` | `tb21/tasks` | Root of TB2.1 task directories |
| `VFE_MEMORY_PATH` | `vfe_memory.json` | Experience memory file |
| `VFE_PROMPTS_DIR` | `extension/../prompts` | Prompt directory override |
| `VFE_MAX_STEPS` | `25` | Max steps per task |
| `VFE_OBSERVE_BUDGET` | `4` | Observations allowed after the chosen sequence |
| `VFE_PYTHON` | `python3` | Python for the Docker sidecar |
| `VFE_SEQ_TIMEOUT` | `600` | Per-sequence timeout (seconds) |

## Tests

```bash
/path/to/pi/node_modules/.bin/tsx --test tests/prompts.test.ts
# or, with Node >= 22.6
node --experimental-strip-types --test tests/prompts.test.ts
```

## License

MIT — see [LICENSE](LICENSE).
