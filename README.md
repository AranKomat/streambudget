# StreamBudget

**API-first, cost-aware, evidence-preserving runtime for continuous video agents.**

This is a working research implementation, not a trained model or a production surveillance product.
Start with a hosted vision model; later point the same adapter at a compatible vLLM/SGLang endpoint.
No SFT, RL, native-audio model, proprietary dataset, or GPU is required for the local integration demo.

The research question is: **can selective perception, shared observations, and persistent evidence
reduce total inference cost while preserving event recall, historical reasoning, and delivered latency?**
The repository does not claim that this has already been demonstrated.

## Read these first

| File | Purpose |
|---|---|
| [SPEC.md](SPEC.md) | Self-contained product hypothesis, architecture, semantics, implementation scope, evaluation, and references |
| [HANDOFF.md](HANDOFF.md) | Exact work remaining for a research agent with APIs/GPUs, commands, acceptance gates, and non-negotiable safeguards |
| [docs/FEATURE_STATUS.md](docs/FEATURE_STATUS.md) | Implemented vs contract-tested vs not implemented/externally validated |
| [docs/BENCHMARKS.md](docs/BENCHMARKS.md) | Public-benchmark adapters, honest baselines, domain evaluation, label separation |
| [docs/REFERENCES.md](docs/REFERENCES.md) | Primary-source references and what they actually support |
| [docs/API.md](docs/API.md) | Live ingestion and query endpoints |

## 1. Run without any API key

Linux/macOS, Python 3.11+. Python 3.13 is the locally tested interpreter.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,server]'
pytest -q
streambudget demo --out runs/demo
```

Open `runs/demo/run/report.html`. The demo generates its own images, sensor observations,
watch definitions, and separate answer keys. It monitors a red box's dwell, reacts to a temperature
predicate without an LLM, and investigates earlier evidence through search and inspection.

**The backend in this demo is an explicit synthetic red-pixel test double, not a VLM.** Its scores
and call counts establish plumbing behavior only. They do not establish accuracy, savings, or a moat.

With existing dependencies, installation is not required for development:

```bash
PYTHONPATH=src python -m streambudget.cli demo --out runs/source-demo
```

## 2. Use a real vision-model API

Choose an endpoint that supports Chat Completions, image inputs, and JSON output. Model IDs are
intentionally configurable; no unverified “latest” model or price is hard-coded.

```bash
cp .env.example .env
# Edit VLM_BASE_URL, VLM_MODEL, VLM_API_KEY.
set -a; source .env; set +a

# Explicit network opt-in; this probe can incur provider charges.
streambudget doctor --config configs/api.yaml --probe \
  --allow-network --out runs/api-probe

# Reuse synthetic input to check end-to-end wiring with your actual model.
# This is still a tiny synthetic test, not evidence of real-world quality.
streambudget run --events runs/demo/input/events.jsonl \
  --tasks runs/demo/input/tasks.jsonl --config configs/api.yaml \
  --out runs/api-demo --allow-network
streambudget score --run runs/api-demo --labels runs/demo/input/labels.jsonl
streambudget report --run runs/api-demo
```

`configs/api.yaml` caps attempts at 80 for the entire run, including planning, retries, and workers.
A separate `planner` can be text-only; `perception` and `verifier` must support images. Optional `memory`,
`embedding`, and `asr` roles can use different endpoints. See `configs/multi-role.example.yaml`.

Without configured prices, costs are **unknown**, not free. To enable a dollar budget, add actual
`prices.input_per_million`, `prices.output_per_million`, optionally `prices.cached_input_per_million`,
and set `budget.max_usd`. Reservations are estimates; see [billing semantics](docs/OPERATIONS.md).

### OpenRouter with a shared credential file

`configs/openrouter.yaml` provides a small vision-model compatibility configuration
with eight attempts, no retries, and a $0.25 estimated admission limit. Model and
prices were checked against OpenRouter's catalog on 2026-09-20; refresh both before
experiments. This is not a benchmark model selection or a guaranteed billing cap.

The launcher reads `OPENROUTER_API_KEY` from the trusted `../.env` file (the shared
`gpu/.env` in this workspace), or from the existing environment. Override the file
with `STREAMBUDGET_ENV_FILE`. Credentials are not copied into this repository.

```bash
# Default: configuration/dependency check, no inference request.
bash scripts/openrouter.sh

# Explicit paid image/JSON compatibility probe when ready.
bash scripts/openrouter.sh doctor --config configs/openrouter.yaml \
  --probe --allow-network --out runs/openrouter-probe
```

### Bounded multi-model compatibility screen

`configs/screening.yaml` selects Gemini 3.8 Flash, Qwen3.8-27B, and GLM-5.3-Flash.
`scripts/screen_models.py` sends each model the same 20 generated eight-image packets,
after one single-image probe per model. It uses **one shared $3 estimated budget and
63-attempt ceiling**, including the probes, with no retries. This is a synthetic
image/JSON/temporal-reasoning screen, not StreamArena or a real-video quality ranking.

```bash
python scripts/screen_models.py prepare --out runs/model-screen
# Inspect contact.png and estimate.json; export OPENROUTER_API_KEY from your trusted environment.
python scripts/screen_models.py probe --out runs/model-screen --allow-network
# Inspect probe/receipt.json before continuing. All three probes must pass.
python scripts/screen_models.py screen --out runs/model-screen --allow-network
python scripts/screen_models.py report --out runs/model-screen
```

Cases and labels are stored separately. Failed responses stay in the denominator;
unknown charges remain held against the cumulative budget. Reusing a phase directory
fails to prevent accidental duplicate dispatch. After interruption, inspect the trace
and reconcile any uncertain attempts before authorizing another run; this script does
not provide automatic durable resume. Rates and model revisions are not provider-pinned.
The [first screening report](research/MODEL_SCREEN_20260920.md) records the actual
results and explains why this tiny synthetic set cannot select a video-quality winner.

The [real-footage follow-up](research/REAL_SCREEN_20260920.md) uses five openly licensed
clips, 20 matched questions and nine concurrent requests. `configs/footage-screening.yaml`
pins Gemini to **Google AI Studio Flex**, with no Vertex or standard-tier fallback;
the runner checks the returned provider/tier rather than just assuming the request
was honored. Other models retain automatic routing. Use `--dataset` to select the
prepared footage and `--concurrency` to set up to 12 simultaneous calls under the
same shared ledger. The report includes download/reproduction and attribution details.

## 3. Bring a recording

Install `ffmpeg` or `pip install -e '.[video]'` (PyAV). Both paths retain presentation timestamps.

```bash
streambudget prepare-video --video /path/to/authorized-recording.mp4 \
  --out data/recording --source camera --fps 4 --max-seconds 120
```

Create `data/recording/tasks.jsonl`:

```json
{"type":"watch","at":0,"id":"package_wait","source":"camera","watch":{"goal":"A package occupies the marked work area","dwell_seconds":10,"max_observation_gap":5}}
{"type":"ask","at":119,"id":"investigation","source":"camera","question":"What occupied the work area earlier? Inspect evidence and cite its timestamps."}
```

```bash
streambudget validate-data --events data/recording/events.jsonl --tasks data/recording/tasks.jsonl
streambudget run --events data/recording/events.jsonl --tasks data/recording/tasks.jsonl \
  --config configs/api.yaml --out runs/recording --allow-network
```

Set task times to the actual recording length. `--timing deterministic` is causal but pauses between
observations for reproducibility; it is **not** a throughput benchmark. `--timing realtime --speed 1`
keeps the source clock moving. Its predecoded-manifest path still excludes original video decoding,
which is reported separately in `preparation.json`.

For **archive-only investigation**, use a tasks file containing only `ask` entries and set
`--timing archive`. All source observations become available first, no eager captioning is performed,
and the agent selectively inspects the recording. Archive mode does not accept live watch tasks.

**Sampling ceiling:** preparing at 1 FPS permanently limits this run to those frames. Asking the agent
for 16 frames cannot recover discarded observations. For fast-event tests, retain sufficient input
FPS or integrate a dense clip store. Camera-rate control and lazy original-video clip reacquisition
are not implemented in v0.

## 4. Live input

```bash
export STREAMBUDGET_TOKEN='a-long-random-secret'
streambudget serve --config configs/api.yaml --out runs/live --allow-network
```

Frames and detector/OCR/ASR/sensor events can be pushed to the authenticated REST API. A trusted bridge
for local-video/RTSP inputs is included (requires PyAV):

```bash
streambudget bridge-video --input /path/to/authorized-recording.mp4 \
  --server http://127.0.0.1:8765 --source camera --fps 4 --log runs/bridge.json
```

Register a watch through `POST /v1/watches`; obtain results through `/v1/events` and `/v1/query`.
This is a single-tenant research server. Do not expose it directly to untrusted networks.
See [API.md](docs/API.md) and [SECURITY.md](SECURITY.md).

## 5. Run the baseline matrix

```bash
python scripts/run_matrix.py \
  --events runs/demo/input/events.jsonl --tasks runs/demo/input/tasks.jsonl \
  --labels runs/demo/input/labels.jsonl --out runs/matrix
```

Variants: fixed-rate background VLM, motion-plus-periodic refresh, adaptive event/dwell/refresh gating,
and recent-window-only QA. These are **our controlled baselines**, not reimplementations claiming
StreamMind/SimpleStream/VSS paper equivalence. Real baseline runs require `--config ... --allow-network`.
Each variant has a separate budget; the total matrix may consume four budgets.

Video-MME import/export and a StreamArena lifecycle bridge are included. Official dataset/model runs,
NIST ActEV scoring, FPS-Bench, and upstream StreamMind/VSS comparisons remain in [HANDOFF.md](HANDOFF.md).

## 6. Move to your GPUs

Run `scripts/serve_vllm.sh` or `scripts/serve_sglang.sh` in a **separate, version-pinned GPU environment**.
Then set `LOCAL_BASE_URL` and `VLM_MODEL` and use `configs/local.yaml`.

The current adapter sends **timestamped image sequences**. It does not automatically enable native
video EVS/VidCom2, incremental prefill, encoder disaggregation, or visual/KV cache reuse. Those are
specific backend experiments, not benefits this repository claims to implement.

## Output files

`run.json` includes settings, events, answers, timings and counters. `trace.jsonl` records stages,
model attempts, coalescing, queue delays, errors and drops. `memory.sqlite` stores source evidence,
transitive lineage, searchable derived views and optional vectors. `media/` preserves uploaded image
bytes. `config.resolved.json` contains environment-variable **names**, never API-key values.
Prompts/media are not copied into operational traces; evidence and answers are still sensitive data.

## License and provenance

Original implementation: [MIT](LICENSE). No upstream StreamMind/OmniAgent/VSS source, weights,
benchmark footage, or noncommercial annotations are bundled. Published schemas and designs are
referenced in [REFERENCES.md](docs/REFERENCES.md). Check licenses for every model, service, and dataset
used in your deployment. See [THIRD_PARTY.md](THIRD_PARTY.md).
