# Research-agent handoff: StreamBudget

Version 0.1.0 • 2026-09-20

## Mission, without needing the prior conversation

Turn this API-first research runtime into a **measured cost-quality advantage for always-on video
agents**. It should monitor standing goals, combine video with structured signals, preserve searchable
evidence, and investigate historical events without sending every frame through an expensive model.
Use open models through hosted APIs first; self-host behind vLLM/SGLang after identifying promising
policies. No training is required to begin. The intended initial buyer is a video-analytics software
vendor, not a robot controller.

Read [SPEC.md](SPEC.md), [FEATURE_STATUS.md](docs/FEATURE_STATUS.md), [BENCHMARKS.md](docs/BENCHMARKS.md),
and [VALIDATION.md](results/VALIDATION.md). The repository is original code, not a fork or reproduction
of StreamMind, OmniAgent, or VSS. The upstream adapter has passed a generated-video smoke run
inside the unmodified pinned driver. That does not reproduce StreamMind or a full official score.

## What was completed locally

There is working source for causal replay/archive investigation, authenticated live frame/signal
input, typed watches and scalar predicates, shared visual observations, bounded scheduling, an
agent tool loop, SQLite evidence/lineage/search, optional compaction and text embeddings, configurable
Chat Completions/ASR/embedding clients, accounting, baseline runners, reporting, and benchmark bridges.

The local suite runs synthetic vision and mocked HTTP contracts, including an actual FFmpeg
encode/extract timestamp test. PyAV media preparation, targeted Ruff and hosted API screening
have since been exercised; the original bundle's missing-dependency notes are historical.
See the validation receipt and public-safe research reports for current commands and results.

**Hosted image-packet screens are not evidence of reliable runtime tool following or commercial
savings.** Hosted embedding/ASR, GPU serving, VSS, real cameras, full official benchmark protocols
and customer workloads remain unqualified. The synthetic backend is a test double, not an ML result.

## Non-negotiable constraints

Do not pass answer keys, future event reference times, or ground-truth evidence hints into runtime
contexts. Do not weaken the snapshot boundary to raise scores. Do not report model/API wall time as
GPU time, missing prices as zero cost, timeouts as free, or deterministic replay as real-time capacity.
Do not pool several models into an “architecture gain” without a same-model comparison. Do not delete
hard cases or failed outputs from the denominator. No external actuation is permitted in this phase.

Obtain permission for paid calls and an explicit provider spending limit. Every executable network
path requires `--allow-network` or the corresponding adapter environment flag. Our budget is estimated
admission control, not a guaranteed provider invoice cap. A matrix runs four separate budgets.
Use credentials only through environment variables. Do not commit keys, customer video, transcripts,
annotations with restricted rights, or unredacted production traces.

## P0-A. Establish actual provider/model compatibility

Choose one modest open VLM and one stronger candidate, based on current official model cards and
provider capabilities. Supply exact model IDs and immutable revisions where available. Do not assume
that a name containing “Omni,” “Flash,” or a new version number guarantees native audio, images, video,
JSON behavior, or an economical deployment.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,server,video]'
pytest -q
streambudget demo --out runs/local-demo

export VLM_BASE_URL='https://YOUR-COMPATIBLE-PROVIDER/v1'
export VLM_MODEL='YOUR-ACTUAL-VISION-MODEL-ID'
export VLM_API_KEY='YOUR-KEY'
streambudget doctor --config configs/api.yaml --probe --allow-network --out runs/api-probe
```

The probe checks an actual tiny image and a JSON response. Inspect whether its content is grounded
in the image, not just whether HTTP returned 200. Then run the same generated input under the real API:

```bash
streambudget run --events runs/local-demo/input/events.jsonl \
  --tasks runs/local-demo/input/tasks.jsonl --config configs/api.yaml \
  --out runs/api-demo --allow-network
streambudget score --run runs/api-demo --labels runs/local-demo/input/labels.jsonl
streambudget report --run runs/api-demo
```

Test malformed JSON, refusals, truncation, rate limits, image counts, model token caps, cached-token
usage reporting, cancellation, and provider timeout behavior. Adapt `token_parameter`, `json_mode`,
`temperature`, and `extra_body` explicitly rather than retrying an unsupported schema repeatedly.
Set actual prices before evaluating dollars. Never infer zero prices from missing information.

**Deliverable:** `research/API_SMOKE.md` and sanitized `research/API_SMOKE.json` with date, provider,
model/revision, image/JSON support, example outcomes, configured prices, observed usage, and failures.
Do not include keys or private media. Acceptance: visually grounded outputs and correctly reconciled
usage on a small checked example; unsupported features must fail clearly.

## P0-B. Build a small, independently reviewed operational workload

Start with authorized fixed-camera footage and a concrete goal such as a packing-area dwell or
process-step exception. Include positive examples, long negatives, false triggers, outages, occlusion,
short events, and stationary deadlines. Use external detector/sensor outputs only when they were
available at the corresponding point in time. Map source timestamps and arrival timestamps separately.

```bash
streambudget prepare-video --video /path/to/authorized.mp4 --out data/operational \
  --source camera --fps 8 --max-seconds 600
# Create tasks.jsonl and independent labels.jsonl following docs/BENCHMARKS.md.
streambudget validate-data --events data/operational/events.jsonl --tasks data/operational/tasks.jsonl
```

Retain sufficiently dense source frames. A 1 FPS prepared manifest is unsuitable for claiming safe
high-frequency adaptive perception. The first-minute synthetic scene is a plumbing check, not this
workload. A strong annotator model may propose labels, but use independent human review and holdouts;
its guesses are not automatically ground truth. Keep labels inaccessible to the runtime process.

Split by camera/day/site, not neighboring frames. Record who owns footage and annotations and which
uses were authorized. Do not download StreamArena/Video-MME/MEVA or other data before checking rights.
The runtime itself does not need benchmark annotations for deployment.

**Deliverable:** a private rights-cleared dataset card, exact public-safe schema, frozen split hashes,
event definitions/latency requirements, and a disagreement audit. Acceptance: meaningful positive and
negative cases, clear temporal labels and source freshness, and an untouched held-out split.

## P0-C. Run matched baselines before inventing more architecture

```bash
python scripts/run_matrix.py --events data/operational/events.jsonl \
  --tasks data/operational/tasks.jsonl --labels data/operational/labels.jsonl \
  --config configs/api.yaml --out runs/api-matrix --allow-network
```

Use the same models and request caps across the initial four policies. Add a genuinely tuned detector/
tracker/rules-plus-VLM baseline using the customer's available stack. Tune fixed FPS/resolution,
periodic refresh, and monitor frequency on a calibration split. Do not compare only against dense
large-model inference. The current `motion` policy already includes a periodic refresh.

Run deterministic replay to debug errors, then `--timing realtime` at speed one for contention tests.
Real-time predecoded replay still excludes video decoding/network; charge preparation separately or
measure the full camera bridge. Model-latency summaries alone are not total camera-service latency.

Record per-class recall, false positives per camera-hour, missed/dropped/late responses, historical
answer quality and evidence faithfulness, all billed attempts and costs, and source-to-delivery latency.
For free-form answers, implement a validated rubric/judge and audit disagreements. The exact text
scorer currently included is too strict for general semantic QA and must be labeled accordingly.

Run several loads and budgets, not just a favorable point. Report paired deltas and confidence
intervals clustered by video/site. Dedicate a frozen holdout to the final report; do not hill-climb it.
Test costs for one retrospective question and for many questions sharing the same history.

**Deliverable:** `research/BASELINE_REPORT.md`, machine-readable points and errors, code/config/model
identifiers. Acceptance: full denominators, no hidden API costs, and measured cost-quality frontiers.
A proposed early target is ~2× capacity or 50% less total cost at an agreed quality/latency floor,
not an assumption that the target will be reached.

## P0-D. Validate the StreamArena bridge and untouched StreamMind

Clone a specific upstream commit separately. Keep its license and dataset terms. Install StreamBudget
in that environment, then use the driver interface documented in R02:

```bash
export STREAMBUDGET_CONFIG=/absolute/path/streambudget/configs/api.yaml
export STREAMBUDGET_OUTPUT=/absolute/path/runs/streamarena
export STREAMBUDGET_ALLOW_NETWORK=1
# Run from your separately checked-out StreamArena repository:
python method/streammind/run_streammind.py \
  --agent streambudget.adapters.streamarena_native:NativeStreamBudgetAgent \
  --dataset /path/to/authorized/StreamArena --language en \
  --video-dir /path/to/videos --videos YOUR_VIDEO_ID --out /path/to/results.jsonl
```

The upstream loader enforces inheritance. The optional `streamarena_native` wrapper supplies it
without vendoring upstream code. `scripts/check_streamarena_native.py` verifies dynamic loading
with generated footage and the mock backend; it requires PyAV and OpenCV in the local environment.
Check frame/ASR arrival times, start/stop isolation, grace periods, qtype handling, answer callbacks,
and proactive delivery time. **Keep `ref_ts` and its ground-truth-derived deadline outside agent state.**
Our adapter intentionally ignores them. Actual user-specified deadlines would be different inputs.

The bridge does not implement external web/image search. StreamArena Tool tasks are therefore not
fully covered; exclude with explicit scope or add a budgeted safe tool before claiming a full score.
Native audio is not passed through; ASR observations are supported. The adapter anchors its clock
at the first legitimate frame/ask/audio callback, excluding decoder warm-up and future references.

Run upstream StreamMind unchanged with the same backbone where practical. Separate paper v1, current
core, and documented v2. Inspect whether the v2 OCR/retrieval code and matching configurations were
actually released. Do not infer parity from a README or a trajectory bundle.

**Deliverable:** `research/UPSTREAM_REPRO.md` with upstream commit, exact commands, supported categories,
callback tests, loader changes, scoring protocol, and explicit incomplete categories.

## P1-A. Self-host and measure the serving layer

Use a separate GPU environment with pinned engine, driver, torch, model revision, precision,
tensor-parallel configuration, image processing limits, and output budgets. The shell templates need
adjustment for your hardware and the engine version:

```bash
export MODEL_ID=/path/to/model
# Inspect each script's environment contract before execution.
bash scripts/serve_vllm.sh
# Or, in an alternative environment:
bash scripts/serve_sglang.sh
```

Point `configs/local.yaml` at the server and perform the same compatibility probe. Profile encoding,
prefill, generation, transfer, batching, CPU decode, memory, queues, retries, and power if available.
Collect actual device measurements with timestamps; do not replace the null GPU field with API
latency. Whole-fleet idle/allocated costs matter even when kernels are not executing.

Add a **separate native-video backend adapter** before claiming comparison with EVS/VidCom2 or another
video-specific engine optimization. The current image-message path does not enable those methods by
default. Stable media identities and exact cache keys need explicit correctness tests; similar pixels
are not sufficient for unrestricted KV reuse. Measure cache warm and cold behavior separately.

**Deliverable:** `research/GPU_PROFILE.md` plus traces and reconciled cost/capacity points. Acceptance:
reproducible matched comparisons, server versions pinned, no unaccounted encoder or background work.

## P1-B. Integrate one real video backend, not all of VSS

Choose the actual customer/NVIDIA VSS version. Verify its video storage/snapshot/clip APIs and event
schemas. `adapters/signals.py` is currently a generic JSON normalizer, **not** a full VSS/Kafka/protobuf
client. Implement authentication, time normalization, retries/idempotency, reconnect, and evidence
availability on top of a pinned service contract. Include external detector and embedding cost.

Test `bridge-video` after installing PyAV. It is a trusted-input research bridge without reconnection,
RTSP packet telemetry, native audio synchronization, or a durable original-video store. Preserve dense
recordings so active inspection can re-read an interval at higher frame rate than the cheap gate used.
Add causal clip reacquisition and high-FPS fixtures; don't “upsample” sparse stored frames.

Use specialized OCR/segmentation only when a task needs it. External OCR observations are supported
now; the `ocr` tool itself is VLM based. Add a dedicated engine as an optional plugin, verify its
license/model weights, and charge its work. Speech transcription is not equivalent to recognizing
sirens or machine sounds; add an actual audio-event tool for those use cases.

**Deliverable:** `research/BACKEND_INTEGRATION.md`, schema fixtures, authorized sample replay,
reconnection/error tests, and full transport/decode accounting.

## P1-C. Expand benchmark coverage deliberately

Video-MME nested JSON import and output export exist. Test them on the released data format in your
environment and run the official scorer. Current subtitle-free mode is intentional. If adding
subtitles, include only subtitles aligned to sampled frames under the official protocol [R12].

NIST ActEV/MEVA official scoring, FPS-Bench, ProactiveBench, OmniPro, OVO-S and drone benchmarks are
**not implemented in this repository**. Re-verify owners, schemas, licensing and current tasks before
adding adapters. Choose one suitable high-FPS/proactive stress test and one customer-domain benchmark
rather than implementing ten shallow loaders. A recognized offline score can show preserved video
ability; it does not prove camera-market demand or live performance.

Maintain a manifest with per-adapter status: schema inspected, fixture-tested, dataset executed,
official scorer matched, actual published results. Never collapse these into a single “supported” flag.

## P2. Train only after the untrained baseline is understood

Consider distilling perception/routing, learning observation actions, or task-specific compression
only after measured bottlenecks justify it. Keep perception, memory and policy changes separately
ablatable. Use authorized training data, a separate calibration split, and untouched evaluation.

Logs from the chosen gate are selection-biased: skipped intervals do not label themselves. Acquire
counterfactual evidence through randomized audits or shadow full-compute checks. Optimize budgeted
quality while enforcing minimum recall and latency rather than rewarding skipping alone. Do not
train on evaluation answer keys or reuse inaccessible noncommercial trajectories for a product.

**Deliverable:** a preregistered hypothesis, dataset/rights manifest, untrained baseline, matched
SFT/RL ablations, and a cost-quality gain that survives held-out cameras/workloads.

## Production gaps requiring engineering, not claims

Runtime restart/resume and durable budgets/watch state; persistent alert idempotency; retention and
privacy deletion; tenant isolation; auth/rate limiting/audit; hardened payload handling; CPU offload;
bounded memory under very long traces; distributed state and vector indexing; priority-aware admission
under a full queue; backend cancellation semantics; live reconnect; reliable multistream time sync;
full media retention; shadow evaluation; model drift/calibration; fleet-level billing and telemetry.
Existing work directories intentionally fail rather than pretending these are solved.

Customer discovery, purchasing authority, legal clearance, dataset rights, and acceptable missed-event
rates require human decisions. Technical success alone does not establish a buyer or a budget.

## Close the loop

For every continuation, update `FEATURE_STATUS.md` and the validation receipt with commands and
artifacts. Add a regression test for every boundary bug. Put public-safe results under `research/`;
keep credentials/private data elsewhere. An honest “no gain over a tuned baseline” result is valuable.
The intended outcome is a reproducible answer to whether this should become an inference business,
not a larger architectural diagram or an unqualified benchmark leaderboard entry.
