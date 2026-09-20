# Operations, accounting, and model configuration

## Local execution and optional dependencies

Python 3.11+; local validation used 3.13. Base runtime: HTTPX, Pydantic, PyYAML, NumPy, Pillow, SQLite
with FTS5. `.[server]` adds FastAPI/Uvicorn; `.[video]` adds PyAV. FFmpeg can prepare local video when
PyAV is unavailable. `.[dev]` adds test/lint tools. The repository does not install torch, model
weights, vLLM, SGLang, NVIDIA containers, databases, or paid SDKs automatically.

Use a separate environment for each self-hosted engine/model combination. Pin dependencies after
real provider/GPU validation. `results/environment.json` is the local receipt, not a universal lock.

## Role configuration

`models.perception` is required. `planner`, `verifier`, `memory`, `embedding`, and `asr` can be added.
Missing generic planner/memory roles fall back to perception; optional verifier and specialist roles
must be configured to activate their features. `supports_images: false` is appropriate for a text-only
planner, not a perception worker.

API keys are looked up from `api_key_env`; the config contains the variable name, not its value.
Use `${VLM_BASE_URL}` and `${VLM_MODEL}` placeholders for non-secret configuration. Unresolved `${...}`
placeholders fail validation. Avoid embedding credentials in model names, body overrides, URLs or logs.
Remote HTTP needs explicit unsafe configuration; prefer HTTPS. API redirects are not followed.

Relevant knobs: `json_mode`, `token_parameter`, `max_output_tokens`, `timeout_s`, `max_retries`,
`retry_timeouts`, `temperature`, and permitted `extra_body`. Never let `extra_body` override the
message, model, stream or output-cap fields. Models that require a different wire format need a
new adapter, not an unchecked override.

`semantic_embeddings: true` enables the configured standard text embedding endpoint. It is metered.
`hash_embeddings` is a deterministic **lexical hash**, not a semantic vision model. Model fingerprints
prevent mixing incompatible vector spaces. Current retrieval is simple SQLite/in-memory scoring,
not a high-throughput distributed ANN service. Embedding writes run at lower priority than alerts.

`compact_every` optionally summarizes groups of captions using the memory role. It adds model cost
and keeps source evidence. Default zero leaves this disabled. Make compaction an ablation rather
than assuming an extra memory-writer model is free or improves accuracy.

## Budget semantics

Each model or specialist attempt consumes an attempt slot. Retries consume new slots. Exact concurrent
identical calls may share one attempt. Sequential calls, changed timestamps, models, or context do not
share semantic answers. No kernel/encoder reuse is claimed.

Dollar admission uses a quote: approximate prompt token count plus image allowance plus maximum
output, at explicitly configured rates. Actual usage is reconciled when available. Cached-token
prices apply only to provider-reported cached tokens. Token reservations can underestimate; an
actual provider charge is not capped by this local estimate. Set provider-side spending controls.

| Report field | Meaning |
|---|---|
| `reported_usd` | Sum calculable from valid returned token usage/configured rates, or measured WAV duration/configured ASR rate |
| `complete_reported_usd` | Null if any attempt is unpriced, unknown, or still reserved; do not treat partial dollars as a complete total |
| `provisional_usd` | Reserved estimates retained for attempts whose billed usage is uncertain |
| `reserved_usd` | Work admitted but not yet settled |
| `unpriced_attempts` | Attempts without enough pricing information |
| `unknown_usage_attempts` | Attempts without usable returned usage/billing basis |
| `gpu_seconds` | Always null in this version; no GPU measurement implemented |

An HTTP 429, transport timeout or client cancellation is conservatively marked uncertain unless a
trusted billing source establishes otherwise. The client does not prove remote cancellation. Bad
provider counts are treated as unknown, not zero. A late response can exceed a previous quote; it is
reported, not hidden. For latency compare wall time; for cost reconcile invoices/usage; for GPU
compute use actual device/engine measurements. They are different quantities.

Our process does not automatically measure a customer's external detectors, embedding servers,
camera hardware, storage, transport or reserved idle GPU expense. Standalone `transcribe` and video
preparation have separate reports that must be combined with the runtime's costs. A prerecorded ASR
manifest must not be counted as free preprocessing in an end-to-end experiment.

## Queue semantics and load

Workers take highest-priority pending jobs; pending identical job keys coalesce. Alert observations
are prioritized over query, background, embedding and compaction jobs. Execution and queue wait are
bounded by deadlines. A full queue rejects new jobs even if they have high priority. There is no
preemption, per-tenant fairness, autoscaling, or backend cancellation guarantee.

Trace counters expose enqueued/coalesced/dropped/failed/completed work. Report queue ages and dropped
queries alongside latency. A fast median after discarding slow work is not a capacity result. Test
bursty simultaneous cameras, queries during active alerts, and slow providers. Isolate deterministic
correctness from paced capacity tests.

## Input and session lifecycle

File preparation stores selected frames using presentation timestamps. FFmpeg preparation reports
wall and child CPU time; PyAV reports its own process CPU work. Neither is compressed-domain pruning.
Keep original recordings where permitted. Multi-hour extraction can be large; bound duration for
initial tests and provision disk space. Source replay currently loads event metadata into memory.

A runtime workdir is one session. Reusing an existing database is rejected because budgets, monitor
state, and alert delivery are not transactionally resumable. Reading an old `EvidenceStore` for
analysis is different from resuming a production runtime. Do not delete the guard until implementing
and testing recovery semantics.

The ASGI server creates SQLite on its serving event loop. Run one worker process per configured
workdir; do not start several Uvicorn workers writing the same runtime. Auth is a single bearer token.
TLS, request-rate controls, source identity, private storage and retention are operator work.

## Outputs

`run.json`: result/config summary, predictions, alerts, costs and timing flags.
`predictions.jsonl`: machine-readable per-question outputs.
`trace.jsonl`: stage and queue events, normalized model usage, errors and drops; no prompt/image bodies.
`memory.sqlite` and `media/`: sensitive evidence and lineage.
`config.resolved.json`: selected model/runtime settings; never place secrets in arbitrary config fields.
`scores.json`: local scorer results, explicitly not an official benchmark result.
`report.html`: escaped local visual report; synthetic status remains visible.

## Publication checklist

Record code commit, environment, dataset rights/split hash, model revision and provider/engine,
precision, input FPS/resolution/audio path, output caps, policy parameters, clock mode, cache warm/cold
state, all preprocessing and external services, budget failures, and confidence intervals. Describe
what input was unavailable. Include the strongest practical baseline and an untouched holdout.
Do not imply the native-video optimization path was active from the serving engine's name alone.
