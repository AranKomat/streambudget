# Live API and normalized observations

This is a single-tenant research API. Bind to loopback by default. Use TLS/reverse-proxy limits for
any remote access, a secret bearer token, authorized input, and provider-side billing limits. See
[SECURITY.md](../SECURITY.md). No endpoint controls a physical device.

```bash
export STREAMBUDGET_TOKEN='YOUR-LONG-RANDOM-SECRET'
streambudget serve --config configs/api.yaml --out runs/live --allow-network
```

The output directory must be fresh. The server cannot resume budget/watch state after restart.
For a no-model test, use `configs/mock.yaml` and omit `--allow-network`. Development-only
`--insecure-local` is accepted only on a loopback bind. The ASGI lifespan owns the runtime.

## Authentication and errors

All `/v1/*` endpoints require `Authorization: Bearer <STREAMBUDGET_TOKEN>` unless explicitly started
in insecure-local mode. `/healthz` is unprotected. Typical failures: 401 authentication, 422 invalid
contract, 429 inference budget, 503 full/closed queue, 502 model response/transport, 504 deadline.
Detailed provider response bodies and keys are not echoed. A 502/504 does not prove a request was free.

## Session time

`GET /v1/status` includes `session_time` in seconds since session startup, model-attempt counters,
queue/evidence status and current ledger. Capture timestamps must share this time origin. The server
assigns receipt availability; it refuses observations more than 0.5 seconds ahead of its clock.
Clock synchronization and network lag must be measured for remote cameras. Use a timestamp origin
mapping for wall-clock sources; do not submit Unix epoch seconds directly.

```bash
curl -fsS -H "Authorization: Bearer $STREAMBUDGET_TOKEN" \
  http://127.0.0.1:8765/v1/status
```

## Frame ingestion

`POST /v1/frames`

```json
{"source":"camera","timestamp":12.25,"jpeg_base64":"BASE64_IMAGE_BYTES"}
```

Response: evidence ID, `accepted: true`, and queue depth. The request stores the received image and
may schedule a visual observation. Acceptance is not confirmation of a completed model check.
The legacy field name says JPEG; image decoding also accepts supported image formats, preserving
received bytes. Inference views and `GET /v1/frame/{id}` return resized JPEGs. Maximum image body is
12 MB; pixel count is capped. Do not upload untrusted arbitrary URLs.

## Structured observations

`POST /v1/signals`

```json
{"source":"camera","timestamp":12.25,"kind":"sensor","text":"Machine temperature sample","data":{"temperature":87.4}}
```

Allowed kinds: `sensor`, `detector`, `ocr`, `asr`, `audio_event`. `start` is optional for a transcript
or event interval. An ASR observation spanning 10–15 seconds has `start:10,timestamp:15`; its actual
arrival is recorded by the server. All fields are predictions/observations, not answer keys.

Example acoustic event:

```json
{"source":"camera","timestamp":16.5,"start":16.0,"kind":"audio_event","text":"External classifier detected an impact","data":{"event":"metal_impact","confidence":0.82}}
```

The source groups evidence for a camera/task. Cross-camera fusion and calibration are not automatic.
`JSONSignalMapper` converts explicit dotted fields from an already-decoded customer/VSS message.
It does not connect to Kafka or infer its schema. See `examples/signal-mapping.json`.

## Register or cancel a watch

`POST /v1/watches`

```json
{
  "id":"packing_area_dwell",
  "source":"camera",
  "goal":"A package remains inside the marked packing area",
  "dwell_seconds":10,
  "max_observation_gap":4,
  "min_confidence":0.8,
  "repeat":false,
  "cooldown_seconds":30
}
```

A newly registered watch cannot claim to have monitored the past. Repeated IDs are rejected; use a
new ID for a revised objective. Confidence is an uncalibrated model output unless you calibrate it.
A missing observation is not a negative observation. Dwell requires a sufficiently fresh sequence
of positive observations; a single old frame plus elapsed wall time cannot produce a dwell alert.

A scalar predicate needs no model:

```json
{
  "id":"temperature_high",
  "source":"camera",
  "goal":"Temperature exceeds the configured operator threshold",
  "sensor_predicate":{"key":"temperature","op":"gt","value":85,"kind":"sensor"},
  "repeat":false
}
```

This is an example threshold, not a safety recommendation. Operators define real semantics.
`DELETE /v1/watches/{id}` cancels a watch. `GET /v1/events` returns the most recent 200 emitted alerts,
including evidence and delivered times; it is polling, not a durable external message stream.

## Investigate

`POST /v1/query`

```json
{"source":"camera","question_id":"investigation-01","question":"What occupied the packing area earlier? Inspect the evidence."}
```

Optional `as_of` must not exceed observed session time. The query freezes its source-evidence
snapshot at admission and may materialize new derivations only from that permitted history.
It returns `text`, `evidence_ids`, `as_of`, `status`, `elapsed_s`, and tool steps. Inspect `status`:
`unverified`, `step_limit`, and `no_evidence` are not normal successful grounded answers.
Elapsed time includes queue wait. The HTTP call waits for this answer, but other ingestion can proceed.
CPU-heavy preprocessing is still synchronous and must be profiled before claiming large-scale capacity.

`GET /v1/evidence/{id}` returns evidence metadata/lineage; `GET /v1/frame/{id}` returns a bounded JPEG
view of frame evidence. Possession of a valid ID does not certify semantic answer correctness.

## File schema versus REST schema

Replay JSONL uses `ts`, `media` (relative path), and optional `available_at`. REST uses `timestamp`
and inline image bytes. `media` may only resolve beneath the manifest directory. Each JSONL line is
one object. Task JSONL has `type`, `at`, `id`, `source`, and question or watch content. Labels are a
separate file consumed only by the scorer, not by any runtime/API route.

## Extensions

A production adapter should add connection recovery, idempotent input/alert IDs, timestamp sync,
source authentication, dense original clip retention, backpressure and delivery acknowledgments.
Do not bypass time/provenance checks when adding a remote clip reader. No audio/video WebSocket or
native Omni transport is implemented; audio can currently enter as ASR/audio-event observations.
