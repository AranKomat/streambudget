# Evaluation and benchmark integration

The code separates **runtime observations**, **tasks**, and **evaluation labels**. Only the first two
enter `streambudget run`. Official benchmark adapters must preserve that separation. A registered
user objective is legitimate; a ground-truth target time is not an observation.

## Normalized schemas

`events.jsonl`:

```json
{"source":"camera","ts":0,"kind":"frame","media":"frames/00000000.jpg"}
{"source":"camera","ts":4,"start":3,"available_at":4.3,"kind":"asr","text":"Package ready","data":{}}
{"source":"camera","ts":5,"kind":"detector","text":"External package detection","data":{"count":1}}
```

Paths are local to the manifest directory. Source time uses seconds, not frame index or epoch time.
`available_at` is when the observation reached the system. A generated caption/ASR signal can have
later availability than its source interval. Omitting availability means it was available at `ts`;
that assumption must be justified for imported prerecorded predictions.

`tasks.jsonl`:

```json
{"type":"watch","at":0,"id":"package_dwell","source":"camera","watch":{"goal":"Package in the work area","dwell_seconds":3,"max_observation_gap":2}}
{"type":"ask","at":20,"id":"q1","source":"camera","question":"What was in the work area earlier?"}
{"type":"cancel","at":25,"id":"package_dwell","source":"camera"}
```

`labels.jsonl`, held by the evaluator only:

```json
{"type":"event","source":"camera","watch_id":"package_dwell","start":13,"end":18}
{"type":"qa","id":"q1","answer":"A package was in the work area."}
```

For dwell, `start` should be the earliest point at which the condition was sufficiently established
under your task definition, not necessarily the initial entry timestamp. State this explicitly.
For speech ASR or OCR prepared offline, add the real/estimated processing availability and cost.
Do not turn a model's retrospective description into an instant ground-truth runtime cue.

## Local metrics—not official benchmark claims

The event scorer matches alerts to truth intervals of the same source/watch ID one-to-one using
maximum-cardinality matching. An alert is matched by its observation timestamp. Duplicates or
out-of-window outputs are false positives. It reports recall, precision, false positives per
camera-hour and matched evidence/delivery delays. A correctly grounded but extremely late alert
can still match by observation time; impose a separate application delivery deadline/quality gate.
This local metric is **not NIST ActEV's scoring protocol** [R13 in REFERENCES.md].

QA supports strict A–D parsing or exact normalized text. Missing/error/dropped outputs are incorrect.
This is not a semantic judge for free-form answers. For deployment use independent human review or
a validated separate judge, retaining failures and measuring evidence faithfulness.

Camera-hours are summed per-source observation spans. One isolated frame contributes no duration.
A production recording system should supply actual coverage/outage intervals. The current manifest
span alone does not distinguish an inactive camera from a long missing-data gap.

## Baseline matrix

`python scripts/run_matrix.py` runs four original controlled policies with the same model config,
isolated memories, budgets, and result directories. The variants are not renamed upstream methods:

| Variant | Purpose |
|---|---|
| Fixed-rate | Reference background processing at a configured interval |
| Motion/cue plus periodic | Stronger cheap-gating baseline, not a deliberately broken motion-only system |
| Adaptive | Adds pending-dwell verification and randomized audits |
| Recent-only | Present perception without an always-built semantic history |

Use the same model, precision, input frames/signals, and output limits for policy ablations. A
role/model change is another axis. Include a tuned actual detector/rules pipeline where appropriate.
Sensor predicates use legitimate source data in all policies. Count their source production cost.

Deterministic replay is causal but pauses input while work drains; it cannot measure throughput.
Real-time speed-one replay does not intentionally pause video for model jobs, but still uses a
predecoded manifest. Report excluded preparation/transport and event-loop CPU overhead. Multi-source
bursts and queue drops matter as much as isolated query speed.

## StreamArena / StreamMind

Delivered: an independent duck-typed `streambudget.adapters.streamarena:StreamBudgetAgent` matching
the inspected `start`, `stop`, frame, ASR, question, and proactive-watch callbacks. Local tests confirm
that future `ref_ts` is not included in watch state. Source file identities are in REFERENCES.md.

Not executed: actual upstream loader/driver, full dataset, official judge, or a model-backed
StreamMind comparison. Our adapter does not implement external web/image search for `Tool` tasks.
It accepts transcribed audio, not raw native Omni input. The dynamic loader may require an upstream
base-class wrapper; validate it rather than assuming contract similarity proves compatibility.

Use authorized data and a pinned upstream checkout. The handoff has the concrete invocation. Keep
upstream question answer/evidence fields away from our runtime. Record event-delivery clock offset,
grace period, reference masking, and supported task categories. A v2 README result is not proof that
the exact OCR/retrieval path is released or reproduced.

## Video-MME

Delivered: conversion from strict nested official-shape JSON plus local videos, and output export.
Dataset acquisition is manual and rights-dependent. The converter requires explicit acknowledgment;
it does not grant permission. The videos list should include `video_id` and `questions`; each question
contains `question_id`, `question`, `options`, `answer`, and optional task metadata. Local filenames
are `videoID.mp4` or `video_id.mp4`, or specify a safe relative `video_file` field.

```bash
streambudget import-videomme --annotations /path/to/nested.json --video-dir /path/to/videos \
  --out data/videomme --fps 2 --limit 3 --acknowledge-data-license
streambudget run --events data/videomme/001/events.jsonl --tasks data/videomme/001/tasks.jsonl \
  --config configs/api.yaml --out runs/videomme-001 --timing archive --allow-network
streambudget export-videomme --template /path/to/nested.json \
  --predictions runs/videomme-001/predictions.jsonl --out runs/videomme-001/official-shape.json
```

Each converted case has separate runtime tasks and labels. Question options are legitimate inputs;
answers are never copied into tasks. Current mode is **subtitles-free**. Adding all subtitles to a
sparse-frame run violates the sampled-frame alignment guidance in the official README. A strong
subtitle index is a distinct protocol, not an interchangeable official comparison [R12].

A single-run export fills unpredicted questions with empty responses. Aggregate predictions from
all intended cases before a whole-set export. Use the official scorer and disclose active inspection,
model/tool behavior, input budget, and timing for publication. No real dataset run was performed here.

## Operational domain and future benchmarks

Start with “watch continuously, then investigate the exception”: authorized industrial video with
independent labels, a standing objective, evidence-linked alerts, and later historical queries.
Include stationary deadlines, missing observations, short transients, occlusion, confusing activity,
and simultaneous cameras. A moving-camera holdout tests whether pixel change is the wrong abstraction.

NIST ActEV/MEVA can inform domain evaluation but needs its own official adapter/scorer. High-FPS,
proactive silence/duplicate, audio-rich, and drone/spatial benchmarks are sensible extensions after
verifying their owners, data formats and rights. They are **not implemented or run** merely because
they were discussed during planning. StreamArena's novelty or repository stars do not translate into
customer value by themselves.

## Result record required before any performance claim

Save method and code commit; exact model/engine/provider; dataset and split hash; rights note; source
FPS/resolution/audio and availability; native-video versus image path; all preprocessing costs; wall,
queue and actual device timing; cached/cold configuration; all errors/drops; per-class outcomes;
confidence intervals; and baselines selected without looking at the final holdout.

Report a cost-quality frontier and a delivered-latency floor. Treat ~2× capacity as a proposed target,
not an expectation. Do not advertise the synthetic fixture's perfect labels or reduced mock calls as
ML improvement. API-based experiments can establish API dollar savings; self-hosted GPU and fleet
cost claims require another measurement layer.
