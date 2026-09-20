# Implementation and validation status

v0.1.0. **Implemented** means source exists and is exercised by local tests where indicated; it does
not mean deployed, reproduced on a published dataset, or production hardened. Exact counts and
commands live in [VALIDATION.md](../results/VALIDATION.md).

| Area | Delivered implementation | Local validation | Not established / remaining |
|---|---|---|---|
| Archive / causal replay | Original runner, explicit source availability, independent labels | Synthetic full flow and boundary tests | Real operational quality |
| Paced/live execution | Nonblocking inference jobs, source clock, REST ingest | Short paced fixture, ASGI tests | Multi-camera capacity, source decode/network accounting |
| Durable evidence | SQLite/WAL, received image bytes, source/derived lineage | Future/late/transitive contamination tests | Original dense video recorder, retention, recovery |
| Search | Visible-corpus keyword ranking, optional text vectors, RRF | Keyword/vector/rank-isolation tests | Large-scale ANN, visual embeddings, relevance evaluation |
| Memory compaction | Optional metered model-derived summaries; original evidence retained | Local lineage/control tests | Quality at hour/day horizons |
| Monitoring | Typed goals, freshness-aware sampled dwell, repeat/cooldown, scalar predicates | Dwell/old response/shared observation/sensor tests | Calibrated recall and production event semantics |
| Observation scheduler | Fixed, motion/cue+periodic, adaptive/dwell/audit, recent-only | Synthetic ablations and queue tests | Learned policy; proof of semantic skip safety |
| Agent | Bounded JSON tool loop; search, inspect, VLM OCR, state, sensors | Tool bounds, evidence IDs, full mock loop | Real model tool-following and answer faithfulness |
| Chat backend | Configurable compatible HTTP, images/timestamps, roles, retries | HTTPX contract tests | Any real provider or self-hosted model run |
| Specialist APIs | Text embeddings and chunked WAV transcription | Mocked HTTP contracts and WAV duration | Real embedding/ASR endpoint; native AV/omni |
| Detector/OCR/sensors | External structured observations + generic mapping | Schema/time mapping tests | Dedicated detector/OCR/segmentation engine deployments |
| Live video bridge | Trusted local video / RTSP to REST using PyAV | Code present; optional PyAV test skipped locally | Actual PyAV/RTSP execution, reconnect, audio sync |
| Video preparation | Presentation timestamps via PyAV or FFmpeg | Actual FFmpeg generated-video test | PyAV path in this environment; long VFR footage sweep |
| Cost controls | Attempt cap, estimated reservation, usage reconciliation, unknown billing | Retries/timeouts/malformed counts/prices tests | Exact invoice cap, external CV/cloud costs, GPU telemetry |
| vLLM/SGLang | Configured HTTP boundary and launch templates | Generic request-shape tests only | GPU deployment, native video, KV/encoder reuse/pruning |
| VSS integration | JSON event mapper and service-boundary design | Synthetic message mapping | Actual pinned VSS/Kafka/protobuf/storage connector |
| StreamArena | Independent lifecycle bridge and GT-ref exclusion | Contract-level tests | Full upstream loader/driver/dataset/scorer run; external Tool search |
| Video-MME | Local nested JSON converter and official-shape exporter | Fixture tests | Real dataset and official scorer; subtitles intentionally off |
| Domain scoring | One-to-one point-in-window event scorer, QA/MCQ local scoring | Matching/duplicate/missing/Pareto tests | Official ActEV protocol or validated semantic judge |
| Other benchmarks | Integration plan only | None | FPS/Proactive/OmniPro/drone/etc loaders and results |
| Training | None, intentionally | Not applicable | SFT/RL/distillation only after meaningful data |
| Build/deployment | Packaging, CLI, Docker/CI definitions | See validation receipt | Docker build and remote CI not executed |
| Security | Single-tenant bearer auth, typed read-only tools, path/time guards | Basic route/contract tests | Hardened production security review; no actuation |

There are no redistributed datasets, weights, benchmark answers, or upstream source trees.
The synthetic mock's successes and call counts are test outcomes—not measured ML quality or savings.
