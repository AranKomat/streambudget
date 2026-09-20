# References and source interpretation

Access/check date: **2026-09-20**. Documentation and default branches can change. Pin versions before
reproduction. These references support the design and interface descriptions, not a claim that our
code matches an upstream implementation or its reported performance. No paper's headline compression
ratio is treated as our measured end-to-end speedup.

| ID | Primary source | What it supports / use here |
|---|---|---|
| R01 | [StreamArena repository](https://github.com/JIA-Lab-research/StreamArena) | Benchmark/toolkit and StreamMind context; external comparison target, no source vendored |
| R02 | [StreamingAgent contract](https://github.com/JIA-Lab-research/StreamArena/blob/main/method/streammind/agent.py) and [driver README](https://github.com/JIA-Lab-research/StreamArena/blob/main/method/streammind/README.md) | Actual lifecycle/callback signatures inspected for our independent adapter; future `ref_ts` is evaluator knowledge |
| R03 | [StreamArena data schema](https://github.com/JIA-Lab-research/StreamArena/blob/main/streamarena/data.py), [paper](https://arxiv.org/abs/2608.05703) | Separates question times, answers, types and evidence hints; motivates long-horizon monitoring/evaluation |
| R04 | [OmniAgent](https://github.com/HarryHsing/OmniAgent), [paper](https://arxiv.org/abs/2606.19341) | Active observation and training inspiration; our code does not reproduce its SFT/RL |
| R05 | [VideoSeek](https://github.com/jylins/videoseek), [paper](https://arxiv.org/abs/2603.20185) | Coarse-to-fine investigation inspiration; no source imported and no results copied as ours |
| R06 | [NVIDIA VSS introduction](https://docs.nvidia.com/vss/latest/index.html), [repository](https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization) | Independent video services, analytics and agent workflows; potential backend, not deployed in this build |
| R07 | [vLLM multimodal inputs](https://docs.vllm.ai/en/latest/features/multimodal_inputs/), [vLLM-Omni streaming video](https://docs.vllm.ai/projects/vllm-omni/en/latest/serving/video_stream_api/) | Multimodal serving interfaces and model-specific paths; generic image requests do not imply native-video optimizations |
| R08 | [SGLang OpenAI-compatible APIs](https://docs.sglang.io/docs/basic_usage/openai_api) | Compatible serving boundary; actual chosen model/server version needs validation |
| R09 | [OpenAI images/vision guide](https://developers.openai.com/api/docs/guides/images-vision) | Timestamped text plus image-message transport design; does not guarantee every compatible provider behaves identically |
| R10 | [Audio transcription API](https://developers.openai.com/api/reference/resources/audio/subresources/transcriptions/methods/create) | Multipart audio request shape used by optional WAV helper |
| R11 | [Embeddings API](https://developers.openai.com/api/reference/resources/embeddings/methods/create) | Optional text-vector request/response shape; local fake HTTP tests are not endpoint validation |
| R12 | [Video-MME evaluation README](https://github.com/MME-Benchmarks/Video-MME), [output template](https://github.com/MME-Benchmarks/Video-MME/blob/main/evaluation/output_test_template.json) | Nested JSON import/export and sampled-frame subtitle constraint. Current converter is subtitles-free |
| R13 | [NIST ActEV](https://actev.nist.gov/) | Official activity-detection evaluation reference. Our event-window matcher is explicitly not its scorer |
| R14 | [Creative Commons BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) | Noncommercial license terms; code, models, annotation files and footage must be reviewed separately |
| R15 | [SimpleStream](https://github.com/EvolvingLMMs-Lab/SimpleStream) | Motivation to retain a recent-window baseline rather than assuming long memory always helps |

## File identities actually inspected for interface work

These are **Git blob hashes for individual files**, not repository commits. They are useful for
identifying the exact interface source encountered, but are not a complete upstream lockfile.

| Repository/path | Inspected blob SHA |
|---|---|
| `JIA-Lab-research/StreamArena:method/streammind/agent.py` | `3f456974707332268055cdb09b850cfdb89f616b` |
| `JIA-Lab-research/StreamArena:method/streammind/README.md` | `ae24163bf8088fb4ab40fb718545b730ed936f82` |
| `JIA-Lab-research/StreamArena:streamarena/data.py` | `00fa9029671668b05a462848765d93aa69e8a818` |
| `MME-Benchmarks/Video-MME:README.md` | `69ade7d2dee0af20470c075de4e36f07ee126160` |
| `MME-Benchmarks/Video-MME:evaluation/output_test_template.json` | `edd71473c57c02170477f16a20842676765b2147` |

## Deliberate qualifications

StreamMind's v2 README/trajectories are not proof that all described OCR and embedding components
are in the same checked-in code path. Verify that before claiming an upstream v2 reproduction.
StreamMind's lower post-query latency is not automatically lower continuous GPU expense. A serving
engine's support for a model family does not prove support for that model's video, audio, JSON-output,
cache or quantization path. Repository activity and stars are not quality measurements.

The proposed future benchmark portfolio includes high-frame-rate, proactive, audio-rich and
industrial activity evaluation. Those are research directions in the handoff, not bundled official
benchmark implementations. Prefer the official owner and verify current access/license before use.
