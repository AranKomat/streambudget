# Local validation receipt

## Real-footage follow-up (2026-09-20)

- **107 tests passed**, two dependency deprecation warnings; targeted Ruff and
  `git diff --check` passed.
- Added tests for real-packet cutoff/label separation/path confinement, nine-way
  concurrent dispatch, returned provider/tier recording, and rejection of an
  unconfirmed Gemini Flex route without dropping its cost.
- Isolated no-key demo: `runs/real-screen-no-key-demo`, 25 mock attempts, $0,
  2/2 toy answers and three toy events; plumbing only.
- Five openly licensed real clips prepared with PyAV; 20 questions/model plus
  three probes completed: **63 calls, $0.11700802** provider-reported cost, no
  failures or unresolved reservations. All 21 Gemini responses confirmed AI Studio Flex.
- [Report and limitations](../research/REAL_SCREEN_20260920.md), including the
  narrow exploratory scope and the remaining citation-equivalence diagnostic issue.

## Hosted synthetic screening (2026-09-20)

Real OpenRouter requests are now exercised against generated visual sequences; see
[results and limitations](../research/MODEL_SCREEN_20260920.md). The historical
"not executed" statements below describe the original bundle, not this later run.

Local validation on macOS / Python 3.13.7:

- `.venv/bin/python -m pytest -q`: **104 passed**, two dependency deprecation warnings.
- Targeted Ruff check on screening, backend, budget, runner and new tests: passed.
- `streambudget demo --out runs/screening-cost-no-key-demo`: 25 mock attempts,
  2/2 toy QA answers, 3 toy events, no false positives, $0; plumbing only.
- `git diff --check`: passed.

The new tests cover frozen balanced packets, evaluator-label separation, citation
bounds, unknown-charge carryover, failed-response denominators, no accidental
redispatch, identical model packets, config changes, provider-reported cost
admission, invalid cost handling, and provider-specific cost-field interpretation.
The pre-existing repository-wide Ruff findings were not changed by this work.

## OpenRouter connection check (2026-09-20)

On macOS with Python 3.13.7, `bash scripts/openrouter.sh` successfully loaded the
shared credential and validated `configs/openrouter.yaml` without inference.
An authenticated GET to OpenRouter's `/api/v1/key` returned HTTP 200. The public
model catalog listed `qwen/qwen3-vl-8b-instruct` with image inputs and response-format
support; its listed token rates were used in the configuration. No paid model
request was made, so image grounding and provider inference remain unvalidated.
`bash -n scripts/openrouter.sh` and `git diff --check` passed.

## Original bundle validation

Build date: **2026-09-20**. This is an execution receipt, not a model-performance report.
Environment: Linux x86-64, Python **3.13.5**. Package versions and unavailable optional packages are
recorded in [environment.json](environment.json).

## Executed successfully

| Check | Result | Evidence |
|---|---|---|
| Full CPU/HTTP-contract test suite | **89 passed, 1 skipped** | [pytest.txt](pytest.txt), [pytest.xml](pytest.xml) |
| Source compilation | `python -m compileall -q src scripts examples` passed | Executed locally after final source edits |
| Editable package installation | `pip install -e . --no-build-isolation --no-deps` passed | [install.txt](install.txt) |
| CLI / no-network doctor | Passed | [cli-help.txt](cli-help.txt), [doctor.json](doctor.json) |
| Wheel build | Passed, universal Python wheel generated | [wheel-build.txt](wheel-build.txt) |
| Synthetic end-to-end watch and investigation | Completed | [synthetic-demo.json](synthetic-demo.json), [synthetic-scores.json](synthetic-scores.json), [HTML](synthetic-report.html) |
| Four-policy synthetic baseline matrix | Completed | [synthetic-matrix.json](synthetic-matrix.json) |
| Real localhost HTTP server, synthetic backend | Auth failure without token; accepted frame; emitted alert; answered with evidence | [live-http-smoke.json](live-http-smoke.json), [server log](live-server.log) |
| Actual FFmpeg generated-video PTS extraction | Passed as part of pytest | `tests/test_replay.py::test_ffmpeg_prepare_with_pts` |
| API schema contracts | Mocked Chat Completions, embeddings, and WAV transcription passed | `tests/test_backend.py`, `tests/test_specialists.py` |

A source-export verification receipt is added separately as [source-export-smoke.json](source-export-smoke.json)
after testing the clean staged source tree.

## What the synthetic results mean

The fixture contains generated red-box frames and a scalar temperature event. The explicit mock
backend reads red pixels and follows a fixed tool policy. The adaptive fixture detected its three
expected events and answered two exact fixture questions. That checks integration only.

The matrix made 69 mock attempts for fixed-rate, 19 for motion/cue-plus-periodic, 25 for adaptive,
and 33 for recent-only. All matched the three toy events; recent-only did not answer the historical
fixture questions correctly. Motion and adaptive achieving identical toy event recall at different
call counts does **not** imply either will be better on real footage. Event latency, richer tasks,
false positives and model behavior require actual workloads. There is no real API bill or GPU work
in this synthetic comparison. Do not use these numbers as a startup performance claim.

## Test coverage

Time validity; frozen source snapshots; late/backdated arrival; summary lineage; legitimate historical
materialization; future-corpus keyword-rank isolation; vector fingerprints/dimensions; immutable IDs;
namespace boundaries; queue admission/coalescing/priorities/deadlines/recovery; unknown/error/retry
billing; invalid provider usage; no secret traces; explicit network opt-in; bounded tool access and
citation IDs; sampled dwell and stale/old completion handling; shared watch perception; scalar events;
archive without eager captioning; label separation; paced replay caveats; API authentication; evaluator
future-reference masking; Video-MME label split/export; and local event/QA scorer behavior.

## Not executed or established

**No real model API calls, hosted embeddings/ASR, GPU inference, vLLM/SGLang service, native video
pruning/cache path, NVIDIA VSS deployment, real RTSP camera feed, full StreamArena/Video-MME evaluation,
official ActEV/FPS/proactive/audio benchmark, real customer footage or commercial saving was tested.**

The optional PyAV test is skipped because `av` was not installed. An installation attempt for PyAV
and Ruff failed due to package-host DNS resolution in this environment. FFmpeg provided a working
local video path instead. Ruff lint was not run. Docker was not built. GitHub Actions YAML was not
executed remotely. Python 3.11/3.12 are declared targets but only 3.13.5 was run locally.

The StreamArena bridge is interface-tested, not a reproduced upstream run. The VSS mapper operates
on already-decoded explicitly mapped JSON, not a running VSS/Kafka/protobuf installation. The original
specification and future integration work are described in [HANDOFF.md](../HANDOFF.md).

## Reproduce

```bash
python -m pip install -e '.[dev,server]'
pytest -q --junitxml=results/your-pytest.xml
python -m compileall -q src scripts examples
streambudget doctor --config configs/mock.yaml
streambudget demo --out runs/your-demo
python scripts/run_matrix.py --events runs/your-demo/input/events.jsonl \
  --tasks runs/your-demo/input/tasks.jsonl --labels runs/your-demo/input/labels.jsonl \
  --out runs/your-matrix
```

Use fresh run directories. Real provider validation is a separate explicit-opt-in operation. The
handoff describes its expected outputs and budget constraints.
