# Local validation receipt

## Provider and tool-loop follow-up (2026-09-20)

- **139 tests passed**, two dependency deprecation warnings; targeted Ruff and
  `git diff --check` passed. Added provider identity/billing, quota conservation,
  diagnostic redaction, remaining-action limits, explicit abstention, frozen audio
  visibility, malformed-action feedback, reconstruction cutoffs/media integrity,
  and protected-answer export checks.
- Isolated no-key demo: `runs/provider-followup-no-key-demo-20260920`, 25 mock calls,
  $0, two toy answers and three toy alerts. Not a model-performance result.
- [Diagnostic report](../research/PROVIDER_FOLLOWUP_20260920.md): 36 pinned synthetic
  provider calls plus 11 Gemini query calls. $0.0163152946 reported; $0.0082756875
  provisional holds for three requests without billing receipts. Do not assume zero
  charge for those failures. No retries and no full new benchmark replay.
- Both Gemini queries now explicitly abstain instead of exhausting their tool loop.
  Neither establishes the requested answer. Provider sample size is too small to
  justify a ranking; the user's priority order remains unchanged.

## StreamArena prefix pilot (2026-09-20)

- **128 tests passed**, two dependency deprecation warnings. Targeted Ruff and
  `git diff --check` passed. Added config-freeze roundtrip/tamper, protected-text
  export, quota conservation, selection independence, bounds and censoring tests.
- No-key demo: `runs/streamarena-no-key-demo-20260920`, 25 mock calls, $0,
  2/2 toy answers and three toy alerts. Plumbing only.
- Native smoke: `scripts/check_streamarena_native.py` executed the unmodified
  pinned StreamArena driver with generated footage and the optional wrapper:
  three answer records, 24 mock calls, no paid inference. OpenCV/PyAV used locally.
- Two pinned 600-second prefixes prepared concurrently: 2,400 source-PTS JPEGs,
  576,727,048 source range bytes. Derivative hashes validated. Full source TAR
  checksums were not verified because complete archives were not downloaded.
- Config freeze initially failed before dispatch because validation coerced integer
  fields to floats. Fixed representation hashing with a regression test; no paid
  retry, altered setting or source replacement was involved.
- [Protocol](../research/STREAMARENA_PILOT_PROTOCOL.md) and
  [results/review](../research/STREAMARENA_PILOT_20260920.md): **911 calls,
  $0.5339023542**, including 13 grading calls. No unknown-charge holds, no retries,
  no quota exhaustion; sub-$1e-15 reservation remainder is numerical roundoff.
- Preserved all outcomes: 25 eligible task-trial pairs plus three right-censored;
  1/14 question reference matches, 0/11 strict proactive hits. Text-only matching
  overcredits event identity and does not establish visual grounding. All protected
  data/raw answers remain local under ignored `data/` and `runs/` directories.

## MMVU / TOMATO screening runner (2026-09-20)

- **119 tests passed**, two dependency deprecation warnings. Targeted Ruff and
  `git diff --check` passed.
- Added coverage for answer-independent stratified selection, six-option answers,
  evaluator metadata exclusion, image hashes/path/timestamp bounds, range-only ZIP
  access, full-duration PyAV sampling, dynamic call counts, and public-safe exports.
- Isolated no-key demo: `runs/benchmark-report-no-key-demo`, 25 mock attempts, $0,
  2/2 toy answers and three toy events. Plumbing only.
- Prepared all 40 fixed benchmark clips successfully (234,881,036 source bytes),
  with no replacement questions. Exact full-output/image reserve quote: $8.033684
  for 123 calls. Local budget reduced to $10 after removing LVBench.
- [Protocol and rights](../research/BENCHMARK_SCREEN_PROTOCOL.md). Real model
  [results](../research/BENCHMARK_SCREEN_20260920.md): 123 paid calls, $1.1422144168,
  one charged truncation included in cost/latency and scored incorrect. The report
  regression test ensures rejected responses do not disappear from model costs.

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
