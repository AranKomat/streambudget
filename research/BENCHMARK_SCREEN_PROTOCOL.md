# MMVU / TOMATO subset screening

Preregistered local protocol, 2026-09-20. LVBench was removed at the user's request
before any paid calls or LVBench media acquisition. This is model screening, not
an official leaderboard reproduction or a StreamBudget scheduling-policy result.

## Frozen inputs

- 20 MMVU public validation multiple-choice questions, stratified by subject.
- 20 TOMATO questions, stratified across its six categories.
- Seed `streambudget-bench-20260920-v1`: SHA-256 ordering within and across strata,
  round-robin selection with unique source videos per benchmark. No answer-based
  selection, difficulty filtering, or replacement of missing media.
- Full-video uniformly spaced PTS seeks, up to 32 MMVU / 64 TOMATO distinct frames;
  no audio, separate subtitles, answer timestamps, rationales, or annotation hints.
  Images are at most 768 pixels per side, JPEG quality 85. Short sources can
  contain fewer distinct frames than the cap.
- Identical timestamped multi-image Chat Completions packets across all models,
  not native video ingestion. JSON option letter, evidence IDs, brief reason.
- OpenRouter models: `z-ai/glm-5.3-flash`, `qwen/qwen3.8-27b`, and
  `google/gemini-3.8-flash`. Low reasoning, temperature zero, 4,096 output tokens,
  no retries. These hosted aliases are not immutable checkpoint revisions.
- Gemini: only `google-ai-studio/flex`, no fallback; response must confirm
  Google AI Studio and `service_tier=flex`.

Primary metric is exact answer-letter correctness among all 20 planned cases per
benchmark/model, with failed or invalid structured outputs counted as incorrect.
Citation-ID validity is only a diagnostic; no human evidence-support grading is
performed. An empty benchmark `anchor_groups` is not proof of grounded reasoning.
One question is five percentage points, so this is a small exploratory screen.

## Sources and rights

- [MMVU](https://github.com/yale-nlp/MMVU), author-hosted HF annotations and videos:
  `yale-nlp/MMVU` revision `b937f414a87e9012acba49d95669020b24fa9ee9`.
  No explicit data license was located. Local evaluation only; no source-data
  redistribution or claim of commercial data rights.
- [TOMATO](https://github.com/yale-nlp/TOMATO), annotations at commit
  `fe2025f4b9e1ce339618e0eecfd10f09caf67142`; HF card states CC BY-SA 4.0.
  Media comes from the authors' linked Drive archive. HTTP byte ranges retrieve
  only selected ZIP members; ZIP CRC and acquired file SHA-256 are recorded.
  Drive content itself is not immutable, so retain local video hashes.

The code's MIT license does not cover these datasets. Downloaded video, questions,
labels, raw model outputs and complete receipts stay under ignored `data/` and
`runs/`. Public exports contain IDs, hashes, protocol, routing and metrics only.

## Admission and execution

Original approval: 183 calls / $35 for three benchmarks. Reduced scope:
**123 calls / $10 estimated-admission ceiling**, including three compatibility
probes and 120 answers. All models share one ledger; probe costs and unknown holds
carry into screening. Reservations are not a guaranteed provider invoice cap.
Nine concurrent calls, no automatic retry or interrupted-phase redispatch.

```bash
python scripts/prepare_benchmarks.py plan --out data/benchmark-mmvu-tomato-20260920 --academic-use-confirmed
python scripts/prepare_benchmarks.py acquire --out data/benchmark-mmvu-tomato-20260920 --academic-use-confirmed --workers 4
python scripts/screen_models.py prepare --config configs/benchmark-screening.yaml --out runs/benchmark-screen-20260920 --dataset data/benchmark-mmvu-tomato-20260920 --benchmark
# Inspect estimate, image packets and test results before explicit paid dispatch.
python scripts/screen_models.py probe --config configs/benchmark-screening.yaml --out runs/benchmark-screen-20260920 --dataset data/benchmark-mmvu-tomato-20260920 --benchmark --allow-network
# Only after all three probes pass:
python scripts/screen_models.py screen --config configs/benchmark-screening.yaml --out runs/benchmark-screen-20260920 --dataset data/benchmark-mmvu-tomato-20260920 --benchmark --allow-network --concurrency 9
python scripts/screen_models.py report --config configs/benchmark-screening.yaml --out runs/benchmark-screen-20260920 --public-safe --report-out research/benchmark-screen-20260920.json
```

Use fresh directories and an environment-provided credential. Never rerun an
interrupted paid phase in a new directory without reconciling its ledger first.
Public benchmark outputs must use `--public-safe`; default reports remain private.
