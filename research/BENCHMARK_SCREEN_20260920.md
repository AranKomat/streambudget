# MMVU / TOMATO: three-model screening

Completed 2026-09-20. **LVBench skipped**, as requested, before any paid requests
or LVBench media downloads. Evaluation code: `e050cc2`; a reporting-only follow-up
includes costs and latency from failed responses. No predictions were rerun.

## Results

Twenty fixed questions per benchmark; failures remain in the denominator.

| Hosted model | MMVU | TOMATO | Combined | API errors / truncations |
|---|---:|---:|---:|---:|
| GLM-5.3-Flash | **18/20 (90%)** | 8/20 (40%) | 26/40 (65%) | 0 |
| Qwen3.8-27B | 15/20 (75%) | 9/20 (45%) | 24/40 (60%) | 0 |
| Gemini3.8Flash, AI Studio Flex | 16/20 (80%) | **11/20 (55%)** | **27/40 (67.5%)** | 1 truncation |

These are **small custom-protocol subsets, not official leaderboard scores**.
Unlike the earlier easy real-footage screen, they expose substantial failure rates.
Combining two equally sized samples is only a convenience, not a benchmark-weighted
overall capability measure.

## Actual cost and latency

| Model | Cost, 40 questions + probe | Median request | P95 request | Slowest |
|---|---:|---:|---:|---:|
| GLM | $0.107029 | 4.03 s | 33.76 s | 46.96 s |
| Qwen | $0.209249 | 14.29 s | 66.91 s | 129.51 s |
| Gemini Flex | $0.825936 | 9.35 s | 18.82 s | 22.75 s |

**123 calls, $1.1422144168 provider-reported total**, all charges reconciled, no
unknown holds. A residual reservation of approximately `8.3e-17` dollars is floating
point roundoff. No retries. 3,375,407 input and 62,097 output tokens reported.
All 120 benchmark attempts finished in **246.2 seconds** at concurrency nine,
excluding acquisition and probes. Latencies include the failed attempt, exclude
probe latency, and measure request time rather than GPU execution or queue wait.

The conservative pre-run reservation was $8.033684 against a reduced $10 / 123-call
guard. These are validated provider cost receipts, not an independently reconciled
invoice. Different image tokenization, routing and lower actual output explain why
the conservative reservation was much higher than the reported charges.

All Gemini requests pinned AI Studio Flex, with no Vertex or standard fallback.
All 40 retained Gemini responses (39 answers plus probe) confirmed that route.
The truncated response was rejected before routing metadata was retained; its
$0.034584 charge was captured in the ledger. We cannot independently confirm its
returned tier from the saved response record. It counts as incorrect, not a retry.
GLM and Qwen used multiple OpenRouter providers, so latency describes this hosted
routing configuration, not intrinsic model speed or a controlled self-hosted setup.

## Where they differed

| TOMATO category | Questions | GLM | Qwen | Gemini |
|---|---:|---:|---:|---:|
| Count | 3 | 0 | 1 | 1 |
| Direction | 4 | 3 | 3 | 4 |
| Rotation | 3 | 1 | 1 | 1 |
| Shape and trend | 3 | 1 | 1 | 2 |
| Velocity and frequency | 4 | 2 | 2 | 2 |
| Visual cues | 3 | 1 | 1 | 1 |

Across 40 matched questions, Gemini alone was correct on five cases where GLM was
wrong, while GLM alone was correct on four where Gemini was wrong. Gemini vs Qwen
was four vs one; GLM vs Qwen was five vs three. These tiny paired differences do not
establish a stable model ranking. Nine questions were missed by all three:

`MMVU-validation_743`, `MMVU-validation_532`, `TOMATO-count-74`,
`TOMATO-count-237`, `TOMATO-rotation-172`, `TOMATO-visual_cues-3`,
`TOMATO-velocity&frequency-29`, `TOMATO-visual_cues-67`, `TOMATO-shape&trend-196`.

Every non-truncated response had valid JSON and in-range citation IDs. That does
**not** verify that the cited frames support its reasoning. Primary scoring is
answer-letter correctness; the inherited `supported_correct` field has no separate
evidence-support meaning here because no official anchor rubric was supplied.

## Inputs and limitations

[Frozen protocol](BENCHMARK_SCREEN_PROTOCOL.md) and
[public-safe metrics](benchmark-screen-20260920.json) include exact model IDs,
settings, source revisions, seed, selected IDs and packet hashes.

- MMVU: 20 public validation questions across 20 subjects; 32 frames each.
  Source duration 10-177 seconds, median 32.5 seconds.
- TOMATO: 20 unique videos across all six categories; 18 packets of 64 frames,
  one of 28 and one of 19 distinct frames. Source duration 1.52-15.68 seconds,
  median 8.82 seconds. All original options retained, including six-option cases.
- Uniform full-video sampling with timestamped images, no audio or separate
  subtitles. No answer-dependent cropping, frame selection, or question replacement.
  Sampling can alias motion; these errors cannot all be attributed to reasoning.
- Selection was deterministic and stratified, not deliberately chosen for hardest
  questions. Twenty cases means one answer changes a benchmark score by five points.
- No text-only control, contamination audit, repeated trials, native-video comparison,
  or matched streaming scheduler/memory ablation. This does not demonstrate a
  StreamBudget cost-quality advantage, nor does it measure live event detection.
- Dataset rights remain separate from MIT source-code rights. Questions, answer
  keys, videos, frames and raw model responses are local-only, not redistributed.

## Working recommendation

Use **GLM as the inexpensive baseline** and retain **Gemini Flex as the motion-focused
comparison**. Gemini's small aggregate lead cost about 7.7 times as much in this
screen; its TOMATO lead warrants keeping it, not declaring universal superiority.
Qwen remains a practical 27B self-hosting candidate, but this hosted run does not
justify choosing it over both alternatives on quality or latency.

Before choosing a default for deployment, compare matched runtime policies on
fresh temporal cases with GLM held fixed, and repeat selected conditions with
Gemini. Do not tune on these 40 questions and then report them as fresh validation.
