# StreamArena prefix pilot (2026-09-20)

Separate non-commercial research; custom protocol, not an official StreamArena score.

**911 model attempts; $0.533902 reported cost**, including grading.
Uncertain/provisional holds: $0.000000; reserved: $0.000000.

| Trial | Calls | Reported $ | RTP | HR | Pro text match | Pro strict | API p50 / p95 (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| glm-fixed-JNpUsYTVM6k | 134 | 0.049247 | 0/1 | 0/1 | 2/2 | 0/2 | 3.52 / 16.62 |
| glm-motion-JNpUsYTVM6k | 128 | 0.047764 | 0/1 | 1/1 | 2/2 | 0/2 | 3.79 / 8.67 |
| glm-adaptive-JNpUsYTVM6k | 131 | 0.048405 | 0/1 | 0/1 | 2/2 | 0/2 | 3.17 / 19.38 |
| glm-fixed-9CQ6qmoOhlQ | 131 | 0.047815 | 0/1 | 0/1 | 1/1 | 0/1 | 4.19 / 14.65 |
| glm-motion-9CQ6qmoOhlQ | 120 | 0.044812 | 0/1 | 0/1 | 1/1 | 0/1 | 2.90 / 11.13 |
| glm-adaptive-9CQ6qmoOhlQ | 119 | 0.043333 | 0/1 | 0/1 | 1/1 | 0/1 | 3.06 / 7.10 |
| gemini-adaptive-JNpUsYTVM6k | 135 | 0.251041 | 0/1 | 0/1 | 2/2 | 0/2 | 3.08 / 4.86 |

Denominators retain errors/misses/ungraded cases, but exclude explicitly right-censored targets.
See the paired JSON for censoring, unknown grades, failures, routes, counts and delivery delays.

## Limitations

- Eight distinct questions from two hash-selected ten-minute prefixes; not a model leaderboard.
- Vision-only 2 FPS, 768-pixel JPEG input; audio-dependent questions retained without audio.
- Seven concurrent speed-one predecoded replays; original decode excluded from runtime timing.
- Same caption/lexical memory in all three GLM policies; not a memory ablation.
- Gemini adaptive tested on only the first video; no full matched model comparison.
- GLM used multiple OpenRouter providers, not a pinned serving endpoint; this confounds comparisons.
- Five-second observation floor plus API latency conflicts with strict two-second Pro timing.
- First Pro alert judged; post-prefix reference events right-censored, never replaced.
- Custom Gemini Flex text-reference grading, not the official judge or independent visual grounding.
- No retries; failures and ungraded cases retained. Unknown billing is not zero cost.
- Source revision/archive lengths pinned; derivative hashes checked, full archive SHA not verified.

[Frozen protocol](STREAMARENA_PILOT_PROTOCOL.md). Protected questions, labels, footage and raw answers are not redistributed.

## Analyst review

This section is a manual review of the retained private traces, not generated grading output.

- **Cost:** $0.2813758542 for 763 GLM calls, $0.25104075 for 135 Gemini calls,
  and $0.00148575 for 13 Gemini grading calls. The original $6-9 forecast was wrong;
  the corrected $1.13 scenario was conservative. No unknown-charge holds remain;
  the residual reservation below $1e-15 is floating-point roundoff.
- **Runtime qualification failed:** 152 of 699 GLM background observation jobs
  failed (21.7%). These were validation/JSON/contract failures plus one scheduler
  timeout, not 152 failed HTTP requests. Gemini completed all 120 background jobs
  successfully. Failure logs retain classes, not the rejected response bodies, so
  the exact malformed fields still require a controlled diagnostic.
- **Questions:** six errors, six tool-step-limit outcomes, two submitted answers.
  Only one of 14 task-trial pairs matched the reference. That credited answer also
  contained an identity mismatch in its rationale; independent visual grounding
  has not been established. Do not interpret this as a clean VLM capability score.
- **Proactive timing:** 0/11 eligible alerts passed the strict timing rule. All
  arrived late, by 4.3-88.4 seconds. Three task-trial pairs were right-censored.
  The text judge accepted all 11 descriptions, but one described a much later
  occurrence. Text matching alone does not prove correct target-event identity.
- **Gating:** all three GLM policies scheduled 120 background observations on the
  moving-camera prefix. On the livestream, fixed scheduled 120, motion 109 and
  adaptive 110. This is modest scheduling reduction, not demonstrated preserved
  quality or large savings. GLM provider routing also varied between calls.
- **Gemini tool loop:** its two queries consumed six steps each without answering.
  The first searched an absent audio/sensor index and caption memory; the second
  searched, then inspected images, but exhausted the step budget. Adding more
  steps blindly is not yet justified; expose capabilities and remaining budget,
  distinguish abstention from loop exhaustion, and inspect retrieval coverage.

**Next gate:** pin and qualify a GLM provider with precise structured-output
contracts and sanitized validation diagnostics; qualify tool completion/abstention
within the existing budget; explicitly separate vision-only from audio-dependent
tasks. Then run a newly frozen prefix set. Do not tune on this pilot and present a
rerun as fresh held-out evidence. No more paid runs were dispatched after grading.

Runtime source: `437199a` (the pre-dispatch hash fix changes no experiment settings).
Source preparation retained 2,400 JPEGs and transferred 576,727,048 range bytes,
rather than fetching both complete archives. Preparation wall times were 431.8 and
1,170.6 seconds concurrently; shared-process CPU measurements must not be summed.
The seven real-time trials then completed in 599.5-628.3 seconds each.
