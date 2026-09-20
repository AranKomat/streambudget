# Frozen StreamArena pilot protocol

## Scope and rights

Separate, user-confirmed non-commercial academic research using StreamArena's
CC-BY-NC-4.0 annotations. Do not redistribute its media, annotations, questions,
answers or raw model completions in this public repository. No LVBench or paid
long-video expansion is part of this pilot.

- Upstream: https://github.com/JIA-Lab-research/StreamArena at
  `ec0325b9552b045c2e53afbf425315ce867313ee`.
- Dataset: https://huggingface.co/datasets/hkuzxc/StreamArena at
  `2960c360deba57b717a17ab66a3ae80f5358a6a3`.
- Selection seed: `streambudget-online-20260920-v1`.
- Hash-order videos containing RTP, HR and Pro asks before 600 seconds; take the
  first two and all qualifying asks. No answer, reference-time or evidence-time
  filtering. Selected IDs: `JNpUsYTVM6k`, `9CQ6qmoOhlQ`, four asks each.
- Sample uniformly from each video's start to 600 seconds: 1,200 source-PTS
  frames at 2 FPS, JPEG max side 768. Vision only, including any audio-dependent
  questions. No oracle crops, transcripts or future labels in runtime input.
- Range-read only needed MP4 blocks inside pinned TAR archives. Check expected
  archive length and hash derivative JPEGs. Full TAR hashes are NOT verified.

## Matched policy run

Six GLM runs: fixed, motion plus periodic refresh, and adaptive on both videos.
One Gemini adaptive comparison on the first selected video. All seven run
concurrently, with isolated memory/budgets, at speed one using predecoded replay.
Original video decode and acquisition are recorded separately, not included in
runtime timing. A separate generated-video smoke test exercises the unmodified
upstream native driver with the optional native bridge; that is plumbing only.

All policies retain the same caption/lexical memory. This is NOT memory vs no
memory. Common settings: five-second minimum/fixed/monitor intervals, 15-second
refresh, adaptive audit probability .03, four recent frames within eight seconds,
16-frame maximum inspection, six tool steps, four workers, queue size 16,
60-second queue deadline, 40-second HTTP timeout and 90-second final drain.
No escalation, verification, embeddings, ASR, compaction or retries. Caption and
query model work are both metered. Every model role falls back to the same model
within its trial. Gemini is pinned to Google AI Studio Flex, with returned-route
verification. Frozen configs preserve exact model IDs and token prices.

## Cost correction and admission

The earlier $6-9 forecast incorrectly treated worst-case output/image reservations
as expected usage. The illustrative token-based scenario is **$1.1262**, not a
promise: $0.3888 GLM observations, $0.2625 GLM queries, $0.3924 Gemini checks,
$0.0825 judging. Measured provider-reported cost supersedes that scenario.

User-authorized admission limits: **1,200 attempts / $10**. Non-overlapping quotas:
six GLM trials each 150 calls/$1; Gemini 150 calls/$3; grading 150 calls/$1.
No quota borrowing, retries or automatic redispatch. Reservation estimates remain
conservative safety guards and are not invoices or expected spend. Unknown-charge
holds survive failures. Provider billing may exceed estimates; the local dollar
limit is an estimated-admission control, not a provider-enforced invoice cap.

## Grading and interpretation

Read labels only after runs finish. RTP/HR: first recorded answer, retaining all
errors, missing answers and step-limit failures. Pro: first emitted notification;
strict delivery window `[reference - 0.5, reference + 2]` seconds. Targets whose
reference is at/after 600 seconds are right-censored, not replaced or counted as
ordinary misses. Report semantic event matching separately from strict timing,
early/late alerts and delivery delays. Five-second polling plus hosted latency is
not expected to satisfy a two-second window reliably; do not change the rule.

Use bounded Gemini Flex text-reference judging with equivalent wording/language
accepted, incorrect facts and abstentions rejected. This is a custom rubric/model,
NOT the exact official judge. It cannot independently establish pixel grounding.
Meter grading and retain judge failures as ungraded cases in eligible denominators.

Report attempts, dollars, unknown holds, operation counts, model latency, query
status, observation skips, errors/drops, censoring and returned provider routes.
Eight unique questions and a single Gemini prefix cannot establish a model ranking
or general commercial savings. No tuning or sample replacement after outcomes.

## Commands

```bash
python scripts/prepare_streamarena.py plan --root data/streamarena-pilot-20260920 --academic-use-confirmed
python scripts/prepare_streamarena.py acquire --root data/streamarena-pilot-20260920 --academic-use-confirmed
python scripts/streamarena_pilot.py prepare --root data/streamarena-pilot-20260920 --out runs/streamarena-pilot-20260920
# Requires authorized OPENROUTER_API_KEY; fresh run directories only.
python scripts/streamarena_pilot.py run --out runs/streamarena-pilot-20260920 --allow-network
python scripts/streamarena_pilot.py grade --out runs/streamarena-pilot-20260920 --allow-network
python scripts/report_streamarena.py --run runs/streamarena-pilot-20260920 --out research/STREAMARENA_PILOT_20260920
```
