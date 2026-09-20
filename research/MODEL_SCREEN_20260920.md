# Three-model synthetic compatibility screen

Date: 2026-09-20. **Real paid inference on generated images, not real-video evaluation.**
Machine-readable evidence, frozen prompts/image hashes, individual responses, usage,
request IDs and exact configuration: [receipt](model-screen-20260920.json).

## Result

| Model | Correct answers | Valid JSON / citation IDs | Median / p95 latency | Provider cost, 21 calls |
|---|---:|---:|---:|---:|
| Gemini 3.8 Flash | 20/20 | 20/20 | 3.39 / 4.67 s | $0.152779 |
| Qwen3.8-27B | 20/20 | 20/20 | 6.82 / 20.24 s | $0.020623 |
| GLM-5.3-Flash | 20/20 | 20/20 | 1.47 / 6.66 s | $0.008355 |

All three single-image probes passed. All 60 screening requests returned successfully;
no retries, timeouts, missing responses or unknown-usage attempts. Total **63 calls**.
Provider-reported total: **$0.1817569616**, versus the $3 approved estimated budget.
This is not an invoice reconciliation. Latencies exclude probes and represent
observed HTTP round trips, not GPU time or guaranteed service latency.

## Method

- Three single-image image/JSON probes, inspected before dispatching the main screen.
- Twenty identical packets per model, each containing eight 640x360 JPEGs (quality 85),
  at integer-second timestamps 0 through 7. This screen samples at 1 FPS; it does not
  change the video-preparation CLI's 2 FPS default.
- Sixteen visual questions: arrivals, departures, temporal ordering, counts, signs,
  identity, occlusion, reappearance, sampling uncertainty and one scene-text injection.
- Four single-step tool-choice questions. No actual tool execution or agent loops.
- JSON-object output, temperature 0, requested reasoning effort `low`, output cap
  4096 tokens, automatic image detail, no retries, 180-second timeout.
- One sequential worker per model, three workers in parallel, shared attempt/dollar
  ledger. OpenRouter automatic provider routing; model/provider revisions not pinned.
- 4096 input-token reserve per image. The preflight full-output reservation estimate
  was $1.443267 across all 63 calls. This is an admission estimate, not an invoice cap.
- Answer keys are separate from model packets. Responses are scored locally without
  a model judge. All planned cases remain in denominators, even if requests fail.

The same requested reasoning effort does not guarantee equal compute across models.
Different providers tokenize/resize images differently. For example, GLM's receipts
reported zero reasoning tokens, while the other routes reported reasoning tokens.

## Citation diagnostic limitation

The original strict anchor-coverage scores were Gemini **18/20**, Qwen **20/20**,
GLM **19/20**. These scores and all flagged responses are preserved unchanged.
They should **not** be interpreted as three hallucination failures:

- On `s11` (unseen departure behind an occluder), GLM cited frame 0 plus occluded
  frames 3 and 7; Gemini cited occluded frames 3 through 7. Both correctly abstained
  from claiming the object left. The heuristic specifically demanded frame 2.
- On `s14` (scene-text instruction), Gemini cited frame 0 rather than required frame 7.
  The blue object is identical in all eight frames; that citation supports its answer.

All supplied citation IDs were valid. ID validity alone is not semantic support;
the local `supported_correct` field is only an answer-plus-anchor heuristic. Before
real-footage testing, preregister evidence-equivalence rules rather than penalizing
legitimate alternative frames. No post-hoc adjusted headline score is reported here.

## Cost accounting correction

The probe receipts revealed that routed prices differed from the public catalog:
GLM cost slightly more per token, Qwen less. The backend now recognizes OpenRouter's
validated numeric `usage.cost`, with token-rate estimates as fallback. Negative,
non-finite, boolean and string costs are rejected as charges. Other providers' fields
are not assumed to use OpenRouter's schema. Regression tests cover both paths.

The probes preceded this correction, so their original configured-rate ledger total
($0.0019368) was carried forward conservatively without releasing any budget. The
screen uses reported costs. Thus the cumulative admission ledger is **$0.1819696116**,
slightly above the all-request provider-reported total. There are no unresolved holds
(the receipt's residual reservation near 5e-17 is floating-point roundoff).

## Decision and next experiment

All three candidates clear the transport/format compatibility gate. **Do not eliminate
any based on these ceiling-level toy scores.** GLM is the cheapest, lowest-median-latency
candidate for the next pipeline smoke test; Gemini remains a comparison candidate,
not an assumed quality leader. Qwen remains the open-weight/self-hosting candidate.
MiMo was excluded before dispatch at the user's request.

Next: prepare a small authorized real-footage set with independently labeled questions
and events, retain sufficiently dense source frames, and screen these three models on
matched packets. Include hard temporal transitions, real OCR, occlusions and ambiguous
evidence. Then fix one backbone and compare StreamBudget's fixed/motion/adaptive/recent
policies under matched total budgets. Do not conflate model and scheduling-policy changes.

Not tested here: StreamArena, Video-MME, native video/audio, continuous observation,
long-horizon memory, real-time ingestion, scheduler effectiveness, complete tool loops,
self-hosting, GPU cost, customer footage or commercial savings.

## Reproduction

Python 3.13.7, Pillow 12.3.0, HTTPX 0.28.1, Pydantic 2.13.5 on macOS.

```bash
python scripts/screen_models.py prepare --out runs/new-screen
# Export OPENROUTER_API_KEY securely; paid commands require a new budget authorization.
python scripts/screen_models.py probe --out runs/new-screen --allow-network
# Inspect probe/receipt.json, then:
python scripts/screen_models.py screen --out runs/new-screen --allow-network
python scripts/screen_models.py report --out runs/new-screen
```

The generator is `src/streambudget/screening.py`; settings are in
`configs/screening.yaml`. Local original artifacts are in `runs/model-screen-20260920/`.
Do not rerun interrupted phases in fresh directories without reconciling prior costs.
Generated images and synthetic answer keys are original fixtures, not benchmark data.
