# Provider and Gemini tool-loop diagnostic

## Scope and cost

User-approved new admission limit: **60 calls / $1**. Actual: **47 calls**,
**$0.0163152946 reported**, plus **$0.0082756875 provisional holds** for three
requests with unknown billing. These holds are not confirmed charges or an invoice
guarantee. No retries, quota borrowing, grading calls or additional benchmark runs.
Machine-readable aggregate metrics: [JSON](PROVIDER_FOLLOWUP_20260920.json).

This is a small development diagnostic, not a provider-quality leaderboard or a
fresh benchmark. Six shared synthetic cases per provider cover three perception
requests and three planner requests. Cases are handcrafted and not independent
samples from a representative workload. **Do not rank providers by 6/6 vs 5/6.**

## Provider checks

The user's priority remains **Z.AI, Fireworks, Novita, Sail Research, Together, Modal**.
Providers were tried in that order; the six cases within each provider ran concurrently.
Each request was pinned with fallback disabled, and returned provider identity was
checked. Normal future pilot configuration permits ordered fallback only within this
list. Historical frozen configurations and results were not rewritten.

| Provider | Valid contracts | Observation |
|---|---:|---|
| Z.AI | 5/6 | One planner output lacked the required `tool` key |
| Fireworks | 6/6 | Passed these six probes only |
| Novita | 4/6 | Two planner outputs lacked the required `tool` key |
| Sail Research | 3/6 | All three image requests failed; text probes passed; billing unknown for failures |
| Together | 6/6 | Passed these six probes only |
| Modal | 6/6 | Passed these six probes only; listed token rates were higher |

There is no basis here for saying Fireworks is better than Z.AI or Novita. The
defensible observation is intermittent contract failure with the current prompts,
including on preferred providers. This does not isolate model variability from
provider implementation, precision or serving effects. Sail's failed image requests
must be diagnosed separately; their retained error classes do not establish the
specific HTTP/capability cause.

Provider probes used the original perception/planner prompts from `44e15a0`, before
the tool-loop edits in this follow-up. Their hashes are retained in the JSON receipt.
No provider probe was rerun after editing the planner prompt, so improvement of GLM
contract compliance from the new prompt is **not yet established**.

Current catalog and routing documentation checked on 2026-09-20:
- https://openrouter.ai/api/v1/models/z-ai/glm-5.3-flash/endpoints
- https://openrouter.ai/docs/guides/routing/provider-selection

Listed per-million input/output prices: Z.AI, Fireworks and Together $0.15/$0.50;
Novita $0.132/$0.44; Sail $0.1425/$0.475; Modal $0.45/$1.50. Catalog listings are
not availability guarantees. Usage receipts, including unknown charges, remain the
accounting authority. Returned provider names do not attest the exact quantization.

## Gemini diagnosis

Both original queries exhausted six planner actions. The first sought an audio cue
that the vision-only input could not supply. The second searched and inspected
visual evidence but did not reach a final answer within six actions.

The updated planner receives remaining actions, frame availability and whether
audio-derived evidence exists **at its frozen cutoff**. It is told to finish on its
last action, explicitly abstain when evidence is insufficient, and use exactly the
typed action envelope. Malformed actions consume a step and return sanitized
feedback; no tool is executed from them. Typed tool-argument failures can likewise
be corrected within the original step bound. No guard is relaxed and no free repair
call is added. Perception/parser diagnostics record safe field paths and error
types, not model text or arbitrary field names.

Rechecked the two retained query states independently, without replaying the video
or collecting new background captions. Imported only original evidence whose source
time/availability and creation wall time preceded the original request. Later
derivations are excluded even when they consume old frames. Parent lineage and
media hashes are validated; labels are never loaded by the runner. Each query has
its own 12-call/$0.25 partition, including perception tools.

| Query category | Previously | Now | Calls | Reported cost |
|---|---|---|---:|---:|
| Audio-dependent | Step limit | Explicit abstention after sensor check | 2 | $0.001225125 |
| Visual identity/history | Step limit | Explicit abstention after search, state, inspections and OCR | 9 | $0.010749375 |

**Termination improved; answer accuracy did not.** Both abstentions mean the
requested fact remains unestablished. These post-hoc states are development inputs,
not a new held-out test. The visual query's failure still points to retrieval,
sampling/OCR or evidence coverage; this run cannot attribute it solely to Gemini.

Audio omission is fatal to identifying events anchored specifically to a horn or
spoken phrase, not to visual monitoring as a whole. Future evaluation should report
vision-only and audio-dependent tasks separately, or add aligned audio processing
with its cost counted. ASR alone would not cover non-speech events such as horns.

## Reproduce the diagnostic

With authorized credentials and a fresh output directory:
```bash
python scripts/qualify_glm_providers.py --out runs/NEW/providers --allow-network
python scripts/check_gemini_tool_loop.py --pilot runs/streamarena-pilot-20260920 --out runs/NEW --allow-network
python scripts/report_provider_followup.py --run runs/NEW --out research/NEW.json
```
The provider script uses the current prompts; reproducing this receipt's original
prompt comparison requires the recorded pre-edit prompt versions. The frozen
original pilot and protected query data stay local. Do not rerun without a new
authorized budget. For subsequent pilot preparation, provider order now comes from
`configs/streamarena-followup.yaml`; old benchmark screening configs remain intact.
