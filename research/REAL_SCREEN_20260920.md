# Real-footage exploratory screen

Date: 2026-09-20. **Real recorded footage and paid model calls; not StreamArena,
Video-MME, a representative video benchmark, or a runtime-policy evaluation.**
Full prompts, source attribution, image hashes, individual answers, costs and returned
provider/tier metadata: [machine-readable receipt](real-screen-20260920.json).

## Results

| Model | Answer accuracy | Valid JSON / citation IDs | Median / p95 HTTP latency | Cost including probe |
|---|---:|---:|---:|---:|
| GLM-5.3-Flash | 20/20 | 20/20 | 2.46 / 5.16 s | $0.012952 |
| Qwen3.8-27B | 20/20 | 20/20 | 5.74 / 23.79 s | $0.029120 |
| Gemini 3.8 Flash, AI Studio Flex | 20/20 | 20/20 | 3.67 / 7.06 s | $0.074935 |

Three successful synthetic compatibility probes plus 60 real-footage calls: **63
attempts, $0.11700802 reported**, no retries, HTTP failures or unknown-cost holds.
The newly authorized cap was 63 calls / $3; preflight reservation estimate $1.090212.
All Gemini responses, including the probe, explicitly reported `Google AI Studio`
and `service_tier: flex`. No Gemini request went to Vertex or standard-tier pricing.
Routing configuration follows OpenRouter's [service-tier documentation](https://openrouter.ai/docs/guides/features/service-tiers)
and the observed [Gemini endpoint catalog](https://openrouter.ai/api/v1/models/google/gemini-3.8-flash/endpoints):
`service_tier: flex`, `provider.only: [google-ai-studio/flex]`, `allow_fallbacks: false`.

The 60-request screen took approximately **44.18 seconds** from first dispatch to
last result with a global concurrency of nine. This excludes media acquisition,
preparation, annotation, probes and internal semaphore waiting from individual HTTP
latency. It is not nine-camera throughput, a streaming deadline result or GPU time.

Qwen and GLM were automatically routed across multiple providers, recorded per request.
Qwen used eleven provider names across 21 calls; GLM used five. Their observed tail
latencies are **routing outcomes**, not intrinsic model-speed rankings. Gemini's tier
also changed relative to the earlier synthetic screen, so those runs are not a
controlled standard-versus-Flex latency comparison.

## Sources and rights

Media were downloaded from Wikimedia Commons only after checking each file's rights
metadata. Original videos remain under ignored `data/real-screen-20260920/`; no video
or extracted image is redistributed in this repository. The receipt preserves source
URLs, file-page metadata, attribution, source hashes and license links.

| ID | Original file title | Attribution | License |
|---|---|---|---|
| c01 | Gigaset Cordless Telephone Production VII - Pneumatic Conveyor Belt.webm | Video/photography: Inke Pickhardt; sound: Kathinka Engels | CC BY 3.0 |
| c02 | Gigaset Cordless Telephone Production II - Engel Injection Moulding Machine.webm | Video/photography: Inke Pickhardt; sound: Kathinka Engels | CC BY 3.0 |
| c03 | JFC-UA Service Members Train NGOs, Liberians on Forklift 150110-A-YW926-001.webm | Sgt. David Greeson | Public domain per source page |
| c04 | Twin Cities METRO Blue Line Doors Closing Chime.webm | chiefbozx | CC BY 3.0 |
| c05 | Red green flashing pedestrian crossing.webm | Chidgk1 | CC0 |

[CC BY 3.0](https://creativecommons.org/licenses/by/3.0/) and
[CC0](https://creativecommons.org/publicdomain/zero/1.0/). Derivatives used for inference:
selected JPEG frames, resized to a maximum side of 768 pixels, JPEG quality 85. Source
aspect ratios were preserved. Audio was not sent. No endorsement by the creators is implied.

## Protocol

Five clips, four multiple-choice questions per clip, eight shared timestamped frames
per question. Source JPEGs were prepared at requested 2 FPS using presentation timestamps;
the selected packets are **not uniformly 1 FPS**. The forklift source was decoded only
through 60 seconds; its selected packet ends at 36.036 seconds. Full originals remain
available for future dense inspection. Source titles/metadata/answers were not given
to the models. Only questions, options, neutral evidence IDs, timestamps and images
were sent, all bounded by the stated observation cutoff.

Questions cover machinery and door-state changes, ordering, OCR, counting, spatial
description and explicit uncertainty about unavailable daily totals/future actions.
No external tools, audio, online memory or scene pre-mapping were used. This is frozen
prefix QA, not a live causal runtime execution or native-video evaluation.

The agent visually inspected the real images and authored the labels before footage
dispatch; **no independent human annotation audit was performed**. One question was
clarified before dispatch to ask for the leftmost digit of the visible marking `91`.
The earlier manifest and amendment reason are preserved locally; probe charges were
retained. No answer keys or questions changed after real-footage responses arrived.

Config: `configs/footage-screening.yaml`. All models requested low reasoning effort,
temperature 0, JSON output, 4096 output-token cap and no retries. These settings do
not imply equal internal reasoning or image tokenization. A shared reservation ledger
accounts for every attempt. Validated OpenRouter charges, not catalog minima, settle
the ledger. Reported charges are not independently reconciled invoices.

PyAV preparation across the five clips recorded approximately 9.39 wall seconds and
11.91 process-CPU seconds in total. Downloading, selecting questions and creating
review contact sheets are additional work, not included in inference cost/latency.

## Evidence diagnostic and limitations

Frozen anchor scores: GLM **19/20**, Gemini **19/20**, Qwen **20/20**. On `r08`, both
GLM and Gemini cited frames 0, 2 and 5, correctly showing an initially open machine,
closed state at 6.5 seconds, and reopening at 22.5 seconds. The evaluator demanded
frame 3 or 4 for the intermediate closed state and omitted the equally valid frame 2.
This is a scorer limitation, not evidence of wrong temporal reasoning. Original scores
are preserved; no post-hoc score upgrade is used to rank models.

The main answer metric is exact-choice accuracy, not an audit of every sentence of
the rationale. Evidence-ID validity alone does not prove semantic support. The anchor
heuristic needs a complete equivalence audit before becoming a selection criterion.

This is a small, correlated set with many easy distractors and broad descriptive
questions. Several clips are near-static; only a subset tests meaningful temporal
reasoning. Twenty correlated questions over five hand-selected clips cannot establish
general video quality, difficult-event recall, long-horizon memory, or model superiority.
Public media may also have been present in model training. All three reach ceiling:
**this screen is insufficient to pick a quality winner**.

## Practical next step

Use GLM as the provisional low-cost baseline for a **fixed-rate versus adaptive
StreamBudget runtime experiment** on door and machinery transitions. Keep Gemini AI
Studio Flex as a comparison model; retain Qwen as the open-weight candidate. First
freeze independent event intervals and complete citation-equivalence groups, then
compare missed events, duplicates, evidence-grounded answers, calls, cost and latency
under matched budgets. Do not spend another round merely repeating easy static QA.

No additional model calls were made after exhausting this phase's 63-attempt allowance.

## Reproduce

Python 3.13.7, Pillow 12.3.0 and the existing PyAV-enabled environment on macOS.
Model/provider revisions remain unpinned. Refresh rights and rates before a new run.

```bash
python scripts/prepare_footage.py --out data/new-footage --download
python scripts/screen_models.py prepare --config configs/footage-screening.yaml \
  --dataset data/new-footage --out runs/new-footage-screen
# With separately authorized spending and OPENROUTER_API_KEY exported:
python scripts/screen_models.py probe --config configs/footage-screening.yaml \
  --dataset data/new-footage --out runs/new-footage-screen --allow-network --concurrency 9
# Inspect receipt and confirmed AI Studio/Flex routing first.
python scripts/screen_models.py screen --config configs/footage-screening.yaml \
  --dataset data/new-footage --out runs/new-footage-screen --allow-network --concurrency 9
python scripts/screen_models.py report --out runs/new-footage-screen
```

No automatic crash resume or billing-reservation release. Do not bypass existing phase
directories by creating a new run without accounting for the old one.
