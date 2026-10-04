# Interactive StreamBudget

Active project specification, status and next-agent handoff. Updated 2026-10-04.
This is the single maintained guide for `streambudget.interactive`; root HANDOFF
points here. Existing SPEC/feature/validation documents and research describe the
retained watch/investigation application, not a competing game plan.

## Objective

Build a generic visual state/action runtime: rendered pixels become persistent,
evidence-backed world beliefs and short categorical actions. Selective higher-level
planning can consume those beliefs and request original images. The first intended
demo is Pokemon Red/Blue progression from a fresh game toward Brock, then Misty.
No native gameplay success is established yet.

Current owner direction: Qwen for System 1, and Qwen may also serve System 2.
The owner screened hosted Qwen variants through OpenRouter's web interface, but
has not separately screened this 27B checkpoint. Local L40S Qwen probes and bounded
intro actions are recorded below. Do not repeat hosted comparisons. New work targets
actor-first asynchronous memory, not more Sol calls. Sol remains optional reference/fallback only.

Use a sufficiently capable available VLM, not necessarily a tiny one. Extraction,
planning, action selection and initial schema compilation are operating roles;
the same checkpoint may serve all four. Do not add a second model just to preserve
a hierarchy diagram. Training and new serving infrastructure wait for measured need.

Allowed inputs: rendered frames, current task/intent, actual button receipts,
within-run beliefs/history and source-linked historical evidence. Model priors may
help but are not visually discovered facts. No emulator RAM, hidden maps/coordinates,
semantic game wrappers, walkthroughs, scripted routes or generated controller code
can enter the actor interface. Game-specific goals belong in configuration.

## Architecture

```text
latest pixels + intent + committed beliefs -> actor -> bounded buttons -> new pixels
       |                                  ^
       +-> sparse index -> enrichment ----+-> shared evidence + temporal memory
       +-> asynchronous Hunyuan/GLM OCR --+              |
                                             selective planning / retrieval
```

One repository, one package distribution and one CLI: `streambudget game ...`.
Existing `streambudget demo/run/serve/...` commands remain. Game buttons are never
tools in the watch agent/server; no physical hardware controller is exposed.

| Module | Responsibility |
|---|---|
| Existing `store.py`, `media.py`, `types.py` | Shared immutable evidence, hashes, snapshots and lineage |
| `interactive/contracts.py`, `ontology.py`, `world.py` | Wire contracts, reviewed schema, beliefs and conversations |
| `interactive/runner.py`, `environment.py`, `locking.py` | Source-bound stepped execution, PyBoy, single-writer ownership |
| `interactive/models.py`, `prompts.py`, `context.py` | Owner-thread admission/settlement, background HTTP, bounded relevant context |
| `interactive/text.py` | Local Hunyuan/GLM transcription, optional RapidOCR baseline and text association |
| `interactive/async_runtime/` | Actor-first coordinator, bounded async transport/jobs and publication-aware memory |
| `interactive/qualification.py`, `report.py`, `fixture.py`, `cli.py` | Probes, reports, explicit test double and commands |

Game actions remain stepped. New trials should use `game background`; the existing
`game run` path, including its earlier async-perception option, remains a labelled
comparison baseline. Both share evidence/media, accounting, environment and OCR
contracts. The retained camera scheduler remains a separate application.

## Actor-First Async Memory

Integrated from the owner's 2026-10-04 update targeting `77e05ef`; reused interfaces
were reviewed against this checkout rather than overwritten. Source and focused tests
live inside the existing package. The delivery installer, duplicate handoff/appendix,
example report and validation copies were not added to the repository. This section
owns the active direction; earlier experiment receipts below remain historical evidence.

The actor reads current pixels without waiting for a rich scene caption. A sparse
scene index proposes a few handles; the owner commits/mints canonical IDs before
forming target-specific enrichment jobs. Index, enrichment and selective planning
share one background HTTP slot; the actor has admission priority. Defaults are two
total HTTP requests, one background request, two enrichment targets, and output caps
of 32 actor / 256 index / 256 enrichment / 1,024 planner tokens. OCR has its own single
local inference worker, so these HTTP slot counts do not include OCR or reserve GPU
resources. All role calls, local OCR setup/inference, drops and failures are recorded.

The main loop alone owns SQLite, capture, buttons and publication. Worker requests
receive frozen inputs, never live database/emulator references. Semantic results have
both source observation sequence and publication revision: late history can add an
old name without rewinding newer location, and an earlier decision cannot retrieve
knowledge that arrived afterward. Conflicting same-source facts stay marked; missing
delta fields do not imply deletion. Identity remains a supported model belief, not
certified re-identification.

`tm_*` tables in the shared evidence database support entities/facts/relations,
events, participant-scoped conversations, observed visits, directed exits/traversals,
source-linked visual references and hot/warm/cold context selection. Mandatory pins
and focus cannot silently vanish to meet context limits. Routes require offered
action evidence; they are historical hints, not motion plans or proof of causality.
Memory tools use the request's causal cutoff and exposed handles; there is no generated
SQL, filesystem/URL lookup, hidden map or extra actuator. Schema compilation/evolution
is disabled in background mode; ordinary facts/instances can accumulate.

An initial valid plan is the default barrier, not rich extraction. Applicable intent
can persist while a replacement plan runs; scene/place boundaries and age limits
invalidate it. Complete validated JSON alone can dispatch, after exact source-frame,
pixel-hash and intent checks. SSE records first token activity, first content and
validated completion separately. Local/tunneled endpoints only; no hosted fallback
or retries. Timeout/uncertain termination quarantines dispatch.

### Selected OCR

`configs/interactive/background.yaml` selects **HunyuanOCR 1.5**, native full-screen,
PIL/SDPA/BF16, with the same plain-transcription prompt as the retained diagnostic.
Use `--ocr-backend glm` to select **GLM-OCR**, 4x nearest full-screen/BF16/SDPA. Only
one reader loads per run; both are not kept resident alongside Qwen. Existing weights
remain in `/workspace/streambudget-ocr/{hunyuan,glm}` on the L40S, never the Mac.
There are no implicit downloads, server restarts or dependency upgrades. Hunyuan uses
the preprovisioned candidate-only Transformers 5.13.0 path in the client process;
GLM keeps the client's existing installation. The Qwen serving process is unchanged.

These plain-transcription modes have **unknown confidence and no region boxes**.
The bridge represents both as null, does not fabricate detections, and does not assign
speakers. Whole-screen occurrence IDs are conservative pixel-based bookkeeping, not
verified dialogue identity; changes elsewhere in a screen can split an occurrence.
Raw response/case/accents/UI symbols are retained. The observed fixed Hunyuan
preamble and exact no-text status response are removed for published text, with a formatting flag; strict
historical benchmark scores remain unchanged. Token-cap outputs fail and are retained
without publishing partial readings. No dictionary, LLM repair or reference labels.

Changed frames coalesce behind a single OCR request; cached reads keep their original
source time. Identical whole-screen readings within an uninterrupted identical-pixel
run publish only once, even when the refresh deadline causes another OCR inference.
Every read and its raw output remains ledger-accounted; deduplicated reads have
`memory_deduplicated: true` and do not refresh the original memory timestamp.
Changed text remains a revision; any intervening observed pixel change starts a new
occurrence, including A -> B -> A when OCR never processed B. This conservative
whole-screen rule does not deduplicate text across sprite/cursor animation or infer
identity from matching words. Region-aware readers can use the existing tracker;
selected generative readers have no boxes, so spatial continuity stays unqualified.
An external host worker can still use `FrameNotice` and `offer_ocr` with
source-linked typed packets. Empty packets do not imply disappearance. Shutdown drains
HTTP/OCR before the checkpoint; unresolved local threads prevent clean resume, stay
accounted, and never write to the closed store. This is not a worker-process supervisor.
PP/RapidOCR is still available explicitly as a baseline, not the selected new reader.
Hunyuan's restrictive license remains documented in its diagnostic below.

### Commands And Qualification

```bash
streambudget game background doctor --config configs/interactive/pokemon-local.yaml \
  --background-config configs/interactive/background.yaml
streambudget game background demo --out runs/background-fixture-001
streambudget game background probe --config configs/interactive/pokemon-local.yaml \
  --background-config configs/interactive/background.yaml --image /private/retained.png \
  --mode mixed --count 3 --out runs/background-probe-001 --allow-network
streambudget game background run --config configs/interactive/pokemon-local.yaml \
  --background-config configs/interactive/background.yaml --rom /private/owned-red.gb \
  --load-state /private/qualified.state --out runs/background-native-001 --allow-network
# Select GLM for a separately labelled new run, without another configuration file:
# add --ocr-backend glm to the run command.
streambudget game background report --run runs/background-native-001
streambudget game background metrics --run runs/background-native-001
streambudget game background memory --run runs/background-native-001 --kind search --text remembered
```

Doctor performs static checks, not inference. The frozen-image probe exercises actor
and index HTTP overlap only, **not local OCR or changing-scene control**. Demo/dummy-SSE
tests are software fixtures, never VLM/GPU/gameplay results. Background reports and
read-only memory queries take the existing run lock. Clean resume uses the same run
path plus `--resume`, verifies source/ROM/checkpoint/database provenance, and allows
only total-budget changes. Switching OCR/model/settings requires a new labelled run.
Old runs are not automatically migrated; a supplied environment checkpoint starts
new memory and must be disclosed.

### Live Async Qualification (2026-10-04)

The existing L40S/Qwen service was kept unchanged; no weights were downloaded to the
Mac. Private receipts under `runs/async-*` were copied locally after completion and
retain every failed response. These small development probes are not task scores.

- `async-actor001`: 3/3 complete action responses, 0.671-0.713 s response wall time,
  0.346-0.390 s first content, on the repeated retained Oak dialogue image.
- `async-mixed001`: both actors completed (0.980-0.981 s); the second index failed
  schema validation because it supplied pixel-coordinate boxes. The index prompt
  now explicitly requires normalized bounds; validation was not relaxed.
- `async-mixed002`: 3/3 actors and 3/3 indices completed with the revised prompt;
  actor wall time 1.002-1.004 s, index 5.874-11.460 s. This run partially overlapped
  the GLM qualification process, so it is a contention smoke check, not an isolated
  actor-versus-mixed speed estimate or proof of sustained tail latency.
- `async-hunyuan001` and `async-glm001`: separate processes successfully loaded
  the selected reader alongside resident Qwen; both transcribed the frozen screen
  as `My name is OAK`. All six concurrent actor probes completed. The private OCR
  ledger's read elapsed time encloses the concurrent actor cohort, not pure OCR
  latency; use the earlier diagnostic for isolated reader timings.
- `async-native001`: 3/3 bounded A actions, 11 settled local calls, no failed calls,
  37.67 s episode wall time, clean checkpoint and no outstanding holds. It started
  new memory from the disclosed old operator checkpoint
  `ec57121381385e0719d566d895354e85e4d464a28d165fcc16ec519c91b079df`.
  Final pixels visibly read `First, what is your name?`; this is intro advancement,
  not overworld/Gym progress. Actor response times were 0.789/1.374/1.065 s.
  One index interpretation was rejected for unsupported identity reuse; no IDs were
  merged by relaxing the identity gate. Two nodes and five memory revisions were
  retained, including three source-bound OCR packets. All action dispatches were
  source-bound; no stale actor was discarded.

Native startup consumed about 29.3 s: an initial planner requested an empty-memory
search, then planned again behind indexing. This is the current demonstrated latency
overhead, not OCR needing to block the actor. Hunyuan also returned its fixed Chinese
no-text status on the blank initial screen; that old response remains in the receipt.
The formatter now suppresses that exact Hunyuan-only status while retaining raw text
and tokens. An empty transcription is not evidence of disappearance. The final frame
was not OCR-read during shutdown; final text above was visually inspected separately.
The formatter fix is CPU-tested but not yet rerun in a new native episode.
Continuation validation: **497 tests passed** with the same two existing warnings;
targeted Ruff, whitespace checks, watch demo and background fixture demo passed.
Local receipt copies matched remote checksums, and the stopped native database
passed SQLite integrity checks with zero pending model calls.

Next: a bounded continuation, checking avoidable planner retrievals and instruction
applicability at the naming/menu boundary. Recurring identity, rich enrichment,
route memory, sustained gameplay, independent grading and changing-scene latency
remain unqualified; no architecture or OCR accuracy gain is claimed from this check.

The supplied 73 focused tests passed against current reused interfaces before local
OCR integration. Added regressions cover selected backends, raw formatting, null
geometry/confidence, source mismatch, setup failures, actor/OCR overlap and unresolved
shutdown. Working-tree integration validation: **495 tests passed**, with the two
existing Starlette/anyio warnings. The isolated staged-public snapshot passed **413
tests**, with the same warnings, independently of unrelated local edits. Both retained
no-key demos and the new background fixture passed; interactive Ruff and Git whitespace
checks passed. A wheel built and its installed background CLI fixture passed outside
the checkout. No new
GPU inference or native gameplay result is implied by this code integration.

## State And Evidence

Fixed node kinds: `entity`, `place`, `surface`. Typed properties, relations, events,
conversations and evidence attach meaning without a class per game noun. New
instances/facts are ordinary learning, not ontology evolution. Properties include
role, appearance, status, text, selected, position_hint and lesson; relations include
located_in, connected_to and interacted_with.

The full world store lives outside the prompt. Context projects relevant entities,
current place, recent observations and retrieved episodes. Optional omissions are
counted; essential task/schema/action definitions cannot silently disappear. The
default 80,000-character cap is not a tokenizer budget. Stable definitions precede
dynamic content, but native KV/visual-encoder sharing is not implemented or guaranteed.

Every source PNG is immutable; interpretations declare parents and remain beliefs.
Facts expose supporting observation times; remembered visibility/location may be
stale. The schema's persistent annotation is not a full expiration policy. Summaries
never replace original source evidence.

New sightings use `new:<name>`. Reusing offered `e_*` IDs requires a sufficiently
confident continuity/re-identification claim; similar appearance alone cannot merge
identity. This is bookkeeping, not an independent identity verifier. Reused sprites
and similar rooms still need native tests. Full frames are primary evidence; optional
source-linked crops aid retrieval but are not segmentation/identity certificates.
Historical images remain historical; entity inspection falls back to full frames.

Conversation threads permit unknown speakers. Text occurrence heuristics distinguish
persistent text from repeated messages but may overcount or miss events. Optional OCR
now feeds source-linked text tracks into the actor and semantic extractor. The general
VLM still interprets dialogue and can read ambiguous pixels itself; OCR does not assign
speakers or semantic object identity. Tracking is heuristic, not a video-text spotter.

`compile_at_start: true` permits one small schema initialization, labelled a prior,
not learned facts. Later additive proposals require evidence, matching parent version,
limits and explicit stopped-run review. They cannot add actions/code or replace
definitions. Post-checkpoint schema edits block automatic resume; reviewed migration
is not implemented. Prefer facts/lessons to new types. Pokemon is not a clean ontology-
discovery benchmark because models may know the domain in advance.

## Actions And Timing

Actor output is only `{"action_id":"RIGHT"}` or another offered ID. The operator
defines eight Game Boy buttons plus WAIT, normally four pressed emulator frames and
four release frames. Each phase is configurable within 1-24 frames. A press is not
one tile, menu item or verified effect. No long generated sequences or arbitrary
`stop_if` instructions are executed.

Before dispatch, actual frame number and pixel hash must match the decision source.
Attempts are journaled as dispatched before execution; buttons release in `finally`.
Receipts count advanced frames, not task success. Uncertain actions are never retried.
Inspect resulting pixels to establish their effect.

PyBoy is GB/GBC (`.gb`, `.gbc`), not FireRed/GBA (`.gba`). Only rendered-screen,
button, tick and operator-checkpoint APIs are used. PyBoy 2.7.0 passed the bundled
smoke test and now operator-controlled Pokemon Red title/menu/dialogue checks.
No overworld/autonomous gameplay or badge completion is established. The owner
supplied a local ROM; none is bundled, downloaded or redistributed by this integration.

Emulation pauses during each actor decision, not background inference. Exact frames and nominal frame/60 source time
are separate from model/wall latency. Boundary screenshots may miss transients inside
an action interval. This is not dense video/audio recording or real-time control.
Doom needs an independent source clock, bounded fresh queues, interruption and
late/dropped-decision measurements before a reactivity claim.

## CPU Setup

```bash
python -m pip install -e '.[dev]'
pytest -q
streambudget game doctor --config configs/interactive/pokemon-local.yaml
streambudget game demo --out runs/game-demo
streambudget demo --out runs/watch-demo
```

Without installation: `PYTHONPATH=src python -m streambudget.cli game ...`.
`make game-demo` runs the interactive fixture; `make demo` retains the watch fixture.
Use fresh output paths. Doctor makes no model calls; its default config retains
`SET_EXACT_MODEL_ID`, while the local YAML now names the selected Qwen FP8 checkpoint.
Static readiness does not mean a GPU server exists or is qualified. The toy game fixture uses a labelled
pixel-rule backend, not a learned model. Its report is a decision slideshow, not video.
Regenerate outputs under ignored `runs/`; no static preview copies are maintained.

## Native Qualification

Install the optional emulator when an authorized ROM is available:

```bash
python -m pip install -e '.[gameboy]'
python -m pip install 'pyboy==2.7.0'  # version actually smoke-tested here
streambudget game capture --rom /absolute/path/owned.gb --boot-frames 1800 --out runs/capture-001
streambudget game manual --rom /absolute/path/owned.gb \
  --load-state runs/capture-001/environment.state --button a \
  --press-frames 4 --release-frames 24 --count 1 --out runs/manual-001
```

Inspect before/after images, release behavior and loading. Default boot is 120
emulated frames, not a promise of a useful screen. Record package versions/ROM hash.
Manual qualification is not autonomous evidence. Manual attempts are now journaled
before dispatch; interruptions retain unresolved status and never retry. Completed
receipts include before/after PNG hashes, local frame numbers and a changed-pixel
bounding box. Changes may be passive animation, not a useful button effect.
Disclose any chosen starting state;
a fresh-game demo cannot silently start from a convenient late save.

Boot remains 120 frames by default for compatibility; `--boot-frames` permits
0..3600 on capture/manual/run, recorded in the environment descriptor. Loading an
operator state performs no extra boot ticks and starts a local segment counter at
zero; it is not a fresh-game provenance claim. For the supplied Red ROM, 120 frames
were blank, 600 showed the intro, and 1800 showed the title. Do not call a blank
early frame a perception failure or discard its source record.

Configure the local endpoint in `configs/interactive/pokemon-local.yaml`.
The metered template intentionally fails validation until current rates are supplied.
Use credential environment-variable references, never keys in YAML/source. Local
billing is loopback-only; tunnel self-hosted remote inference.

### One Qwen Server, Two Profiles

The existing local YAML selects `Qwen/Qwen3.8-27B-FP8` for both endpoint aliases
at the same loopback URL: `fast` serves extraction/actions with thinking disabled;
`plan` now serves planning/initial compilation with **thinking disabled** and
`reasoning_effort: none`, at the owner's request. Earlier 12-13 s measurements used
medium, not the briefly configured but unmeasured low profile. Both the model template
flag and request effort are explicit; roles and output contracts remain distinct
even when both use non-thinking inference. Decoding parameters otherwise stay unchanged.
The earlier low configuration and OCR deduplication passed **500 CPU tests** and both
no-key demos. The none configuration also passed 500 CPU tests. Native none
latency/quality remains unmeasured, but a frozen planner probe is recorded below.

`runs/async-plan-none001` replayed the retained first startup planner context and
image once, with the same prompt/output cap and sampling settings except none effort
and disabled thinking. No emulator action or returned query was executed. The call
completed and settled: 49 output tokens, 3.003 s response wall time, 0.431 s first
content, empty reasoning output and no requested memory search. It instructed the
actor to wait for richer scene context, whereas the historical medium plan selected
intro advancement. Speed did not establish equivalent behavior or gameplay progress.
Approximate generation rate after first token was 18.7 tokens/s; the historical
219/233-token medium plans gave 18.4/18.7 tokens/s using the same timing calculation.
These are short client-observed generation intervals, not a server throughput test.
These are request profiles, not separate resident models. The author's thinking/
non-thinking sampling parameters are explicit, with seed 0 for this trial profile.
The actor remains one categorical ID with strict local validation, not generated code.
The first serving qualification uses one L40S and the same host's CPU PyBoy,
avoiding internet screenshot transfer. The owner supplied hosted selection evidence
for other Qwen variants, not a 27B quality or speed guarantee.

Official metadata/cards and vLLM recipe inspected on 2026-10-04 (metadata only;
no weights downloaded to this Mac):

| Checkpoint | Pinned Revision | Published Weight Files | Deployment Implication |
|---|---|---:|---|
| Qwen3.8-27B | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | 55.56 GB / 51.75 GiB | BF16 weights alone exceed 2x24 GiB |
| Qwen3.8-27B-FP8 | `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a` | 30.87 GB / 28.75 GiB | One L40S loaded the weights using 28.51 GiB; serving qualification below |
| Qwen3.8-Flash-Next | `de4b8e4d43b917e7706784d8bb445c9af86a3540` | 360.00 GB / 335.28 GiB | Not a fully GPU-resident 2x4090 deployment |

The FP8 checkpoint is mixed precision: roughly 24.70B FP8 and 3.08B BF16
parameters, including unquantized components. Weight-file size is not peak VRAM:
vision workspaces, hybrid recurrent/KV state, activations and server graphs need
headroom, and tensor-parallel placement need not be perfectly balanced. Flash-Next's
6B active language path does not imply 6B resident weights; its card lists 125B LM,
51B n-gram embeddings and 4B MTP. CPU/expert offloading is a separate latency/
correctness experiment, not the first deployment. Hosted `qwen/qwen3.8-flash` is
the provider's production variant, not an immutable Flash-Next snapshot guarantee.

Use the existing launcher on the **GPU host**, not this Mac. Pin the installed
vLLM/Transformers/CUDA/driver versions in the run receipt before launch. This is
a bounded starting configuration for one L40S; do not add MTP, 262k context or CPU
offload by default. The PyTorch Vast template is sufficient: use an isolated Python
environment, not Docker-in-Docker, and do not modify the host driver. First deployment
pins vLLM 0.23.0+cu129, torch 2.11.0+cu129 and Transformers 5.10.4; driver 575.57.08.

```bash
MODEL_ID=Qwen/Qwen3.8-27B-FP8 \
MODEL_REVISION=017b9c7af6b5689d5dd426a76e0bc077eb5ca20a \
bash scripts/serve_vllm.sh \
  --served-model-name Qwen/Qwen3.8-27B-FP8 \
  --tensor-parallel-size 1 --max-model-len 8192 --max-num-seqs 2 \
  --gpu-memory-utilization 0.90 --reasoning-parser qwen3 \
  --no-enable-prefix-caching --mm-processor-cache-gb 2 \
  --limit-mm-per-prompt '{"image":3}' \
  --mm-processor-kwargs '{"max_pixels":262144}'
```

Bind only to loopback and SSH-tunnel it for remote diagnostics. Pokemon/PyBoy is
CPU-only and now runs on the inference host; no simulation GPU is required. Start
with one agent/episode, not another batch-capacity project. The reduced 16k-character/
three-image client caps are not an exact 8k token guarantee; overlength requests
must fail explicitly rather than dropping essential task/schema/source evidence.

Self-hosting removes per-request provider/cache-read fees, not GPU rental, idle time
or prefill/vision work. Prefix caching still recomputes changed tails; roles, schemas,
thinking profiles and images can change cache keys. Processor caching is not proof
of GPU vision-encoder reuse. Measure cache hits/TTFT and end-to-end p50/p95 latency
on repeated and changed frames; never infer KV reuse just from faster HTTP or
report client wall time as GPU time. The engine warns that hybrid/Mamba prefix
caching is experimental and that some L40S FP8 shapes lack tuned kernel configs.
Neither warning alone establishes a failure or a speed gain. Dense-role fusion/text
reuse can wait for measurements.

The latency acceptance target is **under one second from capture to validated usable
action**, not merely first-token latency. Use the local-only, non-actuating command:

```bash
streambudget game latency --config configs/interactive/pokemon-local.yaml \
  --rom /private/owned.gb --boot-frames 1800 --count 20 --warmup 2 \
  --out runs/latency-title-001 --allow-network
```

It times capture, image/evidence preparation and a validated categorical action;
all warmup calls remain accounted for. The emulator does not advance and selected
buttons are never dispatched, so this is a cache-friendly frozen-screen condition,
not a full extractor/planner/actor cycle or autonomous success. Failed requests stop
without retry and retain their ledger. Use fresh paths for separately labelled
conditions, then check a short native loop with changing observations. Record warm
p50/p95, the fraction below one second and failures. Short output does not imply
short prefill, vision or reasoning time; speculation is not the first optimization.

### First L40S Measurements

One L40S (46,068 MiB usable), host-local CPU PyBoy, native Red pixels, FP8 and
non-thinking categorical actions. No hosted API calls or additional models.
The initial server used prefix caching, 8k context, two sequence slots and the
launch limits above. Startup compilation/warmup is separate from these measurements.

| Condition | Measured Calls | Warm p50 | Warm p95 | Under 1 s |
|---|---:|---:|---:|---:|
| Frozen title screen, capture through validated action | 20 + 2 accounted warmups | 0.516 s | 0.517 s | 20/20 |
| Frozen opening dialogue, same action interface | 20 + 2 accounted warmups | 0.506 s | 0.507 s | 20/20 |
| Frozen title, prefix caching disabled | 20 + 2 accounted warmups | 0.666 s | 0.668 s | 20/20 |

These are repeated-image conditions with empty world context; every result was
`A`, but no benchmark-selected button was executed. The first title request took
4.528 s and remains recorded as warmup. Server metrics for its 22 requests reported
16,464 / 28,644 prefix-token hits (57.5%); the usage API did not report cached-token
details. Do not interpret the ledger's zero cached-token sum as proof of zero reuse.
The server reserved about 41,219 MiB overall, including caches/workspaces, not just
the 28.51 GiB loaded weights. Small auxiliary models are not yet loaded or qualified.

A separately labelled **three-step native full-loop trial** started from the
operator-prepared opening-dialogue checkpoint, not a fresh autonomous game. It
completed seven calls and three `A` actions, advancing 84 emulated frames in
128.197 wall seconds. Reviewed source pixels moved from the welcome dialogue to
the professor introducing his name. This is opening-dialogue progress only, not
overworld, gym, or independent task-success qualification.

- Actor HTTP/validation time on changing observations: 0.677, 0.804, 0.995 s;
  these exclude capture/preparation and are not a full-loop sub-second guarantee.
- Extraction: 43.060, 22.691, 49.497 s for 787, 414, 908 output tokens.
- One thinking planner call: 10.125 s, 180 reported completion tokens.
- No failed/pending requests or uncertain action dispatches; three source images
  changed, but pixel change alone is not a semantic effect certificate.

Conclusion at that revision: small categorical output was fast enough in these warm conditions;
synchronous rich extraction was **not** a sub-second decision path.
The asynchronous change below removes rich extraction from each action's critical path;
matched quality gains and compact semantic output remain unproven. Do not buy a larger GPU,
switch engines, or expand the ontology merely to address lengthy extraction output.
The no-prefix condition used the same checkpoint, image, sampler, context and output
interface after a separate server restart; it also returned `A` for every request.
It remained sub-second without cross-request KV reuse, so the running service and
recommended launch now disable experimental prefix caching. CPU multimodal processor
caching remains enabled; this is not an entirely cache-free vision path. There were
73 accounted local model calls across the three benchmarks and native loop, no paid
API requests, and no request failures/retries. Startup included a readiness timeout
while compilation progressed; no model request was submitted by that health check.
Only one request was active at a time; `max_num_seqs=2` is an admission ceiling.
Changing-observation full-loop sub-second behavior, larger contexts, cache correctness,
tail latency under System 2 contention and sustained gameplay remain unqualified.
Raw receipts, native PNGs and state checkpoints stay under ignored
`runs/pixel-l40s-*`; only these aggregate results and code are public.

Sources:
- https://huggingface.co/Qwen/Qwen3.8-27B
- https://huggingface.co/Qwen/Qwen3.8-27B-FP8
- https://huggingface.co/Qwen/Qwen3.8-Flash-Next
- https://recipes.vllm.ai/Qwen/Qwen3.8-27B
- https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.html

### Asynchronous OCR And Memory (2026-10-04)

`async_perception: true` makes the actor use the latest image, last committed world
state and available OCR tracks without awaiting rich extraction or planning. One
background Qwen request and one OCR request may be active; observations are coalesced
to the latest retained frame rather than queued indefinitely. Coalescing counts are
reported. Raw boundary frames remain retained, but transient semantic events can be
missed; this is not full stream coverage or real-time gameplay.

Cold start gives semantic extraction the first background turn; subsequent planning
and due semantic refreshes alternate when both need service. Text-track events and
`semantic_refresh_steps` gate memory refresh. This is a simple baseline scheduler,
not learned relevance detection. Long generation can still contend with the actor
on the same GPU; asynchronous dispatch does not guarantee priority/preemption.

Only HTTP/inference runs in workers. Admission, budgets, SQLite writes, world updates
and action dispatch remain on the owner thread. All admitted work is settled before
a clean checkpoint; shutdown drain time is reported separately. Failures are retained,
not retried. A failed background result prevents a clean checkpoint. No new jobs are
launched during shutdown. Clean resume also checks the OCR model/config fingerprint.

World observations retain their original frame/time and all consumed parents. Delayed
interpretations enrich history; facts, mentions, current place and relations project
by source-frame order instead of arrival order. Historical dialogue cannot replace
the active dialogue cache. Async plans require matching source pixels, unchanged intent
and bounded step age before adoption; stale plans are discarded, never used to stop a
newer episode. These conservative checks can discard useful plans and need longer-run
qualification. Future visual/semantic applicability checks must not weaken dispatch fences.

OCR uses RapidOCR 3.9.2 / ONNXRuntime 1.24.4, CPU, two intra-op threads, fourfold
nearest-neighbor scaling and a two-native-pixel white border on recognition crops.
Detection/source boxes are not expanded. Explicit local ONNX files are required; the runtime does
not provision weights. All weights stay on the rented host. Install `.[ocr]` there
and set `ocr_config_path` to a host-local parameter YAML, for example:

```yaml
Det.model_path: /workspace/streambudget-ocr/PP-OCRv6_det_small.onnx
Det.ocr_version: PP-OCRv6
Det.model_type: small
Det.lang_type: multi
Rec.model_path: /workspace/streambudget-ocr/PP-OCRv6_rec_small.onnx
Rec.ocr_version: PP-OCRv6
Rec.model_type: small
Rec.lang_type: ch
Cls.model_path: /workspace/streambudget-ocr/ch_ppocr_mobile_v2.0_cls_mobile.onnx
```

`ch` is RapidOCR's recognition enum for this multilingual checkpoint, not a claim
that its output is Chinese-only. The orientation model is provisioned because the
adapter initializes it, but upright-screen inference skips orientation classification.
The checked ONNX manifests pin model SHA-256 hashes. Doctor validates local files
without loading models or making requests.

Whole-frame identity/refresh gating avoids unchanged rereads. Changed frames run text
detection; byte-identical crops reuse recognition for up to 60 seconds. Cache reuse
does not merge identities: two boxes carrying identical text remain separate sightings.
Geometry/text association and confidently estimated vertical scrolling preserve tracks
where supported. Arbitrary perspective motion, reflow and semantic re-identification
remain unqualified. Only tracks detected in the source frame enter hot context;
old OCR carries source time/hash and an explicit freshness flag. Missing OCR is not
proof that a screen contains no text.

Bounded frozen-screen comparison: five unique native title/intro frames, three passes
per model, with crop-recognition caches cleared before every sample. First sample
per model is retained as warmup, not silently removed from accounting.

| CPU OCR | Warm Median | Warm Maximum | Qualification |
|---|---:|---:|---|
| PP-OCRv6 small | 0.511 s | 0.542 s | Some clean dialogue; pixel-font/case errors and false detections |
| PP-OCRv6 medium | 7.428 s | 9.743 s | Better on some text, but still case/punctuation errors and false detections |

Small is the selected CPU starting point, not a proven accuracy winner. The sample
is not enough for aggregate character-error rates or multilingual claims. There were
30 completed OCR attempts plus one retained configuration-stage failure before model
inference. An earlier parameter-type setup failure is recorded separately; fixing
setup did not retry an uncertain action. No paid API calls. Private receipts/configs/
model manifests are in `runs/pixel-l40s-async-20261004/` (weights are remote only).

Initial native async trial `pixel-l40s-dialogue-async001`: 12 actions in 16.538 s,
including 1.699 s shutdown drain, 24 accounted local calls, 11 OCR submissions, one
whole-frame reuse and one crop-recognition hit. The planner occupied the background
slot throughout; no rich extraction was submitted. Its shutdown result was retained
but not activated, so this trial does not qualify semantic memory or planning.

The longer `pixel-l40s-dialogue-async002` stopped after 16 actions / 18.205 s when
the planner used a frame evidence ID as an entity focus ID. All 31 local attempts
settled; no uncertain actions or automatic retry. HTTP/JSON accounting reports zero
request failures, but the runtime result is **failed** due to contextual validation.
The focus schema now enumerates offered entity handles, explicitly requiring an empty
list when none exist. This failed trial remains in the evidence set.

Corrected, semantic-first `pixel-l40s-dialogue-async003`: 32 completed actions
(29 A, three WAIT), 956 emulated frames, 65 local calls (32 actor, 32 OCR, one rich
extractor). Action-loop wall time was approximately 33.316 s; total clean-stop time
was 55.211 s including 21.895 s background drain. Retained-frame decision preparation
through fresh-source validation measured approximately 1.02 s median / 1.08 s p95
(0.867-1.340 s range), excluding native capture and dispatch/journal/execution. This
does **not** establish the requested sub-second screenshot-to-action target.

The rich extractor took 54.961 s and committed its source-frame-zero interpretation
at shutdown; actors did not yet consume that memory. It retained two source-linked
entities and unknown-speaker dialogue. OCR had seven crop-recognition hits; every
changed observation still required detection. All calls/actions settled, no pending
holds or uncertain actions; no planner request was made in this semantic-first trial.
Initial/final pixels show intro dialogue progressing to a player-character transition,
not overworld or independently verified game success. The 12/16/32-action conditions
are development runs with different scheduler revisions, **not** matched quality
ablations. The rich memory writer remains slow, and its first commit was late; compact
semantic updates, steady-state memory use and applicable async plans are next to qualify.

All three native run directories, including the failed condition, plus OCR receipts
and weight hashes are backed up locally without model weights. Source PNGs, ROM,
checkpoints and model traces remain private/ignored. Qwen's inference service remains
running; no agent loop continues automatically after these bounded trials.

Official OCR sources:
- https://github.com/RapidAI/RapidOCR/tree/v3.9.2/python/rapidocr
- https://github.com/RapidAI/RapidOCR/blob/v3.9.2/python/rapidocr/default_models.yaml
- https://github.com/PaddlePaddle/PaddleOCR

### OCR Diagnostic And Correction (2026-10-04)

Frozen-pixel diagnostic: 20 human-transcribed visible line crops, including incomplete
typewriter strings, and ten full intro screens. Labels/crop coordinates are private
evaluator inputs only; no labels, game memory, font tables or routes enter control.
The detector stayed PP-OCRv6 small while recognizers/scaling/padding varied. Exact
match below normalizes whitespace only; character error preserves case and accents.
These are best settings selected on this small development sample, not held-out accuracy.

| Recognition-Only Condition | Exact Lines | Character Error | Warm Median |
|---|---:|---:|---:|
| PP-OCRv6 small, 2x nearest, padding 2 | 13/20 | 3.8% | 0.029 s |
| English PP-OCRv5 mobile, 2x nearest, padding 2 | 10/20 | 7.9% | 0.024 s |
| Tesseract 5.3.4 English, single-line, 4x bicubic, padding 2 | 11/20 | 6.3% | 0.114 s |

English v5 is not a demonstrated upgrade. Tesseract's whole-frame sparse-text mode
was faster (about 0.17 s) but produced false sprite/border text and incorrect order;
it is not integrated or an automatic fallback. Tesseract used Ubuntu's English
traineddata, with version/hash retained. Its official guidance motivated borders,
single-line segmentation and rescaling, not a game-specific dictionary:
- https://tesseract-ocr.github.io/tessdoc/ImproveQuality.html
- https://github.com/tesseract-ocr/tessdata_fast

The actionable bug was reading order: sorting each box by its top edge placed shorter
words after words to their right. Geometry-only row ordering reduced full-screen
character error from **16.4% to 3.8%** without another inference; padded recognition
reduced it to **3.1%**, with five of ten screens exactly transcribed. Rows use fixed
anchors to avoid chaining neighboring lines. Individual boxes/identities remain
separate; no words are merged, spell-corrected or completed by a dictionary. This is
an upright-row heuristic, not general layout or learned video-text tracking.

Selected runtime remains v6 small with 4x detector input, padded recognition and row
ordering. The best 2x manual-crop result does not qualify a 2x full-frame detector.
Three full-screen replay passes reproduced five exact screens per pass; warm median
**0.502 s**, maximum **0.522 s**, with crop caching cleared throughout. One accounted
cold full-pipeline warmup is excluded from warm latency. Preprocessing/order now enter
the OCR fingerprint, so old incompatible OCR checkpoints fail clean resume validation.

Private receipts/scripts/ledgers: `runs/pixel-ocr-diagnostic-20261004/`. All **488**
local OCR requests settled, including a retained 162-request draft with incorrect
evaluator frame indexing, excluded from quality comparisons. The corrected pass has
an explicit lowercase-label revision and whitespace-normalized rescoring of retained
outputs, not unlogged reruns. No paid calls or emulator actions; weights remain only
on the rented host. This sample covers intro dialogue, not menus, other fonts or
languages. Punctuation/accent/short-string errors remain; OCR is supporting evidence,
not authoritative state. Steady-state memory/planning use remains the next blocker.

### Additional OCR Candidates (2026-10-04)

The same frozen 20 visible line crops and ten intro screens were replayed, with no
game actions, API calls, dictionaries or LLM cleanup. Crops/labels are evaluator-only;
recognizers receive pixels and their official recognition prompt. Whitespace alone
is normalized for strict scoring; case-insensitive results below still retain accents.
These are development diagnostics, not independent held-out or general OCR scores.

| Candidate / Line Input | Device | Exact Lines | Character Error | Warm Median |
|---|---|---:|---:|---:|
| EasyOCR Latin-g2, 2x nearest + padding 2 | CPU | 3/20 | 15.1% | 0.051 s |
| TrOCR small printed, 2x nearest + padding 2 | CPU | 0/20 | 64.9% | 0.296 s |
| GLM-OCR, 2x nearest + padding 2 | L40S | 13/20 | 2.9% | 0.088 s |
| Xiaomi-OCR-0, 4x nearest + padding 2 | L40S | 16/20 | 1.7% | 0.197 s |
| Nemotron OCR v2 English, 4x nearest + padding 2 | L40S | 4/20 | 41.0% | 0.019 s |
| Nemotron OCR v2 multilingual, 2x nearest + padding 2 | L40S | 15/20 | 6.3% | 0.020 s |

Nemotron runs its full detector/recognizer/relational pipeline even on manual crops;
the other line conditions are recognition-only. Line latency is not a matched
recognizer efficiency comparison. Nemotron uses its official default 1024-pixel
detector resolution, with one image and 16-region recognition/relational chunks.

TrOCR's strict result mainly reflects uppercase output: case-insensitive scoring is
10/20 exact and 4.6% character error. It also completes an incomplete `People cal`
as `PEOPLE CALL` and reads `POKéMON!` as `POK&MON!`; it is not a demonstrated upgrade.
EasyOCR confuses ordinary letters/punctuation (`as a profession.` becomes
`as 9 Profession`). PARSeq was inspected but not run: its standard word-level
94-character checkpoint excludes spaces/accents and needs another segmentation step.

| Whole-Screen Candidate, 4x Nearest | Strict Exact | Character Error | Case-Insensitive Exact | Warm Median / Maximum |
|---|---:|---:|---:|---:|
| GLM-OCR | 5/10 | 2.7% | 9/10 | 0.176 / 0.207 s |
| Xiaomi-OCR-0 | 5/10 | 2.7% | 5/10 | 0.313 / 0.335 s |
| Nemotron OCR v2 English | 1/10 | 45.8% | 1/10 | 0.020 / 0.022 s |
| Nemotron OCR v2 multilingual | 2/10 | 11.1% | 2/10 | 0.022 / 0.024 s |

GLM-OCR is document-focused (text, tables and formulas), not a video-text tracker.
It nevertheless reads these screens well: most remaining errors are `é` becoming
`É`; one output also transcribes the rendered advance arrow as `▼`. Xiaomi likewise
transcribes the arrow and drops the accent (`POKEMON`). Native whole-screen Xiaomi
produces identical normalized text at 0.295 s median; GLM native reaches 4/10 exact
at 0.147 s. GPU-versus-CPU speed is deployment latency, not hardware-matched efficiency.

Native full-screen Nemotron performs better than its 4x input: English reaches 2/10
exact with 13.4% character error; multilingual reaches 4/10 exact with 3.8% error,
at **0.021 s median / 0.021 s maximum**. English splits `Welcome` into `We I come`;
multilingual sometimes duplicates an `i`, adds punctuation or loses the accent.
One bounded 640-pixel detector-resolution check to avoid fractional resize on 4x
screens reaches 4/10 exact, but 5.3% character error (0.018 s median), so it is not
an accuracy improvement. At native input that check misses the entire OAK line and
has 9.5% character error despite the same 4/10 exact count. Counts alone hide severity.

Unlike GLM/Xiaomi, Nemotron is designed for natural scene text as well as documents
and returns region geometry/confidence. Its multilingual variant is a promising
low-latency candidate, not yet a qualified runtime replacement: temporal identities,
box conventions, partial strings, false detections and Qwen contention need checks.
Among these candidates GLM is the stronger whole-screen text candidate; Xiaomi wins strict
cropped-line exactness but is slower. No claim about generic scene/document rankings
or held-out gameplay follows from these intro screens.

GLM used Transformers BF16/SDPA and `Text Recognition:`; Xiaomi used BF16/SDPA and
`Extract the text in the image.` with greedy bounded generation, no output truncation.
Xiaomi used the installed Transformers 5.10.4 rather than the card's newer example
stack; it warned that fused linear-attention/convolution libraries were absent and
used the PyTorch fallback. Measured loaded parameters are 1,107,405,824 for GLM and
852,985,920 for Xiaomi; peak allocated/reserved memory is 2.20/2.29 GiB and
1.69/1.78 GiB respectively. Both fit alongside resident Qwen without evicting it;
simultaneous Qwen-inference contention is unqualified. Whole-screen generated text
has no region boxes/confidence and is not a drop-in replacement for tracked OCR.
At the time of this diagnostic, the runtime still used the PP-OCRv6 small adapter;
the actor-first integration above subsequently selected Hunyuan/GLM.

Nemotron's official C++/CUDA extension built against torch 2.11.0+cu129 using the
installed CUDA 12.8 toolkit, Python 3.12 and native L40S sm_89 kernels; no framework,
driver or service changes were needed. English setup, including downloads/build,
took 420 s; multilingual reused the extension. Its initializer also downloaded a
torchvision RegNet checkpoint, then strictly loaded the pinned OCR detector state;
initializer and OCR weight hashes are retained. Measured parameters after classifier
alignment are 53,735,702 English and 83,757,910 multilingual. Default peak
allocated/reserved memory is 1.43/1.46 GiB English and 1.54/1.57 GiB multilingual;
the 640-resolution check is 0.56/0.65 GiB. GPU memory returned to Qwen's unchanged
41,741 MiB after the candidate processes exited. Warm latency includes image
preprocessing/inference, excludes ledger settlement and is not live loop latency.

Pinned sources:
- https://github.com/baudm/parseq/tree/1902db043c029a7e03a3818c616c06600af574be
- https://github.com/JaidedAI/EasyOCR/tree/v1.7.2 (Latin-g2 official checksum verified)
- https://huggingface.co/microsoft/trocr-small-printed/tree/04e994ab854b0089d4929f48c2b4dbe2ce78a340
- https://huggingface.co/zai-org/GLM-OCR/tree/2e85a62840ccac27daa451df36c736c4636b8628
- https://huggingface.co/SeerRay-Lab/Xiaomi-OCR-0/tree/e4d1c4a6804bd9ef342b93d705a73af003e2ef4e
- https://huggingface.co/nvidia/nemotron-ocr-v2/tree/0e83e83f17943524b90afa6c0fd82ac2bc1a40ca (NVIDIA Open Model License; implementation Apache-2.0)

Private manifests, setup logs, scored outputs and SQLite ledgers remain in
`runs/pixel-ocr-diagnostic-20261004/extended/`; checkpoints remain only on the GPU
host. EasyOCR/TrOCR each settled 42 requests; GLM settled 62 plus one retained failed
setup (`accelerate` absent with `device_map`, resolved by CPU load then GPU transfer);
Xiaomi settled 62; each Nemotron variant and the 640-resolution check settled 62.
That is 395 additional settled requests, including one failure,
separate from the prior 488. TrOCR's encoder-pooler missing-weight warning
and GLM's axial-RoPE validation warning are retained in the experiment record.
Working-tree CPU regression: 413 tests passed with the two existing Starlette/anyio
warnings; both no-key demos, targeted interactive Ruff and whitespace checks passed.
No runtime source changed for this candidate screen.

#### HunyuanOCR-1.5 Follow-Up

The linked `tencent/HunyuanOCR` repository now hosts **1.5 at the root**, not the
archived 1.0. Tested revision: `47644ecc4fc854efa4f505155158831f36773ee4`.
Transformers 5.13.0 was installed without dependencies into a candidate-only package
directory and prepended only in the test process; Qwen retains Transformers 5.10.4,
torch 2.11.0+cu129 and its existing service. No vLLM upgrade, CUDA 13 installation,
DFlash draft, archived weights, driver change or remote-code loading was needed.

The first generic SDPA/torchvision path completed but was **unqualified**: 0/10 exact
at native and 4x input, mostly unrelated text, and a warning that torchvision 0.26
substituted bicubic for the configured Lanczos resize. The corrected official-style
path uses the PIL processor, empty system message, explicit template/text/image
processing, BF16 and greedy decoding with repetition penalty 1.08. Its prompt asks
to extract image text (the card's Chinese transcription prompt), not document parsing.
No cleanup, dictionary, prediction labels or game context enter inference.

| Hunyuan 1.5 / PIL Condition | Exact Screens | Character Error | Warm Median / Maximum |
|---|---:|---:|---:|
| Native 160x144, eager attention | **8/10** | **1.5%** | 0.253 / 0.325 s |
| Native 160x144, SDPA | **8/10** | **1.5%** | 0.216 / 0.278 s |
| 4x nearest, eager attention | 3/10 | 19.1% | 0.298 / 0.339 s |
| 4x nearest, SDPA | 4/10 | 17.9% | 0.231 / 0.267 s |

Both native conditions produce identical normalized text. All words/case/accents
match: the two exact-match failures are the rendered advance arrows transcribed as
`▼`, excluded by the dialogue-only evaluator labels. Both also transcribe a generated
ordinary-font printed control exactly. Native input is still internally resized by
the model processor; do not interpret this as inference without resizing. The first
bad condition changed multiple processing/wrapping settings, so its failure cannot
be attributed solely to SDPA or solely to the interpolation fallback. The follow-up
holds the PIL path fixed and establishes that SDPA works for these native screens.

Narrow line crops are not ready for a direct plain-text adapter: eager/PIL reaches
only 1/20 strict exact at 2x (84.1% character error), mainly because it adds a Chinese
"text in the image is:" preamble to 18 outputs. A separately labeled post-hoc fixed
preamble removal reaches 16/20 exact, but still 8.8% character error; the remaining
errors include spacing, lost punctuation, and a single `I` represented as a LaTeX
boxed Roman numeral. At 4x all 20 crops have the preamble (0/20 raw exact; 16/20 after
fixed removal). These are formatting diagnostics, not silently corrected benchmark
scores or a runtime parser. Full-screen 4x also introduces casing/spacing/preamble
errors. The pixel-font native whole-screen result does not qualify menus/other fonts.

Measured loaded parameters: 996,208,112. Peak allocated/reserved memory is
2.22/2.64 GiB eager and 1.97/2.11 GiB SDPA. Both fit alongside Qwen; after all tests
GPU use returned to the unchanged 41,741 MiB. Concurrent Qwen inference, actual
asynchronous loop latency and region tracking remain unqualified. This tested plain
transcription mode provides no geometry/confidence; the model's text-spotting modes
were not tested. Hunyuan is now the strongest strict native whole-screen text result
in this small development sample, while Nemotron multilingual remains much faster.
At the time of this diagnostic, the runtime had not switched from PP-OCRv6 small;
the actor-first integration above subsequently selected Hunyuan/GLM.

Private runs `hunyuan`, `hunyuan002` and `hunyuan003` retain 62, 63 and 23 settled
requests respectively (148 additional, no API calls or emulator actions), including
both printed controls, all bad outputs and setup/package manifests. That brings the
extended diagnostic to 543 settled requests (one earlier setup failure), separate
from the initial 488. The 128-token generation cap was not reached. Weights stay on
the Japan GPU host; only receipts/scripts/control images are backed up on the Mac.

License is **Tencent Hunyuan Community**, not MIT/Apache: its terms exclude the EU,
UK and South Korea and restrict using outputs to improve other AI models. The Japan
evaluation does not establish permission for a worldwide product or weight redistribution.
Official references:
- https://huggingface.co/tencent/HunyuanOCR/tree/47644ecc4fc854efa4f505155158831f36773ee4
- https://huggingface.co/tencent/HunyuanOCR/blob/47644ecc4fc854efa4f505155158831f36773ee4/LICENSE
- https://github.com/Tencent-Hunyuan/HunyuanOCR/blob/1ef4179e53cf860f7e6fd8276a292e5d05b3e927/inference/transformers/infer_hf_8gpu.py
- https://github.com/Tencent-Hunyuan/HunyuanOCR/blob/1ef4179e53cf860f7e6fd8276a292e5d05b3e927/docs/inference/archive/transformers.md

```bash
streambudget game probe --config /path/to/local-config.yaml \
  --image runs/capture-001/frame.png --role extract \
  --out runs/probe-extract-001 --allow-network
streambudget game probe --config /path/to/local-config.yaml \
  --image runs/manual-001/after-000.png --role act \
  --intent "Advance the currently visible dialogue once." \
  --out runs/probe-action-001 --allow-network
```

Probes never actuate. Review grounding, not just JSON/HTTP success. Probes and the
live loop now share exactly the same nearest-neighbor pixel scaling; source evidence
is unchanged. Qualify truncation, limits, thinking settings and latency. The metered
template targets `openai/gpt-6.1-sol` on OpenRouter Chat Completions, with medium
reasoning and explicit Flex. It is intentionally invalid until current tier-specific
input/output rates are supplied. Credentials use `OPENROUTER_API_KEY`, never a key
in YAML or source. Doctor checks key-variable presence without printing its value;
its readiness flag is static configuration evidence, not endpoint qualification.

Typed `openrouter` configuration pins provider tags, prohibits fallback, requires
parameter support and sets per-million price ceilings. It cannot override messages,
actions or tools. The adapter maps medium to `reasoning: {effort: medium}`; direct
OpenAI-compatible endpoints retain the `reasoning_effort` wire field. OpenRouter
responses with an unexpected model or unconfirmed explicit tier stop before action.

The configured 900-second timeout accommodates Flex latency. The selected route's
`max_tokens` includes hidden reasoning as well as visible JSON: the action cap is 2048, not the
local-template 64. This is a preparation choice, not a measured sufficient cap.
There is no automatic retry, tier fallback or model substitution. A separately reviewed
standard trial must explicitly set `service_tier: default` and its own rates/budget;
switching tiers is not an allowed same-run resume configuration edit.

Requested and returned tiers can differ. When an explicit tier is configured, missing
or mismatching returned tiers leave the charge unknown and its reservation held; do
not price standard service at Flex rates. Capacity failures also remain held until
review, rather than guessing that an arbitrary provider's HTTP 429 is free. Reported
token totals include reasoning/cached tokens when present and flag incomplete usage.
An initial live synthetic-pixel qualification is recorded below; native Pokemon
competence and model ranking remain unqualified.

Official OpenAI documentation fetched on 2026-10-04:
- https://developers.openai.com/api/docs/models/gpt-6.1-sol (model ID and medium effort)
- https://developers.openai.com/api/docs/guides/flex-processing (explicit Flex request)
- https://developers.openai.com/api/docs/api-reference/chat/create (returned service tier)

OpenRouter sources fetched on 2026-10-04:
- https://openrouter.ai/docs/guides/routing/provider-selection (provider restrictions and price ceilings)
- https://openrouter.ai/api/v1/models/openai/gpt-6.1-sol/endpoints (selected Flex route)

The observed Flex catalog lists $1/M prompt, $5/M completion, $0.05/M cached read
and $1.25/M cache write, with higher long-context prices at 272,000 prompt tokens.
These are a dated catalog observation, not guaranteed future prices. No search,
tools or plugins are enabled. Refresh rates, bounds and availability before a run.

## Loop, Accounting And Artifacts

Set `max_steps: 3` initially; defaults are ten steps and forty total attempts.

```bash
streambudget game run --config /path/to/local-config.yaml \
  --rom /absolute/path/owned.gb --out runs/game-001 --allow-network
streambudget game report --run runs/game-001
streambudget game export --run runs/game-001 --output runs/game-001/transitions.jsonl
```

Metered requests also require `--allow-paid`. All real requests, including localhost,
need network opt-in. Compilation/extraction/planning/retrieval/action requests share
the run-wide attempt cap. Reserve before calls; unknown usage stays held. Malformed
responses can cost money. No implicit retries, fallback models or scored resets.
Each probe has a fresh local ledger, not free shared campaign capacity. Failed
probes retain accounting in their report, including interrupted/unknown holds.
Configured reservations are admission control, not a provider invoice guarantee.

`known_provider_usd` uses configured-rate token usage for ordinary endpoints and
the observed `usage.cost` invoice for confirmed OpenRouter model/tier responses;
missing or invalid invoices retain holds rather than guessing cache-write cost. Local API
billing is zero, but rental/energy/storage cost is unmeasured, not free. Never reset
or reuse another project's ledger implicitly. Use provider-side spend controls. The
owner-authorized live qualification below also reserves in the existing shared
paid ledger under its $95 ceiling and $3 trial cap, serializing against other
campaigns with their existing lock. Historical uncertain charges are not released
or retried. Generic CLI run ledgers do not enforce that private workspace-wide cap;
use the shared admission guard for further owner-paid experiments.

Each run has `run.json`, `memory.sqlite`, immutable `media/`, `trace.jsonl`, a clean
checkpoint/state and `report.html`. Model inputs/responses, frames, dialogue and ROM
states are sensitive. Runs/ROMs/checkpoints are ignored by Git. Do not publish raw
traces casually. `goal_claimed` is for review; `game_success` remains null without an
independent grader. Exported actions/teacher replies are unreviewed, not correct labels.

## Resume

Only clean step/wall/budget/blocked/goal-claim stops may resume. ROM, checkpoint bytes,
rendered pixels, immutable configuration, schema, action counts and ledger counts must
agree. Pending inference, uncertain actions, failures or post-checkpoint edits block
automatic continuation. `--resume` cannot combine with arbitrary `--load-state`.

```bash
streambudget game run --config /path/to/local-config.yaml \
  --rom /absolute/path/owned.gb --out runs/game-001 --resume --allow-network
```

Only total step/call/estimated-budget and invocation wall caps may change explicitly.
Counts/spend remain cumulative; total elapsed time is retained. This is not crash-safe
exactly-once physical control or schema migration. Do not weaken fences for a demo.
Clean checkpoints now also bind model inputs/calls and action receipts with a streaming
digest, detecting same-count edits, and bind the retained current frame and emulator
version/window. Older pre-digest checkpoints are review-only, not silently migrated.
Historical reports remain readable. Digests detect inconsistency, not malicious edits
to all data/manifests by an attacker with write access.

## First Live Trial And Evaluation

Use the metered template's three-decision ceiling, twelve-attempt ceiling, compilation
disabled and 4-press/4-release-frame actions. Do not increase limits just to conceal a
failed first trial. Run extraction/action probes first, then inspect every actual
transition in the short loop. All four roles initially use the same model. No GPU is
needed for PyBoy or a hosted endpoint; a GPU is only needed for local model serving.

Preflight: authorized ROM and its digest; tested PyBoy version; visibly useful native
screen; manually calibrated short button actions; configured image/JSON endpoint;
current Flex rates/reservation and credential presence; explicit network/paid opt-in;
fresh private output directory. Do not borrow another project's budget ledger.

Independent review occurs after execution, outside model context. Record each reviewed
decision/milestone with before/after evidence IDs, expected visible effect, actual
visible effect, reviewer, and `passed`, `failed` or `unknown`. Category examples:
grounding/OCR, button timing, wrong local action, stale belief/identity, missing planner
intent, malformed output, transport/capacity, budget stop and uncertain execution.
Do not reclassify a transport failure as a model-intelligence failure.

First useful milestones are a correctly advanced dialogue/menu and a visibly reached
local interaction target; later, retain useful state through a revisit and verify
Gym progression from source frames. These are review criteria, not a route or hidden
game-state grader. An ambiguous screen stays unknown. A model's claimed goal completion
is neither a badge nor independently verified success.

Read-only diagnostics and paired-condition checks:

```bash
streambudget game metrics --run runs/game-001
streambudget game compare --runs runs/recent-001 runs/world-001
```

Metrics report attempts (including unresolved ones), executed-frame receipts, retained
frames, unchanged-pixel transitions, per-role attempts/median latency, reported tokens,
images, costs/holds and wall time. Unchanged pixels are not automatically a semantic
stall: they can be dialogue waiting, animation, or a harmless action. True stalls,
wrong identity links and meaningful progress require independent frame review.

For a `recent`/`world` pair, copy the same configuration and change only `baseline`.
Use the same ROM, emulator, disclosed starting state/screen, schema, action durations,
model/effort/tier, planner/retrieval settings and total caps. Start fresh independent
runs; never replay a successful condition's chosen actions into the other. The compare
command refuses mismatched settings, initial screenshot hashes, native provenance or
resume histories. It includes failed stopped runs but does not select a winner or
compute success rates. `recent` hides the persistent world projection while retaining
short text/image history and lexical retrieval: it is not a memoryless/direct-one-call
baseline, and inference role counts remain comparable, not necessarily identical.

First use one matched pair as an integration screen, not a benchmark. If there is a
promising difference, repeat from several independently disclosed starts, alternate
condition order and retain every attempt. Report milestones/progress and failure
categories alongside calls/images/tokens/source frames/wall time/known cost/unknown
holds. Report a success rate only when a common independent success criterion and
enough repeated episodes exist; do not infer it from one run or a model narrative.

## Validation

OCR correction on 2026-10-04 adds checks for padded input with unchanged source boxes,
row ordering without line bridging, distinct cached-region identities, bounded integer
preprocessing and preprocessing-sensitive checkpoint fingerprints. Validation: 413
working-tree tests and 331 isolated public-snapshot tests passed; both no-key demos,
Ruff, compileall and whitespace checks passed. The same two existing Starlette/anyio
warnings remain. Frozen native replay is an OCR diagnostic, not gameplay success.

Asynchronous integration on 2026-10-04 adds regressions for owner-thread admission,
in-flight budget enforcement, source-ordered late facts/place/relations/dialogue,
nonblocking extraction, stale-plan rejection, bounded/coalesced jobs, failed-work
drain, clean resume, OCR provenance/crop caching and bounded hot text. These software
tests do not establish OCR accuracy, perception coverage or autonomous gameplay.
Validation: 405 working-tree tests and 323 tests with imports explicitly routed to
the isolated staged public snapshot passed, with the same two pre-existing
Starlette/anyio warnings. Both game/watch no-key demos passed in both trees. Ruff,
compileall and Git whitespace checks passed. No unrelated dirty watch/research
changes or private run evidence are included in this integration commit.

The supplied archive reported 98 CPU tests, fake-device emulator checks and a localhost
socket exchange against a synthetic responder. Its packaging tests froze source blobs
and tested now-unneeded installers; they were replaced with live application contracts.
Historical source hashes establish provenance, not a bar to deliberate core fixes.

Local integration on 2026-10-04:
- Existing working-tree baseline: 221 CPU tests passed.
- Combined suite: 318 CPU tests passed (221 existing plus 97 interactive/integration).
- Clean staged public snapshot: 236 CPU tests passed (139 committed existing plus
  97 interactive/integration); the integration does not depend on preserved local edits.
- Both suites reported two existing Starlette/anyio deprecation warnings.
- Game fixture: 14 decisions, persistent evidence/world state and HTML report generated.
- Existing watch fixture: completed, synthetic QA 2/2 and alerts 3/3; not model scores.
- Artifact directories: `runs/pixel-integration-20261004-game/` and
  `runs/pixel-integration-20261004-watch/`.
- Ruff passed for interactive source/tests and the unified CLI; compileall and Git
  whitespace checks passed.
- Wheels built with `uv` from both the working tree and clean public snapshot.
  The public wheel installed into an isolated environment outside the repository;
  both doctor commands and both no-key demos passed without repository imports.
- Doctor made no network calls. PyBoy is not installed and the model ID remains an
  intentional placeholder, not an inference-ready configuration.

The selective import adds 26 files, not the archive's 81 entries. Duplicate guides,
overlay installers, frozen-blob checks and generated previews are not maintained.
No GPU, ROM download/execution, external model, benchmark scoring or training was used.
Native qualification and cost/quality advantage remain unverified.

Offline preparation after integration, 2026-10-04:
- PyBoy 2.7.0 installed; its bundled `default_rom.gb` rendered a nonblank 160x144
  `PyBoy / No ROM Found` screen at 120 boot frames. One manual A receipt advanced
  eight frames. The native regression saved/loaded identical pixels and reproduced
  the next bounded transition; no commercial ROM or useful game effect was tested.
- Native outputs: `runs/pixel-offline-20261004-native-capture/` and
  `runs/pixel-offline-20261004-native-manual/`. Bundled demo ROM SHA-256:
  `d2c2627aa7167d58e5e0b0072d9173714de146821f43a9f45102dded191987f9`.
- Failure regression coverage added for interrupted/uncertain press, boot failure,
  partial checkpoint writes, constructor/final-write cleanup, stale frame numbers,
  late actor results, compilation admission stops and same-count checkpoint edits.
- Real loopback HTTP regression verifies cumulative call caps/usage over clean
  budget-stop/resume; its responder is synthetic, not an LLM.
- Both 14-decision fixture conditions completed and passed matched-setting checks:
  `runs/pixel-offline-20261004-{recent,world}/`. Both used 34 synthetic role calls and
  112 advanced frames. This is software evidence, not hierarchy/performance benefit.
- Report checked in Chromium at 1440x900 and 375x812: before/after frames rendered,
  navigation updated receipts, no horizontal overflow. Missing after frames remain
  unknown; final records are explicit. Raw run artifacts remain private/ignored.
- Final working-tree suite: 355 tests passed (221 retained plus 134 interactive).
  Isolated staged public snapshot: 273 passed (139 committed retained plus 134
  interactive). Both reported the same two Starlette/anyio deprecation warnings.
- Ruff, compileall and whitespace checks passed. The public snapshot wheel built
  and installed in an isolated environment; doctor, game/watch demos and matched
  comparison commands passed outside repository imports. PyBoy remains optional
  and was not installed in that core-only wheel-check environment.
- No external model requests, paid calls, GPU provisioning, model weights, commercial
  ROM downloads or training were performed. The code/guide publication is tracked in
  Git on `codex/pixel-agent-integration` and https://github.com/AranKomat/streambudget/pull/1.

### Offline Checklist Status

| Item | Status |
|---|---|
| CPU emulator setup | Done for supplied Red ROM title/menu/dialogue and checkpoint; overworld controls still pending |
| Execution/resume audit | Fixes and failure regressions implemented; no stronger crash recovery claim |
| GPT-6.1 Sol medium/Flex preparation | OpenRouter live probes reviewed; one Flex capacity failure retained; separate standard toy loop completed |
| First short trial preparation | Live three-decision toy loop done; native Qwen loop awaits self-hosted endpoint, not a ROM |
| Evaluation definition | Source-review criteria and read-only metrics ready; independent gameplay grader absent |
| Matched comparison preparation | Configuration/provenance checks and fixture pair done; real comparison not run |
| Report usability | Before/after, source/action/model timing/usage and desktop/mobile checks done |
| Consolidation/publication | Existing guide/modules/tests only; verification above and publication in Git/PR #1 |

### First Live Model Qualification

2026-10-04, owner-selected OpenRouter `openai/gpt-6.1-sol`, medium. Credentials
were loaded in-process from the owner-specified private `.env`, never printed,
committed or included in request artifacts. No GPU or model weights were needed.

Input was the original 160x144 software fixture, shown to inference as a 480x432
nearest-neighbor PNG. This is deliberately simple synthetic pixel evidence, not
a Pokemon task, native controller qualification or proof of a memory advantage.
Only source frames, beliefs, intent and the fixed button manifest reached the model.

| Stage | Result | Requests | Known Cost | Unknown Hold |
|---|---|---:|---:|---:|
| Flex extract/action probes | Correct text and blue-character/gate grounding; RIGHT selected; no actuation | 2 | $0.00700725 | $0 |
| Flex closed loop | Extraction returned; planner received explicit provider-unavailable capacity error; zero actions, stopped without retry | 2 | $0.00253045 | $0.30 |
| Separately authorized standard closed loop | Three RIGHT actions; clean step-limit stop; all returned model/tier checks passed | 7 | $0.05550180 | $0 |

Totals: 11 attempts, ten validated responses, one capacity failure, $0.06503950
observed provider charges plus the retained $0.30 hold. All attempts are in the
existing workspace-wide ledger; shared spend plus historical/new holds was
$89.154523494300 of $95 after this qualification. Reread it before more paid work.
The standard trial used the owner's standing permission after explicit Flex
unavailability, a fresh synthetic episode and standard pricing, not a silent
same-run tier switch or automatic retry of an uncertain call. The failed Flex
episode remains failed and is not automatically resumable.

The standard loop used three extraction calls, one planner call and three actor
calls, nine image submissions, 20,011 reported prompt tokens, 1,472 completion
tokens and 3,848 cached tokens. Reported reasoning tokens were zero despite the
medium request; do not infer an unreported thinking process. Wall time was 38.86s,
with extraction consuming 26.66s, planning 3.18s and actors 8.92s. Emulation advanced
24 frames (0.4 nominal seconds); model latency is not source/game time. The episode
stopped after three decisions, not after task success.

Independent before/final pixel review confirms the blue rectangle moved from the
left toward the still-distant gate (left edge x=12 to x=48). No gate crossing,
conversation, room revisit, ambiguous identity, recovery or native gameplay was
tested. `game_success` remains null. The report distinguishes a synthetic environment
using a chat endpoint from a synthetic rule-based model; an endpoint alone is not
proof of a learned model, which was separately checked here through served identity.

Private, ignored evidence on this workstation:
- `runs/pixel-live-20261004-sol-flex-001/`: probes, failed loop, source image and wire receipts.
- `runs/pixel-live-20261004-sol-standard-001/loop/`: clean checkpoint, source images,
  per-role inputs/outputs, action receipts, usage and `report.html`.
- `runs/pixel-live-20261004.py`: private shared-budget guarded qualification driver,
  no embedded credentials. Do not rerun existing attempt IDs or reset these ledgers.

Production code/tests remain in existing modules; no public overlay/status files
were added. Probes now share runtime scaling, failure reports include charge holds,
doctor separates placeholder/missing-key setup from static readiness, OpenRouter
routing is typed and non-fallback, and billing uses its observed invoice rather than
incorrectly ignoring cache writes. Nonfinite/mismatched responses cannot actuate.

Working-tree CPU validation after these changes: 378 tests passed with the same
two retained Starlette/anyio deprecation warnings. Isolated staged public snapshot:
296 tests passed with those same warnings; both no-key demos passed. Ruff,
compileall and Git whitespace checks passed. A wheel built from that snapshot
installed outside the repository; static doctor and both installed no-key demos
passed without repository imports (PyBoy remained optional/not installed there).
Real native evaluation, recent/world
comparison, sustained progress and model/latency optimization remain undone.

### Native Red Controls And Qwen Direction

2026-10-04 continuation: owner supplied `Downloads/Pokemon - Red Version (USA,
Europe).gb`. Original ROM SHA-256:
`5ca7ba01642a3b27b0cc0b5349b52792795b62d3ed977e98a09390659af96b7b`.
PyBoy 2.7.0, null window, no model calls, GPU use, paid calls, downloads or ROM
redistribution. Original ROM bytes remained unchanged. ROM/state/screenshots stay
private and ignored; the public guide records only commands/hashes and observations.

| Check | Observed Result | Remaining Limit |
|---|---|---|
| Native boot | 120 frames blank; 600 Game Freak intro; 1800 Red title | These three checkpoints do not identify the earliest usable frame |
| START 4+4 | Immediate screen unchanged; menu appeared after 240 neutral WAIT frames | This is delayed observation, not proof a 4-frame press failed |
| START 24+24 | Immediate screen unchanged; menu appeared after 192 neutral WAIT frames | No demonstrated need for longer held START; waits were sampled, not exact latency |
| DOWN then UP, each 4+24 | Cursor moved NEW GAME -> OPTION -> NEW GAME | Menu motion only, not overworld movement |
| A on NEW GAME, 4+24 | White transition, then opening Professor dialogue after 192 neutral WAIT frames | Operator setup, not autonomous task progress |
| A at dialogue, 4+24 then WAIT48 | Visible text scrolled and completed the next dialogue page | Text completion/deduplication still needs agent-loop validation |
| Native checkpoint replay | Restored menu pixels matched; same DOWN transition reproduced byte-identical before/after PNGs | Only this state/action checked, not universal determinism |

Evidence directories: `runs/pixel-native-20261004-red-*`, including blank/intro
captures, both START cases, settling frames, menu DOWN/UP, the repeated DOWN
transition and opening dialogue. The latest operator checkpoint is
`runs/pixel-native-20261004-red-dialogue-settle48/environment.state`; using it
for a diagnostic must be labeled an operator-prepared start. A fresh autonomous
trial starts from the ROM boot/title, not silently from this dialogue checkpoint.

Manual receipts now retain dispatch/failure status, frame numbers, before/after PNG
hashes and changed-pixel bounds. UI animations can create those changes; independent
image review, not the bounding box, established the menu/text effects above. Default
4+4 actions were not silently redefined globally. The local Qwen profile uses 4+24
buttons and WAIT24+24 to expose settling explicitly; every model choice remains one
bounded categorical action, with no hidden wait-until-ready script.

Native control qualification is **partial**: title/menu/dialogue/checkpoint passed,
but B/SELECT, left/right, overworld tile timing, sustained movement, recovery,
agent identity/room memory and badge progression remain untested. No gameplay
success rate is computed. Do not require all speculative tests before the first
short native agent loop; test further controls when that loop reaches them.

Model direction changed by the owner: Qwen can perform both System 1 and System 2;
Sol's earlier synthetic qualification remains reference evidence, not the selected
execution stack. No Qwen paid comparison was performed here. The local YAML and
existing vLLM launcher now prepare one pinned FP8 checkpoint with separate thinking
profiles; unit tests validate configuration/request/launcher contracts only, not
model quality, GPU fit, throughput or cache reuse. Next resource needed: a GPU
host/SSH tunnel or already-running compatible Qwen API endpoint. No live self-hosted
endpoint has been supplied for this phase yet, and no weights were downloaded here.

Native CLI/config regression coverage includes bounded boot frames, forwarding the
operator boot setting, interrupted manual dispatch without retry, source hash/frame
chaining and one-checkpoint Qwen profiles. Validation: 388 working-tree tests and
306 isolated staged-public tests passed, both with the same two retained
Starlette/anyio warnings. Ruff, compileall, shell parsing and Git whitespace checks
passed. Both no-key demos passed in the public snapshot and after wheel installation
outside the repository; static doctor validated the two Qwen profiles without
calling a model. No Qwen inference, GPU, cache-speed or autonomous-game result is
implied by these software checks.

## Next Sequence

1. Software integrity: both CPU suites, both no-key demos, CLI and installed packaging.
2. Native controls: title/menu/dialogue/state checked on Red. Qualify overworld
   controls when reached; do not silently treat the manual setup as agent progress.
3. Self-host one Qwen checkpoint: pinned revision/engine receipt, loopback/tunnel,
   bounded context and prefix cache. No repeat hosted model-selection screen.
4. Qualify the new actor-first memory path: inspect identities, recurring dialogue,
   similar rooms, blocked routes and out-of-order completion before speed claims.
   Then measure actor-only versus mixed HTTP probes and one selected OCR reader on
   the existing Qwen service; changing-scene tail latency remains unqualified.
5. Sustained progress: adjacent-target precision, revisits, Brock then Misty. Independent
   source evidence, not the model's progress narrative, establishes success.
6. Contribution test: matched models, source/control/start state and budgets;
   recent context versus retained memory path versus new background memory. Recent
   still has text/history; it is not memoryless. Do not merge old run schemas automatically.
   Include all failures and measure progress/stalls/calls/images/tokens/wall time/cost.
7. Optimize measured repetition: extraction/action fusion, text caching, shared packets
   or actual prefix reuse. Do not build ontology/tracking/serving stacks before need.
8. Another domain: unfamiliar game or camera workflow; real-time adapter before fast
   gameplay. Extract a standalone repo only after common code survives multiple domains.
9. Training only for a repeatable measured latency/cost/error bottleneck; label exports
   independently and keep cross-domain holdouts.

These are dependency gates, not reasons to run every speculative ablation before a demo.
Automatic ontology evaluation, independent identity verification, robust learned video tracking,
dense audiovisual recording, real-time async control, GBA/Doom, physical actuation,
Agmina integration, KV/encoder sharing, compaction, GPU/energy measurement, production
service and independent gameplay grading remain absent. They are not all prerequisites.
Address the next demonstrated blocker, not another architecture diagram.

## Provenance And Consolidation

Owner-supplied `streambudget_pixel_agent_20261004.zip` was based on reviewed commit
`076783012377d074b82efd7aaf296a8175b3a7f4`. All 80 manifest payloads passed SHA-256
verification before import. ZIP SHA-256:
`4830833b48ab89a16c91872c24501f98ed3c7813c4517f54454c791a01073b7e`.

It is a partial reconstructed tree, not a replacement clone. Its store/media/types/
license matched base HEAD; the newer local store/types and runtime/research edits
were preserved. Import only the interactive package, six behavior tests and two
templates, then integrate CLI/tests/docs. Do not overwrite legacy core from the ZIP.

No parallel GAME_SPEC/HANDOFF/STATUS/VALIDATION/UPSTREAM documents, duplicate agent
rules, merge installers, bundle-verifier helpers, static fixture previews, build logs
or second package manifest were imported. This guide, Git and existing tooling replace
those standalone-delivery artifacts. The original archive remains outside the source
tree for provenance. Existing unrelated edits are not automatically part of this commit.

Reference sources (not reproduced claims):
- https://github.com/AranKomat/streambudget
- https://docs.pyboy.dk/ and https://github.com/Baekalfen/PyBoy
- https://github.com/AranKomat/physical-agent-harness (identity/evidence design donor)
- https://github.com/AranKomat/agmina-runtime (future scheduling, not world store)
- https://github.com/ruc-datalab/EvoOntology (bounded schema governance)
- https://arxiv.org/abs/2608.05703 (StreamArena/StreamMind reference, not our score)
- https://typesafe.ai/blog/introducing-system-one-models-and-jev (typed interface)
- https://github.com/Clad3815/gpt-play-pokemon-firered (privileged-state reference only)
- https://arxiv.org/abs/2203.10539 and https://arxiv.org/abs/2207.08417 (text research)
- https://github.com/PaddlePaddle/PaddleOCR (possible plugin, not bundled)
