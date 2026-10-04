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
pixels -> extraction -> shared evidence + world beliefs
                                    |
                       selective planning / retrieval
                                    |
latest frame + intent + beliefs -> one action ID -> bounded buttons -> new pixels
```

One repository, one package distribution and one CLI: `streambudget game ...`.
Existing `streambudget demo/run/serve/...` commands remain. Game buttons are never
tools in the watch agent/server; no physical hardware controller is exposed.

| Module | Responsibility |
|---|---|
| Existing `store.py`, `media.py`, `types.py` | Shared immutable evidence, hashes, snapshots and lineage |
| `interactive/contracts.py`, `ontology.py`, `world.py` | Wire contracts, reviewed schema, beliefs and conversations |
| `interactive/runner.py`, `environment.py`, `locking.py` | Source-bound stepped execution, PyBoy, single-writer ownership |
| `interactive/models.py`, `prompts.py`, `context.py` | Synchronous role calls, accounting, bounded relevant context |
| `interactive/text.py` | Standalone text association/change/scroll utilities |
| `interactive/qualification.py`, `report.py`, `fixture.py`, `cli.py` | Probes, reports, explicit test double and commands |

The game coordinator is synchronous/stepped; the retained camera scheduler is
asynchronous. Their execution and ledger semantics differ. Reuse storage, but do
not force one runner/ledger onto both merely to reduce file count.

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
persistent text from repeated messages but may overcount or miss events. The general
VLM extracts text today. Temporal text/cache/scroll utilities are fixture-tested,
not connected to a dedicated OCR engine in the live loop.

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
button, tick and operator-checkpoint APIs are used. PyBoy 2.7.0 boot/capture, bounded
execution and save/load passed on its bundled demo ROM. That static screen does not
qualify Pokemon menu/navigation timing or prove any button had a useful game effect.
Supply an authorized ROM;
none is bundled, downloaded or redistributed by this integration.

Emulation pauses during inference. Exact frames and nominal frame/60 source time
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
Use fresh output paths. Doctor makes no model calls; `SET_EXACT_MODEL_ID` deliberately
means the template is not inference-ready. The toy game fixture uses a labelled
pixel-rule backend, not a learned model. Its report is a decision slideshow, not video.
Regenerate outputs under ignored `runs/`; no static preview copies are maintained.

## Native Qualification

Install the optional emulator when an authorized ROM is available:

```bash
python -m pip install -e '.[gameboy]'
python -m pip install 'pyboy==2.7.0'  # version actually smoke-tested here
streambudget game capture --rom /absolute/path/owned.gb --out runs/capture-001
streambudget game manual --rom /absolute/path/owned.gb \
  --load-state runs/capture-001/environment.state --button a \
  --press-frames 4 --release-frames 4 --count 1 --out runs/manual-001
```

Inspect before/after images, release behavior and loading. Default boot is 120
emulated frames, not a promise of a useful screen. Record package versions/ROM hash.
Manual qualification is not autonomous evidence. Disclose any chosen starting state;
a fresh-game demo cannot silently start from a convenient late save.

Configure a current exact image/JSON model in `configs/interactive/pokemon-local.yaml`.
The metered template intentionally fails validation until current rates are supplied.
Use credential environment-variable references, never keys in YAML/source. Local
billing is loopback-only; tunnel self-hosted remote inference.

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
| CPU emulator setup | Done for PyBoy bundled demo; Pokemon controls await authorized ROM |
| Execution/resume audit | Fixes and failure regressions implemented; no stronger crash recovery claim |
| GPT-6.1 Sol medium/Flex preparation | OpenRouter live probes reviewed; one Flex capacity failure retained; separate standard toy loop completed |
| First short trial preparation | Live three-decision toy loop done; native Pokemon trial awaits authorized ROM and button qualification |
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

## Next Sequence

1. Software integrity: both CPU suites, both no-key demos, CLI and installed packaging.
2. Native controls: authorized ROM boot/capture and single-button/state qualification.
3. Probes: menu/dialogue/room/navigation/ambiguous-target frames with a capable VLM.
4. Three to ten decisions: inspect each interpretation/action; separate grounding,
   button timing, JSON, stale context, identity, place memory and intent failures.
5. Sustained progress: adjacent-target precision, revisits, Brock then Misty. Independent
   source evidence, not the model's progress narrative, establishes success.
6. Contribution test: matched models, source/control/start state and budgets;
   `baseline: recent` versus `world`. Recent still has text/history; it is not memoryless.
   Include all failures and measure progress/stalls/calls/images/tokens/wall time/cost.
7. Optimize measured repetition: extraction/action fusion, text caching, shared packets
   or actual prefix reuse. Do not build ontology/tracking/serving stacks before need.
8. Another domain: unfamiliar game or camera workflow; real-time adapter before fast
   gameplay. Extract a standalone repo only after common code survives multiple domains.
9. Training only for a repeatable measured latency/cost/error bottleneck; label exports
   independently and keep cross-domain holdouts.

These are dependency gates, not reasons to run every speculative ablation before a demo.
Automatic ontology evaluation, independent identity verification, learned tracking/OCR,
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
