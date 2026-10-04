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
button, tick and operator-checkpoint APIs are used. Actual emulator boot, version,
button timing and checkpoint behavior remain unqualified. Supply an authorized ROM;
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

Probes never actuate. Review grounding, not just JSON/HTTP success. Qualify image
conventions, truncation, limits, thinking settings and latency. Generic Chat Completions
does not implement provider-specific Flex routing. Keep the owner's GPT-6.1 Sol/medium/
Flex-preferred preference when configuring such a route later; do not silently choose
regular serving or a different model. No live endpoint, prices or ranking is qualified.

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
Each probe has a fresh separately authorized budget, not free campaign capacity.
Configured reservations are admission control, not a provider invoice guarantee.

`known_provider_usd` is configured-rate usage, not invoice reconciliation. Local API
billing is zero, but rental/energy/storage cost is unmeasured, not free. Never reset
or reuse another project's ledger implicitly. Use provider-side spend controls.

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
