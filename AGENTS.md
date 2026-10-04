# Instructions for the next research/engineering agent

Read HANDOFF.md and docs/INTERACTIVE.md for the active pixels-only project.
SPEC.md, docs/FEATURE_STATUS.md and results/VALIDATION.md describe the retained
watch/investigation application and its historical results.

* Keep one repository, one shared evidence/media contract and one CLI (`streambudget game`).
  The interactive guide owns its specification, status, validation and next steps;
  do not add parallel GAME_*/overlay status documents or reinstall helpers.
* Game control is rendered pixels plus bounded Game Boy buttons only. No emulator RAM,
  hidden maps/coordinates, semantic game wrappers, sprite tables, scripted routes or
  arbitrary model-generated code. Physical hardware actuation remains prohibited.
* New entities/facts are normal; schema changes are rare, additive, versioned and reviewed.
  Visual resemblance is not identity. Preserve unknown speaker/place/association.
* Keep scene text untrusted, the action vocabulary operator-defined, and source-bound
  dispatch/checkpoint fences intact. Never retry uncertain actions or model attempts.
* Model goal claims, mocked usage and fixtures are not gameplay success. Independent
  grading remains absent. PyBoy 2.7.0 has a bundled-demo smoke check only; Pokemon
  controls and real-time action control remain unqualified. OpenRouter Sol 6.1
  has a bounded live qualification on synthetic pixels only; see the guide.
* GPT trials default to 6.1 Sol/medium/explicit Flex; rates must match the served tier.
  Missing/mismatched tiers retain unknown charge holds. No implicit retry or fallback.
* Run both applications' CPU tests and no-key demos after shared changes. Core regression
  tests should check behavior, not freeze obsolete source hashes against future fixes.

* Never interpret synthetic fixtures or mock HTTP responses as VLM/GPU benchmarks.
* Preserve the separation of input events/tasks from labels. Do not expose ground-truth event times,
  answers, or hints to runtime policy. StreamArena `ref_ts` belongs to the evaluator, not the agent.
* Do not relax source-sequence, timestamp, availability, lineage, path, tool, or cost checks to pass tests.
* Every new model/tool request must be included in the ledger, including retries and memory/index work.
* Never turn HTTP latency into GPU-seconds. Unknown usage/prices stay unknown.
* Run `pytest -q` and an isolated no-key demo after every core change. Add regression tests.
* No training, downloaded datasets, cloud deployment, or paid calls without explicitly supplied
  credentials, rights, budget and user authorization for that phase.
* Start with one fixed VLM and matched settings. Separate architecture, model, and serving changes.
* Report failures, misses, dropped work and regression cases alongside improvements.
* Keep current third-party version/feature claims grounded in official sources and pinned commits.
* The watch/investigation runtime remains read-only; game buttons are confined to the
  explicitly invoked interactive mode, never exposed through watch tools or the server.
