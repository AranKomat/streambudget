# Instructions for the next research/engineering agent

Read SPEC.md, HANDOFF.md, docs/FEATURE_STATUS.md and results/VALIDATION.md before changing behavior.

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
* Do not add autonomous physical actuation to this read-only research runtime.
