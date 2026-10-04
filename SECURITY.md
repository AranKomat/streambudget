# Security and safety boundaries

StreamBudget is a single-tenant research prototype. The watch/investigation application is
read-only observation, evidence retrieval and alert emission. The separate, explicitly
invoked `streambudget game` mode can execute bounded emulator buttons; no physical hardware
controller is exposed. It is not a safety controller, enforcement system, or authorization
to monitor people without appropriate permissions.

## Included protections

* Remote inference is disabled until explicitly enabled. Credentials come from environment variables.
* Runtime tools are fixed, typed functions; model outputs cannot run Python, shell, arbitrary SQL,
  choose network destinations, modify policies silently, or read arbitrary local paths.
* Scene text and retrieved observations are marked as untrusted. This reduces confusion but is not
  a proof of prompt-injection resistance. Watch/server tools cannot actuate. Game actions
  are limited to an operator-specified button manifest, not arbitrary code or tools.
* Media path traversal is rejected. Uploaded images are size checked and decoded before use.
* Source-time and availability-time filters, source-sequence snapshots, and transitive provenance
  block accidental future evidence exposure. These are correctness boundaries, not a sandbox for
  a malicious plug-in that deliberately lies about its inputs.
* Research REST endpoints require a bearer token, except public health. `--insecure-local` is
  explicitly restricted to loopback in the CLI. No CORS wildcard is enabled.
* Prompt bodies and images are omitted from operational traces. DB evidence, output answers,
  and local reports can still contain confidential or personal data.

## Not production-hardened

No multi-tenant isolation across processes, encrypted database, key rotation, row-level access rules,
retention/deletion policy, hardened reverse proxy, comprehensive rate limits, chunked-body limits,
malicious-media sandbox, secret manager, distributed transaction semantics, or signed evidence store
has been implemented. Use one isolated directory/process per trusted session. A new watch
runtime refuses an existing DB because watch/budget resume is not implemented. Game mode
supports only validated clean checkpoints, not arbitrary crash recovery.

Game mode records model contexts/responses and immutable frames in private run directories.
Its clean-resume checks bind ROM/state/pixel hashes, configuration, schema and attempt counts.
Ambiguous writes and pending inference cannot be replayed. Separate source time (stepped
emulator frames) from model wall time; there is no real-time freshness guarantee yet.
ROMs/checkpoints are ignored by Git. Obtain rights before use and never publish private
observations, prompts or checkpoints merely because a report was generated.

No statistical recall guarantee is implied by confidence values or cheap gates. Tiny/static/occluded
or very short events can be missed. A camera-off trigger cannot recover what was never captured.
Keep an appropriate recorder and independent safety systems. Do not automatically operate hardware
or deny people access based only on this prototype's observations.

For deployment, perform privacy, data-rights, model-license, security, and domain-specific reviews;
keep initial evaluations in shadow mode. Do not upload customer footage to an external model without
permission and an approved data-handling arrangement.
