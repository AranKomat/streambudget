# StreamBudget: product and implementation specification

Version 0.1.0 • 2026-09-20 • API-first research implementation

This document is self-contained. It describes the problem, the proposed product, the code that exists,
its limits, and the experiments required before claiming commercial value. The validation receipt is
in [results/VALIDATION.md](results/VALIDATION.md); outstanding work is in [HANDOFF.md](HANDOFF.md).
Primary sources are identified as R01–R15 in [REFERENCES.md](docs/REFERENCES.md).

## 1. What this is and why build it

StreamBudget is a runtime that places a computation policy between continuous observations and
expensive vision-language-model calls. It ingests video frames and structured observations, maintains
searchable evidence, monitors standing objectives, and selectively obtains more visual evidence when
an agent investigates something. Recorded and live input use the same evidence and tool semantics.

The product hypothesis is that an existing video-analytics developer could operate more streams or
more semantic monitoring objectives within a given inference budget, without unacceptable loss of
accuracy or delivered-response latency. The buyer would purchase capacity, predictable behavior, and
integration—not merely a reduction in a model's nominal visual-token count.

This is a hypothesis, not a measured market or performance result. The code has local CPU and HTTP
contract tests, but has not demonstrated real-world accuracy, GPU savings, customer demand, or an
advantage over a tuned commercial stack. A small recent-frame model or detector-plus-rules pipeline
may be the right solution for some tasks. The evaluation must permit that result.

Initial intended use is read-only industrial/security video analysis: watch a packing area or loading
dock, retain evidence for exceptions, and answer a later investigation. Drones and mobile cameras are
possible subsequent workloads, but motion-based gating must be re-evaluated under camera motion.
Robot motor control, autonomous driving control, biometric identification, and automated enforcement
are explicitly outside v0. No external physical action is implemented.

## 2. Where the design comes from

Three existing lines of work inform the design; none is claimed as our invention.

**StreamMind / StreamArena** provide a continuous-agent reference: separate observation, memory,
retrieval, and monitoring responsibilities and a driver whose video clock continues while an agent
answers. StreamArena is a benchmark and toolkit; StreamMind is one agent implementation. Their
released lifecycle was inspected to implement an independent bridge [R01–R03]. We do not reproduce
StreamMind's model prompts, memory architecture, scores, or v2 behavior merely by using that bridge.

**OmniAgent and VideoSeek** motivate bounded, explicit observation actions rather than one enormous
video prompt [R04–R05]. Our planner can search, inspect, read text, and answer. Its policy is prompted
and hand-engineered, not trained with OmniAgent's SFT/RL recipe.

**NVIDIA VSS** motivates using existing video infrastructure for recording, analytics, and tools,
rather than rebuilding an entire camera product [R06]. vLLM and SGLang provide model-serving
interfaces [R07–R08]; StreamBudget should generally sit above them. Their native-video and
model-specific optimizations are separate experiments, not implemented by our generic image client.

These sources establish useful interfaces and prior art. They do not establish that combining them
will be cheaper. The software here is original, not a source-level merge of three repositories.

## 3. Architectural boundaries

```text
AUTHORIZED SOURCES
  frames                 detector / OCR / ASR / audio-event / scalar sensor observations
    |                                             |
    +-------------------- ingress ----------------+
                              |
                    time/provenance validation
                              |
              durable evidence + compact searchable state
                              |
                 gate / standing objectives / query
                              |
                 bounded shared-work scheduler
                    /                       \
       visual observations                 investigation agent
       shared across watches               bounded JSON tool loop
           |                            search / inspect / OCR / state
           +----------------------+----------------+
                                  |
                           model-role adapter
                    perception / planner / verifier /
                       memory / embedding / ASR
                                  |
                  hosted API now; vLLM/SGLang later
                                  |
                   evidence-backed answers and alerts

Cross-cutting: snapshot checks, attempt ledger, deadlines, drops, traces, benchmarks.
```

The implementation is a single-process Python runtime using SQLite and HTTPX. CPU-heavy frame
handling and database operations are currently synchronous within an event-loop process. A bounded
queue prevents unbounded inference jobs; it does not make those CPU operations production scalable.

### Three source modes

| Mode | What can be inspected? | Intended claim |
|---|---|---|
| Archive | All observations in the prepared recording | Active retrospective investigation |
| Deterministic causal replay | Only evidence available by the replay cursor | Reproducible causal policy tests; not real-time throughput |
| Real-time replay / live | Arrived evidence while the clock continues | Concurrency and delivered-latency experiments, with explicit input-path caveats |

Archive mode does not eagerly caption every frame and accepts questions, not future-event watches.
It makes source evidence available first, then lets tools inspect selectively. Deterministic replay
drains work between events; this intentionally removes hardware contention to isolate correctness.
Real-time replay paces input independently of model work. Running at a speed other than one is a
stress/simulation setting, not a real-world delivered-latency measurement.

The current file-replay input is a predecoded JPEG manifest. Its runtime timing excludes original
video decode; the preparation command reports that separately. The optional PyAV bridge accepts a
trusted local recording or RTSP source but has not been run against a real camera in this build.

## 4. Evidence, time, and memory contracts

### Time is not one number

An observation has a capture interval `[start, end]` and an `available_at` time. A transcript covering
seconds 10–15 that arrives at 17 is not available at 15. The store also assigns sequence numbers for
source arrivals. A query snapshot contains an `as_of` cutoff and the maximum source sequence it may
consume. This prevents a late, backdated observation from silently appearing in an earlier query.

A derived item declares all consumed parent evidence. The store propagates the maximum consumed
source end time, source availability, and source sequence through the lineage. A summary made from
an entire hour cannot be visible at minute ten simply because its prose mentions minute nine.

There is an important exception to a literal frozen-database interpretation: a tool **may compute a
new caption now from frames that were already permitted by the snapshot**. Its row is newly created,
but all source leaves remain within the snapshot. This makes active inspection possible without
allowing new source evidence to leak backward. Model compute time is measured separately.

Snapshot filtering happens before keyword relevance statistics and vector ranking. Future text
must not change earlier search scores indirectly through global document-frequency statistics.
The tests cover this subtle case, as well as direct future access, late arrivals, and transitive
summaries. Plugins must honestly declare their inputs; this is not a sandbox that can prove an
arbitrary malicious Python plugin did not read another file.

### Keep three memories conceptually separate

1. **Durable evidence:** immutable source observations and received image bytes, addressed by evidence
   IDs and media hashes. The application retains references for audit and later reinspection.
2. **Searchable state:** captions, facts, events, OCR/transcripts, and summaries derived from evidence.
   SQLite FTS provides local keyword retrieval. Optional text embeddings add cosine retrieval and
   reciprocal-rank fusion. The default does not require a remote embedding service.
3. **Hot model state:** encoder features or KV caches. These are serving-engine concerns and are not
   implemented by the application store. Exact concurrent request coalescing is present; semantic
   answer reuse, encoder caching, and cross-request KV reuse are not.

Received image bytes are preserved, but this is **not an original-video recorder**. A manifest
prepared at one frame per second limits the agent to those observations. It cannot later recover an
unseen 100 ms event. A production integration should retain the dense original recording or connect
a lazy clip reader with causal access control.

Compaction optionally creates a higher-level textual view with lineage; it does not delete original
evidence. No finite lossy summary is promised sufficient for arbitrary future questions. Retention,
privacy deletion, and cold-storage policy are deployment work, not solved by keeping every frame.

## 5. Standing objectives and the inference policy

A `Watch` describes a source, natural-language goal, start/optional expiry, dwell time, confidence
threshold, observation freshness, repeat behavior, and cooldown. An optional scalar predicate evaluates
structured sensor/detector/audio-event data without a model call.

For visual watches, one observation request includes the active goals for a source, a small recent
frame set, and relevant recent structured signals. A worker returns a caption, facts, and `yes`, `no`,
or `unknown` for each goal. Missing verdicts become unknown. Low-confidence or unknown verdicts may
be sent to a separately configured verifier. Model confidence is not calibrated probability.

A dwell state advances only on newer positive observations. Unknown evidence or excessive observation
gaps break continuity. A timer alone cannot prove a condition remains true. Out-of-order model
completions cannot rewind the state. This is **sampled continuity**, not a guarantee about every
unobserved instant. Repeating watches can emit again after cooldown while a condition remains true;
edge-triggered per-episode behavior would need another explicit policy.

The initial observation policies are controlled baselines:

| Policy | Behavior |
|---|---|
| `fixed` | Periodic background observations at a configured interval |
| `motion` | Pixel/tile change or available cues, plus periodic refresh |
| `adaptive` | Change/cues/refresh, plus pending-dwell checks and randomized audits |
| `recent_only` | No general background semantic index; recent-frame QA and explicit watch polling |

Structured scalar predicates are available to all policies. Signals are observed inputs, not hidden
answer labels. Depending on arrival ordering, a signal can request inspection of a current frame or
mark a cue for the next fresh frame. The policy is not a learned semantic novelty model. The included
pixel gate is especially unsuitable as a standalone assumption for a moving camera.

The runtime shares a visual check across multiple goals rather than making independent identical
calls. It also coalesces exact in-flight jobs and requests. Sharing across different sources, models,
tenants, or partially overlapping visual work requires more sophisticated scheduling and is not
present. Identical pixels at a new timestamp are still a new observation; the code does not equate
them with an identical semantic answer.

## 6. Active investigation tools

The planner receives a bounded task context and returns one validated JSON action at each step.
The initial toolset is intentionally narrow:

| Tool | Actual v0 behavior |
|---|---|
| `search` | Search visible evidence with keyword retrieval and optional compatible text vectors |
| `inspect` | Select a bounded set of stored frames within a permitted interval and ask the perception model |
| `ocr` | Ask the configured VLM to read text in selected frames; no dedicated OCR engine is bundled |
| `read_state` | Read relevant visible facts/evidence |
| `query_sensor` | Retrieve stored, timestamped structured sensor observations |
| `answer` | Return a response with IDs of evidence exposed during the tool loop |

Tool schemas validate time ranges, source, frame limits, and argument shape. Arbitrary code, shell,
SQL, URLs chosen by scene text, and physical actions are not tools. The planner cannot authorize itself
to inspect future frames. Scene text and transcripts are marked untrusted inputs, but prompt injection
remains a risk; typed read-only tools reduce consequence, not eliminate the risk.

An answer can only cite evidence IDs from the tool context. Valid IDs do not prove the prose is
supported; semantic faithfulness still needs evaluation. The loop ends at a configured step limit or
budget/deadline. A small text-only planner may coordinate a larger visual worker. Model role swaps
are configuration changes, but their quality and tool-following ability need empirical verification.

No training is needed to execute this pipeline. A later learned action policy could choose resolution,
inspection, retrieval, refresh, and escalation. That policy should be trained only after establishing
strong hand-designed baselines and collecting consented traces with meaningful outcome labels.

## 7. Model and specialist interfaces

The default network backend uses an OpenAI-compatible Chat Completions format with timestamp text
and base64 JPEG image items [R07–R09]. Models and prices are user configuration, not hard-coded
assumptions about the newest release. Configure JSON mode, output-token parameter, image detail,
model/revision, retries, timeout, and provider-specific body fields explicitly.

A text-only planner does not require native audio. Optional WAV transcription uses a separate
`/audio/transcriptions` endpoint; text embedding uses `/embeddings` [R10–R11]. Environmental audio
may arrive as `audio_event` metadata from an external classifier. There is no bundled classifier,
native audio/video token interleaving, or full-duplex session implementation.

Hosted APIs are the first integration target. Later, serve appropriate checkpoints with a pinned
vLLM or SGLang build and point the adapter at that HTTP endpoint. The included shell launchers are
starter templates, not hardware validation. Native-video pruning, streaming prefill, KV persistence,
encoder disaggregation, and specialized cache UUIDs require dedicated backend work and testing.
Sending a list of image messages does not automatically activate a server's video-specific path.

Network inference requires explicit opt-in. No API credentials, benchmark datasets, model weights,
or paid services are included. The synthetic backend intentionally uses a red-pixel rule and a fixed
test policy; it never substitutes for a real VLM benchmark.

## 8. Scheduling, budgets, and reporting

The scheduler uses a bounded priority queue. Monitoring work precedes query jobs, which precede
background caption/index/compaction work. Jobs have an enqueue timestamp and a total deadline.
Expired or rejected jobs are recorded. Under a full queue, a new high-priority job can still be
rejected; there is no preemptive low-priority eviction or multi-tenant admission guarantee in v0.

Every network attempt, including retries, reserves an estimated cost and an attempt slot. Response
usage is reconciled against configured rates. Failed, cancelled, timed-out, malformed-usage, or
unpriced requests remain explicitly uncertain/provisional. A client timeout is not evidence that
remote billing stopped. Invalid provider counts must not silently become zero cost.

A configured dollar budget is an **estimated admission guard**, not an invoice guarantee. Visual
input tokenization varies by model/provider. Actual usage can exceed a reservation, which is recorded
as an overrun. Use provider-side spend controls as well. Embeddings, ASR, verification, and memory
writes share the ledger; specialist cost is not assumed free. External detectors and storage need
separate cost measurements and are not magically measured by this process.

Operational traces include stage timing, thread CPU time where instrumented, role/model/revision,
number of frames, attempts, retries, queue wait, errors, and drops. They exclude prompts and image
bodies. Evidence and result files still contain sensitive customer data. Reports intentionally leave
`gpu_seconds` null until real backend instrumentation exists. API wall time is not GPU time.

Run directories are isolated. Existing databases cannot be resumed through `Runtime`: durable watch,
budget, and alert-idempotency recovery are not yet implemented. This fails explicitly rather than
silently resetting budget state while reusing old memory.

## 9. Evaluation plan and acceptance decisions

### Three distinct kinds of evidence

A passing synthetic integration test proves that components connect and invariants hold. An API
smoke test proves that a chosen provider accepts requests and returns usable outputs. A representative,
held-out workload comparison is needed to support accuracy and cost claims. These must never be
presented as interchangeable.

The benchmark input schema has no `answer`, `reference_answer`, evidence hints, or target trigger
fields. Labels are read only by a separate scorer. The StreamArena bridge deliberately ignores the
future reference time supplied by its evaluation callback. That reference must not become a hint for
when to inspect or emit. The bridge follows inspected interfaces, but an actual upstream execution
is outstanding [R01–R03].

Video-MME conversion and output export support user-provided nested JSON and local footage [R12].
The current mode excludes subtitles. Future subtitle support must follow the benchmark's sampled-
frame alignment rule. Our exact/MCQ scorer is useful locally; an official result requires the official
protocol. NIST ActEV, high-FPS tests, audio-rich tasks, and other proposed domains are experiment plans,
not falsely labeled completed adapters [R13].

### What to measure

Use all-in cost per camera-hour, sustainable streams and objectives per deployment, per-class event
recall, false alerts per camera-hour, historical QA, evidence faithfulness, and delivered response
latency. Count missing, late, dropped, and malformed outputs rather than filtering them away. Retain
both evidence timestamp and delivered timestamp. Do not measure only model latency after a long
input chunk has already accumulated.

The local event scorer uses one-to-one matching between alerts and labeled intervals. Duplicate
alerts are false positives. It is not the official ActEV metric. Exact text scoring is not a reliable
semantic judge for open-ended answers. Use blinded human review or a separately validated judge with
an explicit rubric for those tasks.

Hold checkpoint, precision, serving settings, input access, and output budget constant while studying
the computation policy. Compare tuned fixed-rate, motion/cue-plus-periodic, detector/rules-plus-VLM,
recent-window, and active-agent variants. Reproduce upstream StreamMind and the relevant VSS workflow
separately; the four local baselines do not stand in for those systems.

Use independent cameras/days/sites for splits. Include short events, stationary deadlines, camera
motion, lighting, occlusion, outages, late sensor data, and bursty multi-stream activity. Densely enough
captured input is essential for evaluating dynamic observation budgets. Separate one-off archive
questions from many queries over the same recording, since index costs amortize differently.

A suggested early commercial gate is approximately 2× sustainable capacity or 50% lower total cost
at a customer-agreed absolute quality floor and latency requirement. This is a target to investigate,
not an achieved result or a guarantee. Report confidence intervals and a cost-quality frontier rather
than selecting a flattering single score. Hold a final evaluation set untouched during tuning.

## 10. Integration and commercialization scope

VSS/customer integrations should normalize observations into source-time-stamped evidence and expose
recorded frames or clips through a bounded tool contract. The repository contains a configurable JSON
signal mapper, not a deployed Kafka/protobuf client or a verified VSS release integration. Treat
VSS's concrete endpoints, authentication, model licenses, and storage path as version-specific work.

Do not make a full video management system or independent serving engine the first deliverable. The
initial differentiating work is computation policy, evidence reliability, shared tasks, and complete
measurement. Whether that produces a defensible company depends on actual customers, implementation
burden, and savings over their optimized existing systems.

Benchmark annotations are not a runtime dependency. Deployment creates predictions, captions, tracks,
and indexes from the customer's authorized inputs. Benchmark questions and answer keys are separate
materials with their own licenses. Code licensing does not override footage/annotation/model terms.
The original code here is MIT. No noncommercial dataset or upstream source tree is redistributed.
See [THIRD_PARTY.md](THIRD_PARTY.md) and Creative Commons' own license language [R14].

## 11. Research-agent continuation

Start with [HANDOFF.md](HANDOFF.md). The priority is a real API smoke test, a small independently
reviewed workload, matched baselines, and an upstream compatibility run. GPU profiling follows once
a useful policy difference exists. Do not begin by training a model simply because training code is
available elsewhere.

Later SFT/RL should optimize a constrained cost-quality objective, not maximize raw skipping. Logs of
chosen actions do not reveal the outcome of every unchosen action. Randomized audits, shadow
full-compute comparisons, or carefully designed counterfactual rollouts are needed to avoid training
on biased “the gate never made a mistake” labels. Frozen holdouts and rights-cleared training data
remain mandatory. Critical recall and latency requirements should not be traded away through an
unexamined scalar reward.

The authoritative implementation status is [FEATURE_STATUS.md](docs/FEATURE_STATUS.md). Every future
result should identify code commit, config, dataset rights/split, model revision, hardware, input
path, timing mode, full costs, and failures. The product should earn its performance claims through
those artifacts, not through an architectural resemblance to a recent paper.
