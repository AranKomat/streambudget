# Examples

Start with `streambudget demo --out runs/demo`. It creates its own frames, observations, tasks and
independent labels; no downloaded dataset is needed. Do not use its synthetic backend for a real
benchmark claim.

`tasks.jsonl` illustrates a visual dwell objective, a scalar sensor predicate and a later investigation.
Use it only with a `camera` recording of at least 119 seconds, or change the question time to match
your recording. The numeric threshold is an example, not an operator safety recommendation.

`signal-mapping.json` is a generic schema example for `JSONSignalMapper`. It is **not** a claim that a
specific NVIDIA VSS release produces those exact fields. Map actual source timestamps, origin, and
nested fields after inspecting the deployed upstream schema.

```python
import json
from streambudget.adapters.signals import JSONSignalMapper
mapping = json.load(open("examples/signal-mapping.json"))
message = {
    "sensor": {"id": "camera"}, "timestamp_ms": 12000,
    "description": "External package detection",
    "detection": {"count": 1, "label": "package", "confidence": 0.9},
}
api_signal = JSONSignalMapper(mapping).convert(message)
# POST api_signal to /v1/signals through an authenticated trusted client.
```

`push_image.py` posts one local image to the authenticated live server using session receipt time.
It can trigger paid model work when the server has a real backend enabled. It is not a video
synchronization benchmark. The full `bridge-video` CLI requires PyAV; its actual RTSP behavior is
not locally validated. See [API documentation](../docs/API.md).
