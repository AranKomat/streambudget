from __future__ import annotations

from ..types import ContractError


def field(payload: dict, path: str):
    value = payload
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ContractError(f"Missing mapped signal field: {path}")
        value = value[part]
    return value


class JSONSignalMapper:
    """Map a decoded VSS/Kafka/customer JSON event into the runtime signal schema.

    Explicit mappings are required because VSS topic payloads vary by service
    and version. No guessed protobuf layouts or automatic model-generated mapping.
    """
    def __init__(self, mapping: dict):
        allowed = {"source_path", "timestamp_path", "timestamp_unit", "kind", "text_path", "data_fields",
                   "timestamp_origin"}
        if set(mapping) - allowed:
            raise ContractError("Unknown signal mapping keys")
        if mapping["timestamp_unit"] not in {"seconds", "milliseconds"}:
            raise ContractError("Specify timestamp_unit=seconds or milliseconds")
        self.mapping = mapping

    def convert(self, payload: dict) -> dict:
        m = self.mapping
        ts = float(field(payload, m["timestamp_path"]))
        if m["timestamp_unit"] == "milliseconds":
            ts /= 1000
        ts -= float(m.get("timestamp_origin", 0))
        if ts < 0:
            raise ContractError("Signal precedes session origin")
        return {"source": str(field(payload, m["source_path"])), "timestamp": ts, "kind": m["kind"],
                "text": str(field(payload, m["text_path"])) if m.get("text_path") else "",
                "data": {name: field(payload, path) for name, path in m.get("data_fields", {}).items()}}
