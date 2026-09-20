"""Schema diagnostics without logging model text, field values or arbitrary keys."""
from pydantic import ValidationError


SAFE_FIELDS = frozenset({"caption", "facts", "checks", "watch_id", "status", "confidence", "detail",
    "tool", "arguments", "query", "source", "start", "end", "limit", "frames", "question", "text",
    "evidence_ids", "abstain"})


def issues(exc: ValidationError) -> list[dict]:
    return [{"type": e["type"],
             "path": [part if isinstance(part, int) or part in SAFE_FIELDS else "<field>" for part in e["loc"]]}
            for e in exc.errors(include_url=False, include_context=False, include_input=False)[:20]]


def validate_model(schema, value, trace, operation):
    try:
        return schema.model_validate(value)
    except ValidationError as exc:
        trace.emit("schema_failure", operation=operation, schema=schema.__name__, issues=issues(exc))
        raise


def validate_response(schema, response, trace, operation):
    from .backend import BackendError

    try:
        value = response.json()
    except BackendError:
        trace.emit("schema_failure", operation=operation, schema=schema.__name__,
                   issues=[{"type": "invalid_json_object", "path": []}])
        raise
    return validate_model(schema, value, trace, operation)
