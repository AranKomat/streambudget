from streambudget.bench.metrics import score_events, score_qa, mcq_letter, pareto_frontier


def test_duplicate_alert_is_false_positive():
    labels = [{"type": "event", "source": "cam", "watch_id": "w", "start": 1, "end": 3}]
    preds = [{"source": "cam", "watch_id": "w", "observed_at": 2, "confidence": 1}] * 2
    scored = score_events(labels, preds, 1)
    assert (scored["tp"], scored["fp"], scored["fn"]) == (1, 1, 0)


def test_early_alarm_cannot_match_future_event():
    labels = [{"type": "event", "source": "cam", "watch_id": "w", "start": 10, "end": 20}]
    preds = [{"source": "cam", "watch_id": "w", "observed_at": 9, "confidence": 1}]
    scored = score_events(labels, preds, 1)
    assert scored["tp"] == 0 and scored["fp"] == 1


def test_overlap_uses_maximum_cardinality():
    labels = [{"type": "event", "source": "cam", "watch_id": "w", "start": 0, "end": 2},
              {"type": "event", "source": "cam", "watch_id": "w", "start": 1, "end": 1}]
    preds = [{"source": "cam", "watch_id": "w", "observed_at": 1, "confidence": 1},
             {"source": "cam", "watch_id": "w", "observed_at": 2, "confidence": 1}]
    assert score_events(labels, preds, 1)["tp"] == 2


def test_missing_qa_prediction_counts_as_wrong():
    labels = [{"type": "qa", "id": "a", "answer": "A"}, {"type": "qa", "id": "b", "answer": "B"}]
    preds = [{"question_id": "a", "text": "A", "status": "ok"}]
    assert score_qa(labels, preds)["accuracy"] == .5


def test_mcq_parser_refuses_ambiguous_prose():
    assert mcq_letter("(B)") == "B"
    assert mcq_letter("Answer: C.") == "C"
    assert mcq_letter("A or B, not sure") is None
    assert mcq_letter("The answer could be D") is None


def test_pareto_ignores_missing_cost():
    rows = [{"name": "a", "cost": 1, "quality": .8}, {"name": "b", "cost": 2, "quality": .7},
            {"name": "c", "cost": None, "quality": .9}, {"name": "d", "cost": 2, "quality": .9}]
    assert {r["name"] for r in pareto_frontier(rows)} == {"a", "d"}
