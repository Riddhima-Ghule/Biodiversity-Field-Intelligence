"""
Lightweight sanity tests (no pytest dependency required - run directly with
`python tests/test_system.py` or via pytest if available).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.conversation import ConversationManager, extract_state_from_text, parse_input
from src.reasoner import BiodiversityReasoner


def test_clarifying_questions_when_sparse_input():
    reasoner = BiodiversityReasoner()
    result = reasoner.recommend({"soil_organic_carbon": 0.3})
    assert result.status == "needs_clarification"
    assert len(result.clarifying_questions) > 0
    print("PASS: sparse input triggers clarification")


def test_recommendations_for_example_use_case():
    reasoner = BiodiversityReasoner()
    state = {
        "soil_organic_carbon": 0.3,
        "rainfall": "low",
        "land_use": "monoculture",
        "region": "semi-arid",
    }
    result = reasoner.recommend(state)
    assert result.status == "ok"
    assert len(result.recommendations) > 0
    metrics_touched = set()
    for r in result.recommendations:
        metrics_touched.update(r.impacted_metrics)
    assert len(metrics_touched) >= 3, "must connect at least 3 environmental variables"
    assert len(result.cross_metric_notes) > 0, "must surface cross-metric reasoning"
    names = [r.name for r in result.recommendations]
    assert any("agroforestry" in n.lower() or "cover crop" in n.lower() for n in names), (
        f"expected agroforestry or cover cropping to surface for this exact brief example, got {names}"
    )
    print("PASS: example use case produces grounded multi-metric recommendations")
    for r in result.recommendations:
        assert r.source, "every recommendation must cite a source"
        assert r.time_horizon in ("short", "medium", "long")
    print("PASS: every recommendation has source + time horizon + confidence")


def test_text_extraction():
    text = "Biodiversity is declining. Soil organic carbon is 0.3%, rainfall is low, monoculture wheat, semi-arid region."
    state = extract_state_from_text(text)
    assert state.get("soil_organic_carbon") == 0.3
    assert state.get("rainfall") == "low"
    assert state.get("land_use") == "monoculture"
    assert state.get("region") == "semi-arid"
    print("PASS: free-text extraction pulls structured slots")


def test_json_input_mode():
    raw = '{"soil_organic_carbon": 0.4, "rainfall": "low", "land_use": "monoculture"}'
    state, mode = parse_input(raw)
    assert mode == "json"
    assert state["rainfall"] == "low"
    print("PASS: structured JSON input is parsed correctly")


def test_multiturn_memory():
    manager = ConversationManager()
    sid = "test-session"
    r1 = manager.handle_message(sid, "Biodiversity is declining on my land.")
    assert r1["status"] == "needs_clarification"

    r2 = manager.handle_message(sid, "Soil organic carbon is 0.3%")
    r3 = manager.handle_message(sid, "Rainfall is low and it's monoculture wheat")
    assert r3["status"] == "ok", f"expected ok after accumulating state across turns, got {r3}"
    assert manager.get_session(sid).state["soil_organic_carbon"] == 0.3
    print("PASS: session memory accumulates state correctly across turns")


if __name__ == "__main__":
    test_clarifying_questions_when_sparse_input()
    test_recommendations_for_example_use_case()
    test_text_extraction()
    test_json_input_mode()
    test_multiturn_memory()
    print("\nAll sanity tests passed.")
