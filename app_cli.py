#!/usr/bin/env python3
"""
app_cli.py
----------
Terminal chat interface for the Darukaa.Earth Biodiversity Intelligence
system. Run with:

    python app_cli.py

Type free text ("Biodiversity is declining on my land, rainfall is low and
it's monoculture wheat in a semi-arid region") or paste a JSON object, e.g.:

    {"soil_organic_carbon": 0.3, "rainfall": "low", "land_use": "monoculture",
     "region": "semi-arid"}

Type 'exit' to quit, 'state' to see accumulated session memory, 'reset' to
clear it.
"""

import json
import sys
import uuid

from src.conversation import ConversationManager

BANNER = """
================================================================
 Darukaa.Earth - AI Biodiversity Intelligence (CLI)
 Describe your land's condition in plain text or JSON.
 Commands: 'state' | 'reset' | 'exit'
================================================================
"""


def print_recommendation(i: int, rec: dict) -> None:
    print(f"\n  [{i}] {rec['recommendation']}")
    print(f"      Why it works : {rec['why_it_works']}")
    print(f"      Metrics      : {', '.join(rec['impacted_metrics'])}")
    print(f"      Expected     : {rec['expected_effect']}")
    print(f"      Time horizon : {rec['time_horizon']}   Confidence: {rec['confidence']}")
    print(f"      Source       : {rec['source']}")
    if rec.get("matched_conditions"):
        print(f"      Matched on   : {'; '.join(rec['matched_conditions'])}")


def main() -> None:
    print(BANNER)
    manager = ConversationManager()
    session_id = str(uuid.uuid4())

    while True:
        try:
            raw = input("\nYou > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            sys.exit(0)

        if not raw:
            continue
        if raw.lower() in ("exit", "quit"):
            print("Goodbye.")
            break
        if raw.lower() == "state":
            print(json.dumps(manager.get_session(session_id).state, indent=2))
            continue
        if raw.lower() == "reset":
            manager._sessions.pop(session_id, None)
            print("Session memory cleared.")
            continue

        response = manager.handle_message(session_id, raw)

        if response["status"] == "needs_clarification":
            print(f"\nSystem > {response['message']}")
            for q in response["clarifying_questions"]:
                print(f"  - {q}")
            if response["known_so_far"]:
                print(f"\n  (Known so far: {response['known_so_far']})")
        else:
            print(f"\nSystem > Based on: {response['state_used']}")
            print("\n  Reasoning trace:")
            for step in response["reasoning_trace"]:
                print(f"    - {step}")
            if response["cross_metric_notes"]:
                print("\n  Cross-metric linkages:")
                for note in response["cross_metric_notes"]:
                    print(f"    - {note}")
            print("\n  Recommendations:")
            for i, rec in enumerate(response["recommendations"], start=1):
                print_recommendation(i, rec)


if __name__ == "__main__":
    main()
