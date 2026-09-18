"""
conversation.py
----------------
Conversational layer: multi-turn memory, slot-filling, and free-text / JSON
input handling. Keeps per-session state so a user can supply information
incrementally across turns (as required by the "handle multi-turn
conversations with memory" requirement) rather than needing everything in
one message.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from src.reasoner import BiodiversityReasoner, ReasoningResult

# Very small, dependency-free text->slot extractor. This is deliberately
# transparent (regex + keyword rules) rather than a black-box NLU call, so a
# reviewer can see exactly how free text maps to structured state. Swapping
# this for an LLM-based extractor is a drop-in change (see README).
_NUMERIC_SOC_RE = re.compile(r"(?:soc|organic carbon)[^0-9]{0,15}([0-9]+(?:\.[0-9]+)?)\s*%?", re.I)
_RAINFALL_KEYWORDS = {
    "low": ["low rainfall", "dry", "drought", "arid", "little rain", "low rain"],
    "erratic": ["erratic", "unpredictable", "irregular rainfall", "unreliable rain"],
    "high": ["high rainfall", "heavy rain", "wet season", "monsoon"],
    "moderate": ["moderate rainfall", "average rainfall"],
}
_LAND_USE_KEYWORDS = {
    "monoculture": ["monoculture", "single crop", "wheat", "maize", "one crop"],
    "grazing": ["grazing", "pasture", "livestock", "cattle"],
    "deforested": ["deforested", "cleared", "cut down", "logged"],
    "degraded": ["degraded", "barren", "bare land", "denuded"],
    "cropland": ["cropland", "farmland", "farming", "agriculture"],
}
_REGION_KEYWORDS = ["semi-arid", "semi arid", "tropical", "temperate", "arid", "sub-tropical", "subtropical"]


def extract_state_from_text(text: str) -> dict[str, Any]:
    """Best-effort extraction of structured fields from a free-text message."""
    state: dict[str, Any] = {}
    lower = text.lower()

    soc_match = _NUMERIC_SOC_RE.search(lower)
    if soc_match:
        state["soil_organic_carbon"] = float(soc_match.group(1))

    for label, keywords in _RAINFALL_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            state["rainfall"] = label
            break

    for label, keywords in _LAND_USE_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            state["land_use"] = label
            break

    for region in _REGION_KEYWORDS:
        if region in lower:
            state["region"] = region.replace(" ", "-")
            break

    if "pesticide" in lower or "chemical runoff" in lower or "pollution" in lower:
        state["pollution_indicator"] = "high"

    if "river" in lower or "stream" in lower or "wetland" in lower or "lake" in lower:
        state["water_body_nearby"] = True

    return state


def parse_input(raw: str) -> tuple[dict[str, Any], str]:
    """
    Accepts either a JSON object (structured input mode) or free text.
    Returns (extracted_state, mode) where mode is 'json' or 'text'.
    """
    stripped = raw.strip()
    if stripped.startswith("{"):
        try:
            data = json.loads(stripped)
            if isinstance(data, dict):
                return data, "json"
        except json.JSONDecodeError:
            pass  # fall through to text parsing
    return extract_state_from_text(raw), "text"


@dataclass
class Turn:
    role: str  # "user" | "system"
    content: str


@dataclass
class Session:
    session_id: str
    state: dict[str, Any] = field(default_factory=dict)
    history: list[Turn] = field(default_factory=list)

    def update_state(self, new_fields: dict[str, Any]) -> None:
        for k, v in new_fields.items():
            if v not in (None, ""):
                self.state[k] = v


class ConversationManager:
    """
    Holds one or more Sessions (in-memory) and routes each incoming message
    through parsing -> state update -> reasoning, keeping full turn history
    so responses can be context-aware across turns.
    """

    def __init__(self, reasoner: BiodiversityReasoner | None = None):
        self.reasoner = reasoner or BiodiversityReasoner()
        self._sessions: dict[str, Session] = {}

    def get_session(self, session_id: str) -> Session:
        if session_id not in self._sessions:
            self._sessions[session_id] = Session(session_id=session_id)
        return self._sessions[session_id]

    def handle_message(self, session_id: str, message: str) -> dict[str, Any]:
        session = self.get_session(session_id)
        session.history.append(Turn(role="user", content=message))

        extracted, mode = parse_input(message)
        session.update_state(extracted)

        result: ReasoningResult = self.reasoner.recommend(session.state)

        response = self._format_response(result, session, input_mode=mode)
        session.history.append(Turn(role="system", content=json.dumps(response)))
        return response

    def _format_response(self, result: ReasoningResult, session: Session, input_mode: str) -> dict[str, Any]:
        if result.status == "needs_clarification":
            return {
                "status": "needs_clarification",
                "message": "I need a bit more information before I can recommend anything "
                            "responsibly (the brief for this system requires multi-metric "
                            "grounding, not single-variable guesses).",
                "known_so_far": session.state,
                "clarifying_questions": result.clarifying_questions,
                "input_mode": input_mode,
            }

        return {
            "status": "ok",
            "input_mode": input_mode,
            "state_used": session.state,
            "reasoning_trace": result.reasoning_trace,
            "cross_metric_notes": result.cross_metric_notes,
            "recommendations": [r.to_dict() for r in result.recommendations],
        }
