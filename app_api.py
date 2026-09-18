#!/usr/bin/env python3
"""
app_api.py
----------
Flask server for the Darukaa.Earth Biodiversity Intelligence system.

Serves:
  - GET  /              the field-console web UI (templates/index.html)
  - POST /chat           {session_id, message} -> structured reasoning response
  - GET  /api/knowledge  a JSON slice of the knowledge base, used to render
                          the "Knowledge base at a glance" table on the
                          homepage from the single source of truth rather
                          than duplicating content in the frontend.
  - GET  /health          liveness check

Run:
    python app_api.py
Then open http://localhost:5000
"""

from __future__ import annotations

import uuid

from flask import Flask, jsonify, render_template, request

from src.conversation import ConversationManager
from src.knowledge_store import KnowledgeStore

app = Flask(__name__)
manager = ConversationManager()
store = KnowledgeStore()


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/chat")
def chat():
    payload = request.get_json(force=True, silent=True) or {}
    session_id = payload.get("session_id") or str(uuid.uuid4())
    message = payload.get("message", "")
    if not message:
        return jsonify({"error": "message is required"}), 400
    response = manager.handle_message(session_id, message)
    response["session_id"] = session_id
    return jsonify(response)


@app.get("/api/knowledge")
def knowledge():
    """A trimmed view of the knowledge base for the homepage preview table."""
    interventions = [
        {
            "id": iv["id"],
            "name": iv["name"],
            "mechanism": iv["mechanism"],
            "impacted_metrics": iv["impacted_metrics"],
            "connected_variables": iv["connected_variables"],
            "source": iv["source"],
            "time_horizon": iv["time_horizon"],
            "confidence": iv["confidence"],
        }
        for iv in store.interventions
    ]
    return jsonify({"interventions": interventions, "count": len(interventions)})


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
