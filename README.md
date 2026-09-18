# Darukaa.Earth — AI Biodiversity Intelligence Chatbot

An AI system that behaves like an environmental scientist, not a chatbot: it
retrieves grounded evidence from a structured knowledge base, reasons across
multiple environmental variables at once, and produces recommendations with
explicit mechanism, impacted metrics, expected effect size, time horizon,
confidence, and citation — never generic advice like "use sustainable
practices."

## 1. Architecture

```
                         ┌─────────────────────────┐
   User (text or JSON) ─▶│   ConversationManager    │  session memory,
                         │   (src/conversation.py)  │  multi-turn slot-filling
                         └────────────┬─────────────┘
                                      │ land-state dict
                                      ▼
                         ┌─────────────────────────┐
                         │   BiodiversityReasoner    │  clarifying-question
                         │   (src/reasoner.py)       │  logic, condition
                         └────────────┬─────────────┘  matching, ranking
                                      │ query
                                      ▼
                         ┌─────────────────────────┐
                         │     KnowledgeStore        │  TF-IDF vector index
                         │  (src/knowledge_store.py) │  over structured KB +
                         └────────────┬─────────────┘  reference documents
                                      │
                     ┌────────────────┴────────────────┐
                     ▼                                  ▼
        knowledge/knowledge_base.json         knowledge/references/*.txt
        (10 evidence-backed interventions,     (paraphrased summaries of
         clarifying questions, cross-metric     FAO / IPCC / IPBES reports
         relationship notes)                    used for retrieval grounding)
```

Two front ends sit on top of the same core (`src/`):

- **`app_cli.py`** — terminal chat, mandatory text-input mode, session memory
  kept in-process per run.
- **`app_api.py`** — Flask app serving `POST /chat`, `GET /api/knowledge`,
  and a styled single-page web console (`templates/index.html` +
  `static/`). The UI is a live client for the same reasoning pipeline —
  every recommendation and knowledge-base row it shows is fetched from the
  API, not hardcoded — so it doubles as a way to demo the retrieval and
  reasoning behavior visually without weakening the "not UI-heavy" spirit
  of the brief: the design's job is to make the reasoning legible (an
  inspectable pipeline diagram, an expandable reasoning trace on every
  answer, a live knowledge-base table), not to decorate a thin wrapper.

### Why TF-IDF instead of a hosted embedding model / vector DB

The retrieval layer (`KnowledgeStore`) uses scikit-learn's `TfidfVectorizer`
+ cosine similarity over the knowledge base and reference corpus. This is a
genuine vector-space retrieval method (each document becomes a sparse vector,
queries are compared by cosine distance) — it satisfies "use RAG, embeddings,
or vector databases" without requiring a downloaded model or an external API
key, so a reviewer can run the whole thing offline in under a minute.

**To upgrade to dense embeddings / a real vector DB** (e.g. for a stronger
submission or production use), swap `KnowledgeStore._vectorizer` for
`sentence-transformers` embeddings and store vectors in Chroma/FAISS/Pinecone
— the `retrieve()` interface (`query, k -> list[RetrievedChunk]`) does not
need to change, so nothing downstream (`reasoner.py`, `conversation.py`)
would need to be touched.

### Why retrieval + rules, not a single LLM prompt

The reasoning layer combines **retrieval** (semantic match to the land state)
with **structured condition scoring** (`applicable_when` blocks per
intervention, e.g. `soil_organic_carbon_pct_max`, `land_use in [...]`). This
is what lets every recommendation come with an inspectable "why this was
selected" trace (`matched_conditions`, `relevance_score`) instead of being an
LLM's unverifiable guess. It also directly satisfies the brief's explicit
constraint: "No generic LLM-only solutions."

An LLM is still a natural addition **on top** of this layer, to turn the
structured JSON into more conversational prose for an end user, or to power a
better free-text slot extractor than the regex/keyword one currently in
`conversation.py`. That integration point is intentionally left as a single,
clearly-marked seam (`extract_state_from_text` in `src/conversation.py`) so
it can be swapped for a call to the Anthropic API (`claude-sonnet-4-6` or
similar) without touching the reasoning or knowledge layers.

## 2. Data / Schema

`knowledge/knowledge_base.json`:

```jsonc
{
  "interventions": [
    {
      "id": "KB001",
      "name": "...",
      "applicable_when": {                 // structured condition block
        "soil_organic_carbon_pct_max": 1.5,
        "land_use": ["monoculture", "cropland"]
      },
      "action": "...",                     // what to do
      "mechanism": "...",                  // why it works (scientific reasoning)
      "impacted_metrics": ["..."],         // which environmental metrics improve
      "connected_variables": ["..."],      // which of the 5 core categories this spans
      "effect_estimate": "...",            // measurable improvement, with numbers
      "time_horizon": "short|medium|long",
      "confidence": "high|medium|low",
      "source": "...",                     // citation
      "source_type": "..."
    }
  ],
  "clarifying_questions": { "soil_organic_carbon": "...", "...": "..." },
  "metric_relationships": [
    { "pair": ["metric_a", "metric_b"], "relationship": "...", "source": "..." }
  ]
}
```

10 interventions currently cover soil health, land use, water/rainfall,
pollution, and habitat structure — each explicitly tagged with which of the
five required knowledge categories it touches, so multi-metric reasoning is
traceable rather than incidental.

`knowledge/references/*.txt` — five short, paraphrased summary documents
standing in for indexed research reports (FAO, IPCC AR6 AFOLU, IPBES Global
Assessment, IPBES Pollinators Assessment, FAO/WRI restoration). In a
production system these would be full paper/report chunks embedded via the
same pipeline described above.

## 3. Local Setup

```bash
git clone <repo-url>
cd darukaa-biodiversity-ai
python3 -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt

# Run the CLI chat:
python app_cli.py

# OR run the web console + API:
python app_api.py
# then open http://localhost:5000 in a browser for the field-console UI,
# or POST directly to http://localhost:5000/chat

# Run the sanity test suite:
python tests/test_system.py
# (or, if pytest is installed:  pytest tests/ -v)
```

No API keys or external network calls are required to run the core system —
everything (knowledge base, retrieval, reasoning) runs locally.

### Example session (text input)

```
You > Biodiversity is declining on my land.
System > I need a bit more information...
  - What is the soil organic carbon percentage (SOC%)? ...
  - What is the rainfall pattern in the area ...
  - What is the current land use or crop pattern ...

You > Soil organic carbon is 0.3%, rainfall is low, monoculture wheat, semi-arid region.
System > Based on: {'soil_organic_carbon': 0.3, 'rainfall': 'low', 'land_use': 'monoculture', 'region': 'semi-arid'}
  Recommendations:
  [1] Construct semi-circular stone/earth bunds or zai planting pits ...
  [2] Integrate multipurpose trees (agroforestry) ...
  [3] Introduce legume-based cover crops ...
  [4] Diversified crop rotation ...
```

### Example session (structured JSON input)

```json
{"soil_organic_carbon": 0.3, "rainfall": "low", "land_use": "monoculture", "region": "semi-arid"}
```
paste directly into the CLI, or POST as `{"session_id": "...", "message": "<the JSON above as a string>"}` to `/chat`.

## 4. CI/CD

This repository ships a minimal GitHub Actions workflow at
`.github/workflows/ci.yml` that, on every push/PR:

1. Sets up Python 3.11
2. Installs `requirements.txt`
3. Runs `python tests/test_system.py` (exits non-zero on any failed assertion)

There is no deployment step configured by default (no live infrastructure is
assumed) — the workflow is a correctness gate, not a deploy pipeline. To add
deployment (e.g. to Render/Railway/Fly.io for `app_api.py`), add a step after
tests pass that builds and pushes the app; the app has no stateful external
dependencies (session memory is in-process) so it deploys as a single
container.

## 5. Design decisions & trade-offs (for reviewers)

- **In-memory session store**: sessions live in a Python dict inside
  `ConversationManager`, not a database. This is a deliberate scope cut for a
  hackathon timeline — swapping in Redis/Postgres for session persistence is
  a small, isolated change (`ConversationManager._sessions`).
- **Free-text slot extraction is regex/keyword-based, not an LLM call**: kept
  transparent and dependency-free by design; the seam for upgrading to an
  LLM-based extractor is documented above and in code comments.
- **Confidence levels** on each recommendation are a qualitative
  `high/medium/low` editorial judgment based on the strength of the cited
  evidence (meta-analysis / systematic assessment vs. single study), not a
  statistically calibrated score — flagged as `optional but valuable` in the
  brief and included for completeness.

## 6. Meeting the evaluation criteria

| Criterion | How this repo addresses it |
|---|---|
| Depth of reasoning (30%) | Every recommendation is retrieved + condition-scored against the actual input state, not templated; `reasoning_trace` in every response shows the retrieval → scoring → ranking steps. |
| Scientific grounding (25%) | Every intervention cites a named source (FAO/IPCC/IPBES/peer-reviewed study) with a numeric effect estimate. |
| Knowledge system design (20%) | TF-IDF vector retrieval (`KnowledgeStore`) over a structured JSON knowledge base + indexed reference documents; upgrade path to dense embeddings documented above. |
| Conversational intelligence (15%) | `ConversationManager` keeps per-session memory, asks clarifying questions when critical fields are missing, and accumulates state across turns (see `test_multiturn_memory`). |
| Output clarity (10%) | Every recommendation returns `recommendation / why_it_works / impacted_metrics / expected_effect / time_horizon / confidence / source` as structured JSON, rendered readably in both the CLI and API. |
