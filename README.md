# The Flame Threat Database

**A threat-intelligence repository for the Fantastic Four: relational records, hybrid RAG retrieval, ML threat classification and an analyst review workflow.**

> DEV OR DIE · Web Development Track · Culrav-Avishkar, MNNIT Allahabad
> Problem Statement 02 — *Villain Intelligence & Threat Analysis*

---

## Contents

1. [Overview](#1-overview)
2. [Problem Statement Coverage](#2-problem-statement-coverage)
3. [System Architecture](#3-system-architecture)
4. [Tech Stack](#4-tech-stack)
5. [Project Structure](#5-project-structure)
6. [Database Design](#6-database-design)
7. [Authentication, Clearance and Audit](#7-authentication-clearance-and-audit)
8. [Link and Assessment Review Workflow](#8-link-and-assessment-review-workflow)
9. [RAG Pipeline](#9-rag-pipeline)
10. [Machine Learning](#10-machine-learning)
11. [Pattern Alerts and Heat Map](#11-pattern-alerts-and-heat-map)
12. [Synthetic Dataset](#12-synthetic-dataset)
13. [Evaluation Protocol](#13-evaluation-protocol)
14. [API Reference](#14-api-reference)
15. [Local Setup](#15-local-setup)
16. [Deployment](#16-deployment)
17. [Testing](#17-testing)
18. [Known Limitations](#18-known-limitations)
19. [Acknowledgements](#19-acknowledgements)

---

## 1. Overview

The PS workflow is **Record → Classify → Link → Search → Review**. The system implements each stage as a concrete component:

| Stage    | Component                                                                                         |
| -------- | ------------------------------------------------------------------------------------------------- |
| Record   | PostgreSQL schema for villains, incidents and reports; every report is chunked, embedded (384-d) and indexed in pgvector on write |
| Classify | Random Forest suggests a 4-class threat level (Low/Medium/High/Critical) from 6 numeric and 2 categorical features, with a confidence score |
| Link     | Cosine-similarity search over report embeddings proposes incident–villain links, stored with status `suggested` |
| Search   | Hybrid retrieval (HNSW vector search + Postgres full-text + SQL filters, fused with Reciprocal Rank Fusion) behind a cited-answer assistant |
| Review   | State machine (`suggested → confirmed \| rejected`) on every AI-generated link and assessment; all transitions audited |

### Core design decisions

| Decision | Rationale |
| -------- | --------- |
| AI output is stored as `suggested`, never `confirmed` | Similar wording between two reports is not evidence of responsibility; only an analyst transition confirms a link |
| Clearance is a SQL `WHERE` predicate applied before ranking and before the LLM call | Filtering only at display time would still leak restricted text into the prompt |
| The LLM has no SQL access; it calls typed backend functions | Counts and lookups are exact queries, not generated text |
| Every citation is validated against the retrieved set | Cited IDs not in the retrieved set are stripped; an answer with no valid citation is replaced by an "insufficient intelligence" response |
| Relational store and vector index share one PostgreSQL instance | One transaction covers the report row and its chunks; no sync layer between two databases |

### Worked example (illustrative output)

Input reports:

| ID   | Text                                                                 |
| ---- | -------------------------------------------------------------------- |
| #101 | Unexplained energy surge near a Manhattan research laboratory        |
| #102 | Eyewitness reports metallic objects moving without physical contact  |
| #103 | Archived: similar electromagnetic disturbances in a past encounter with Doctor Doom |

`GET /incidents/101/similar` returns the top-k neighbours by cosine similarity, excluding the query incident itself, and `POST /ai/analyze/101` writes a `suggested` link and a threat assessment:

```json
{
  "incident_id": 101,
  "similar": [
    {"report_id": 103, "cosine_similarity": "<score>", "villain_id": 7},
    {"report_id": 102, "cosine_similarity": "<score>", "villain_id": null}
  ],
  "suggested_links": [
    {"villain_id": 7, "status": "suggested", "evidence": [103, 102]}
  ],
  "assessment": {
    "suggested_threat_level": "High",
    "model_confidence": 0.82,
    "review_required": true
  },
  "uncertainty": "Evidence insufficient to confirm villain involvement."
}
```

---

## 2. Problem Statement Coverage

**Basic (required)**

| PS requirement           | Implementation                                                                 |
| ------------------------ | ------------------------------------------------------------------------------ |
| Secure login & roles     | JWT (HS256) access tokens, Argon2id password hashes, `analyst` / `admin` roles |
| Villain profiles         | `villains` table: name, aliases[], abilities[], affiliations[], last known location (text + lat/lng), status |
| Intelligence reports     | `reports` table: source type, reliability score (0–1), clearance, FK to incident and optional villain |
| Threat classification    | `threat_level` enum with mandatory `reason`; list endpoints filterable by level |
| Incident linking         | `incident_villains` join table with status and evidence references            |
| Search & profile history | Name/ability/location filters (trigram + array containment), chronological history per villain |

**Advanced (optional)**

| PS requirement          | Implementation                                                                  |
| ----------------------- | ------------------------------------------------------------------------------- |
| Pattern alerts          | Four rule-based detectors with configurable thresholds (Section 11)            |
| Threat heat map         | Leaflet heat layer, severity-weighted points from an aggregated endpoint        |
| Audit trail & clearance | Integer clearance levels enforced in SQL; append-only `audit_logs`              |
| Smart search            | Hybrid RAG assistant with validated citations (Section 9)                       |

---

## 3. System Architecture

```mermaid
flowchart TB
    UI[React SPA] -->|HTTPS + Bearer JWT| API[FastAPI]
    API --> DEP[Dependencies: auth, role, clearance, audit]
    DEP --> ROUTES[Routes]
    ROUTES --> SVC[Services]
    SVC --> PG[(PostgreSQL 16 + pgvector)]
    SVC --> RAG[RAG: ingestion, retriever, assistant]
    SVC --> ML[ML: classifier, similarity]
    RAG --> PG
    RAG -->|prompt + retrieved context| LLM[Groq LLM, temperature 0]
    ML --> ART[(joblib artifacts)]
```

**Assistant request path**

```mermaid
flowchart LR
    Q[Question] --> R{Intent router}
    R -->|count / aggregate| S[statistics.py]
    R -->|villain lookup| P[profile query]
    R -->|open-ended| H[Hybrid retrieval]
    H --> C[Clearance predicate in SQL]
    C --> RRF[RRF fusion, top-8]
    RRF --> G[LLM generation]
    G --> V[Citation validation]
    V -->|valid| A[Response + citations]
    V -->|no valid citation| I[Insufficient intelligence]
    S --> A
    P --> A
```

The router is rule-based (regex and keyword patterns for count, date-range and "who is" queries) with an LLM classification fallback that returns a constrained enum (`stats | profile | search`).

---

## 4. Tech Stack

| Layer            | Technology                                                                   |
| ---------------- | ---------------------------------------------------------------------------- |
| Frontend         | React 18, Vite, Tailwind CSS, Recharts, Leaflet + leaflet.heat, Axios        |
| Backend          | Python 3.11, FastAPI, Pydantic v2, Uvicorn                                   |
| Database         | PostgreSQL 16, pgvector, SQLAlchemy 2.0 (psycopg 3)                          |
| Auth             | PyJWT (HS256), Argon2id via pwdlib                                           |
| RAG              | LangChain (documents, splitters, Groq client), Groq-hosted instruction model |
| Embeddings       | `all-MiniLM-L6-v2` (384-d, max sequence length 256 tokens)                   |
| ML               | scikit-learn (RandomForestClassifier, CalibratedClassifierCV), pandas, NumPy, joblib |
| Hosting          | Vercel (frontend), Render (backend), managed Postgres with pgvector          |

---

## 5. Project Structure

```
flame-threat-database/
│
├── README.md
├── .gitignore
├── .env.example
│
├── frontend/
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   ├── .env.example
│   └── src/
│       ├── main.jsx
│       ├── App.jsx
│       ├── index.css
│       ├── components/
│       │   ├── Layout.jsx
│       │   ├── ProtectedRoute.jsx
│       │   ├── ThreatBadge.jsx
│       │   ├── DataTable.jsx
│       │   ├── SearchBar.jsx
│       │   ├── Loader.jsx
│       │   ├── ErrorBoundary.jsx
│       │   └── CitationList.jsx
│       ├── pages/
│       │   ├── Login.jsx
│       │   ├── Dashboard.jsx
│       │   ├── Villains.jsx
│       │   ├── VillainProfile.jsx
│       │   ├── Incidents.jsx
│       │   ├── IncidentDetail.jsx
│       │   ├── Assistant.jsx
│       │   ├── ThreatMap.jsx
│       │   ├── ReviewCenter.jsx
│       │   └── AdminPanel.jsx
│       ├── context/
│       │   └── AuthContext.jsx
│       └── services/
│           └── api.js                  # Axios instance, token interceptor, endpoint helpers
│
├── backend/
│   ├── requirements.txt
│   ├── .env.example
│   │
│   ├── app/
│   │   ├── main.py                     # FastAPI app, CORS, exception handlers
│   │   ├── config.py                   # Settings, alert thresholds, retrieval params
│   │   ├── database.py                 # Engine, session, pgvector registration
│   │   │
│   │   ├── auth/
│   │   │   ├── security.py             # Argon2id hashing, JWT encode/decode
│   │   │   └── permissions.py          # Role guards, clearance predicate builder
│   │   │
│   │   ├── models/
│   │   │   ├── user.py
│   │   │   ├── villain.py
│   │   │   ├── incident.py
│   │   │   ├── report.py
│   │   │   ├── link.py                 # incident_villains + review status
│   │   │   ├── document_chunk.py       # text, vector(384), tsvector
│   │   │   ├── threat_assessment.py
│   │   │   ├── alert.py
│   │   │   └── audit_log.py
│   │   │
│   │   ├── schemas/                    # Pydantic request/response models
│   │   │
│   │   ├── routes/
│   │   │   ├── auth.py
│   │   │   ├── villains.py
│   │   │   ├── incidents.py
│   │   │   ├── reports.py
│   │   │   ├── ai.py
│   │   │   ├── reviews.py
│   │   │   ├── alerts.py
│   │   │   ├── analytics.py
│   │   │   └── admin.py
│   │   │
│   │   ├── services/
│   │   │   ├── intelligence.py         # CRUD, linking, state transitions
│   │   │   ├── alerts.py               # Rule-based detectors
│   │   │   ├── statistics.py           # Typed aggregate queries for the assistant
│   │   │   └── audit.py
│   │   │
│   │   ├── rag/
│   │   │   ├── ingestion.py            # Validate, chunk, embed, upsert
│   │   │   ├── embeddings.py
│   │   │   ├── retriever.py            # Vector + FTS + SQL filters, RRF fusion
│   │   │   ├── assistant.py            # Intent router, generation, citation validation
│   │   │   └── prompts.py
│   │   │
│   │   ├── ml/
│   │   │   ├── features.py
│   │   │   ├── classifier.py           # Threat level inference
│   │   │   ├── similarity.py           # Related-incident search
│   │   │   ├── train.py
│   │   │   └── trained_models/
│   │   │
│   │   └── utils/
│   │       ├── exceptions.py
│   │       └── logging.py
│   │
│   ├── scripts/
│   │   ├── init_db.py                  # CREATE EXTENSION vector, create_all, indexes
│   │   ├── seed_db.py
│   │   ├── ingest_reports.py           # Batch embed and index
│   │   └── create_admin.py
│   │
│   └── tests/
│       ├── conftest.py
│       ├── test_auth.py
│       ├── test_clearance.py
│       ├── test_crud.py
│       ├── test_rag.py
│       └── test_ml.py
│
├── data/
│   ├── generate_dataset.py             # Seeded synthetic generator
│   ├── villains.json
│   ├── incidents.json
│   ├── reports.json
│   ├── training_data.csv
│   └── evaluation/
│       ├── holdout_incidents.csv       # Held-out campaigns and templates
│       ├── related_pairs.csv           # Labelled related / unrelated pairs
│       └── rag_eval_questions.json     # Questions with gold report IDs
│
└── notebooks/
    └── threat_classification.ipynb     # Training, calibration, confusion matrix
```

---

## 6. Database Design

Tables are created by `scripts/init_db.py` from the SQLAlchemy models (`CREATE EXTENSION vector` followed by `create_all` and index DDL).

| Table                | Key columns                                                                                                  |
| -------------------- | ------------------------------------------------------------------------------------------------------------ |
| `users`              | `id`, `email` (unique), `password_hash`, `role` (enum), `clearance` (smallint 1–4)                           |
| `villains`           | `id`, `name`, `aliases` text[], `abilities` text[], `affiliations` text[], `last_location`, `last_lat`, `last_lng`, `status` (enum), `threat_level` (enum), `threat_reason` |
| `incidents`          | `id`, `title`, `description`, `event_time` timestamptz, `lat`, `lng`, `category`, `impact_level` (1–10), `energy_intensity`, `threat_level`, `clearance` |
| `reports`            | `id`, `incident_id` FK, `villain_id` FK null, `content`, `source_type` (sighting / eyewitness / archive), `reliability` real (0–1), `clearance`, `created_by` FK, `created_at` |
| `incident_villains`  | `incident_id` FK, `villain_id` FK, `status` (suggested / confirmed / rejected), `evidence_report_ids` int[], `source` (manual / ml), `reviewed_by`, `reviewed_at`; unique `(incident_id, villain_id)` |
| `document_chunks`    | `id`, `report_id` FK, `chunk_index`, `text`, `embedding` vector(384), `tsv` tsvector, `clearance`, `event_time`, `location` |
| `threat_assessments` | `id`, `incident_id` FK, `suggested_level`, `confidence`, `reason`, `model_version`, `review_status`, `reviewed_by` |
| `alerts`             | `id`, `type`, `dedupe_key` (unique), `description`, `incident_ids` int[], `status`, `created_at`             |
| `audit_logs`         | `id`, `user_id`, `action`, `entity_type`, `entity_id`, `timestamp`; append-only (no update/delete route)     |

**Indexes**

| Index | Definition | Purpose |
| ----- | ---------- | ------- |
| Vector | HNSW on `document_chunks.embedding` with `vector_cosine_ops` (`m = 16`, `ef_construction = 64`) | Approximate nearest-neighbour search |
| Full-text | GIN on `document_chunks.tsv` | Keyword and report-ID search |
| Name/location search | GIN trigram (`pg_trgm`) on `villains.name`, `villains.last_location` | Fuzzy search |
| Abilities | GIN on `villains.abilities` | Array containment filter |
| Time/threat | B-tree on `incidents(event_time)`, `incidents(threat_level)` | Date-range and level filters |

---

## 7. Authentication, Clearance and Audit

| Aspect | Implementation |
| ------ | -------------- |
| Passwords | Argon2id via pwdlib; plaintext never stored or logged |
| Tokens | HS256 JWT, 60-minute expiry; claims `sub`, `role`, `clearance`, `exp` |
| Login protection | Rate-limited login endpoint (5 attempts / minute / IP); identical error for unknown user and wrong password |
| Roles | `analyst`: read/write intelligence, review suggestions. `admin`: analyst rights plus user management and audit log access |
| Clearance | Integer 1–4 on users, incidents, reports and chunks. A record is visible iff `record.clearance <= user.clearance` |
| Enforcement point | `permissions.clearance_filter(user)` returns a SQLAlchemy predicate applied in every list, search, retrieval and statistics query |
| Retrieval | The predicate sits in the `WHERE` clause of both the vector and full-text queries, so restricted chunks are never ranked, fused or sent to the LLM |
| Audit | A dependency logs `view`, `create`, `update`, `approve`, `reject` and `login` events with user, entity and timestamp |
| Validation | Pydantic schemas on all request bodies; custom exception handlers return a uniform `{error, detail}` JSON shape |

---

## 8. Link and Assessment Review Workflow

```
            ┌──────────────┐   analyst approves    ┌───────────┐
 ML / RAG → │  suggested   │ ────────────────────► │ confirmed │
            └──────┬───────┘                       └───────────┘
                   │ analyst rejects
                   ▼
              ┌──────────┐
              │ rejected │
              └──────────┘
```

- Manual links created by an analyst are written as `confirmed` with `source = manual`; ML/RAG links are written as `suggested` with `source = ml`.
- Transitions are allowed only from `suggested`, only for `analyst` or `admin`, and each writes an audit row with the reviewer and timestamp.
- `threat_assessments` follow the same states. On approval the incident's `threat_level` is updated; on rejection it is left unchanged.
- The assistant treats only `confirmed` links as fact; `suggested` links are reported as hypotheses.

---

## 9. RAG Pipeline

### Ingestion (on report create)

1. Validate the payload and insert the `reports` row.
2. Build a LangChain `Document` with metadata: `report_id`, `incident_id`, `location`, `event_time`, `clearance`.
3. Split with a recursive character splitter at **~200 tokens, 30 overlap**. `all-MiniLM-L6-v2` truncates input at 256 tokens, so larger chunks would silently lose text. Reports under 200 tokens are stored as a single chunk.
4. Encode with `all-MiniLM-L6-v2` (384-d, L2-normalised), then insert `text`, `embedding`, `tsv = to_tsvector('english', text)` and metadata into `document_chunks`.

### Hybrid retrieval

| Stage | Detail |
| ----- | ------ |
| Metadata filter | Date range, location and clearance as SQL predicates, applied before ranking |
| Vector search | `ORDER BY embedding <=> :query_vec LIMIT 20` (cosine distance, HNSW) |
| Keyword search | `ts_rank_cd` over `tsv` with `websearch_to_tsquery`, `LIMIT 20` |
| Fusion | Reciprocal Rank Fusion, `score = Σ 1 / (60 + rank_i)` across both lists |
| Selection | Top 8 chunks after fusion; chunks from the same report are merged and deduplicated |

Exact-ID queries (e.g. "R-101") resolve through the keyword path; "similar to this incident" queries use the vector path only.

### Generation and citation validation

- Temperature 0; the prompt contains only retrieved chunks with their report IDs and `confirmed` / `suggested` link status.
- The model must return JSON: `answer`, `cited_report_ids[]`, `evidence_strength` (`weak | moderate | strong`), `uncertainty`.
- **Validation:** any cited ID outside the retrieved set is removed. If no valid citation remains, the response is replaced by the insufficient-intelligence message. One retry with a reformulated query is allowed before falling back.
- `evidence_strength` is computed from retrieval signals (number of distinct supporting reports, fused score, reliability of sources), not self-reported by the model.

### Typed tools (no raw SQL)

`get_villain_profile(id)`, `get_incident_history(villain_id)`, `count_incidents(filters)`, `incident_trend(window_days)`, `find_similar_incidents(incident_id, k)`. Parameters are validated by Pydantic before they reach the database.

---

## 10. Machine Learning

### Threat classifier

| Item | Detail |
| ---- | ------ |
| Task | 4-class ordinal classification: Low, Medium, High, Critical |
| Numeric features | `impact_level` (1–10), `energy_intensity`, `previous_related_incidents`, `incident_frequency_7d`, `verified_evidence_count`, `source_reliability` |
| Categorical features | `ability_category`, `incident_type` (one-hot via `ColumnTransformer`) |
| Model | `RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)` |
| Calibration | `CalibratedClassifierCV` (isotonic) so `model_confidence` approximates true class probability |
| Split | Stratified 80/20 on the generated set, plus a separate holdout built from different campaigns and templates (Section 12) |
| Output | `suggested_threat_level`, `model_confidence`, `review_required` |
| Review rule | `review_required = true` when confidence < 0.70 or predicted level ∈ {High, Critical} |
| Artifact | `joblib` file with a version string stored in `threat_assessments.model_version` |

The LLM may explain a prediction only from the actual feature values and retrieved evidence; it does not generate reasons independently.

### Related-incident finder

| Item | Detail |
| ---- | ------ |
| Representation | Report embeddings already stored in `document_chunks` (384-d) |
| Query | `embedding <=> :vec` cosine distance via pgvector, top-5, excluding the source incident's own chunks |
| Threshold | Chosen on `related_pairs.csv` by maximising F1 over similarity cut-offs, not set by hand |
| Output | Neighbour report IDs, similarity scores and each neighbour's linked villain, used to create `suggested` links |

---

## 11. Pattern Alerts and Heat Map

### Detectors

Run on incident insert as a FastAPI background task. All thresholds live in `config.py`.

| Detector | Rule (default) |
| -------- | -------------- |
| Repeated location | ≥ 3 incidents within 2 km (haversine) in 7 days |
| Repeated ability | Same ability tag on ≥ 3 incidents in 14 days |
| Timing cluster | ≥ 3 incidents in the same 2-hour time-of-day window within 14 days |
| Frequency spike | Today's incident count > trailing-30-day mean + 2σ |

Each alert carries a `dedupe_key` (type + normalised parameters + window), enforced unique, so re-running a detector does not duplicate alerts. Alerts reference their `incident_ids` and appear on the dashboard and in the Review Centre.

### Heat map

`GET /analytics/heatmap` returns `[lat, lng, weight]` triples aggregated to a fixed grid, with weight derived from threat level (Low = 1 to Critical = 4) and filterable by date range. The frontend renders them with `leaflet.heat`.

---

## 12. Synthetic Dataset

`data/generate_dataset.py` (seeded, `random.seed(42)`) generates the data. The PS provides none.

| Data | Target |
| ---- | ------ |
| Villain profiles | 20–30 |
| Incidents | 150–300 |
| Reports | 200–400 |
| ML training rows | 2,000–3,000 |
| Labelled report pairs | 100–200 |
| RAG evaluation questions | 20–30 |

**Generation design**

- **Campaigns:** 6–8 latent campaigns, each fixing a villain, energy signature, ability set, area and time pattern. About 60% of incidents belong to a campaign; about 40% are unrelated noise.
- **Realism:** ~10% ambiguous reports (no named villain), 5–10% missing fields, varied phrasing from several text templates per campaign.
- **Labels:** threat level derived from a rule over impact, intensity, history and evidence, with label noise added so the classifier has a non-trivial boundary.
- **Splits:** train/holdout are split by campaign and template group (`GroupShuffleSplit`), not by row, so the holdout contains patterns the model never saw.

---

## 13. Evaluation Protocol

All metrics are computed on the synthetic holdout (Section 12) and recorded in `notebooks/` after evaluation. Results reflect the simulated scenario only.

| Component         | Metric                                              | How it is measured                                                    |
| ----------------- | --------------------------------------------------- | --------------------------------------------------------------------- |
| Threat classifier | Macro-F1, per-class recall, confusion matrix        | Held-out campaigns and templates                                      |
| Threat classifier | Within-one-level accuracy                           | Predictions that are at most one level from the label (ordinal error) |
| Threat classifier | Calibration                                         | Reliability curve of `model_confidence` against observed accuracy     |
| Similarity        | Precision@5                                         | Labelled related / unrelated pairs in `related_pairs.csv`             |
| Retrieval         | Recall@k and MRR                                    | Evaluation questions with gold report IDs                             |
| Assistant         | Citation validity                                   | Every cited ID must be in the retrieved set (asserted in tests)       |
| Assistant         | Refusal on unanswerable questions                   | Evaluation questions with no supporting report                        |
| Clearance         | Restricted chunks returned to a lower-clearance user | Asserted to be zero across every list, search and retrieval path      |

---

## 14. API Reference

All endpoints except `/auth/login` require `Authorization: Bearer <token>`.

| Method | Endpoint                  | Purpose                                         |
| ------ | ------------------------- | ----------------------------------------------- |
| POST   | `/auth/login`             | Exchange credentials for a JWT                  |
| GET    | `/villains`               | List/search (`q`, `ability`, `location`, `threat_level`, `status`, `limit`, `offset`) |
| POST   | `/villains`               | Create profile                                  |
| GET    | `/villains/{id}`          | Profile with incident history                   |
| PATCH  | `/villains/{id}`          | Update profile                                  |
| GET    | `/incidents`              | List/filter (`from`, `to`, `location`, `threat_level`) |
| POST   | `/incidents`              | Create incident (triggers alert detectors)      |
| POST   | `/reports`                | Create report (triggers chunk, embed, index)    |
| GET    | `/incidents/{id}/similar` | Top-k similar incidents with scores             |
| POST   | `/ai/query`               | Assistant question → answer, citations, evidence strength |
| POST   | `/ai/analyze/{id}`        | Classifier + similarity → `suggested` assessment and links |
| POST   | `/reviews/{id}/approve`   | `suggested → confirmed`                         |
| POST   | `/reviews/{id}/reject`    | `suggested → rejected`                          |
| GET    | `/alerts`                 | Pattern alerts                                  |
| GET    | `/analytics`              | Counts by level, status and time window         |
| GET    | `/analytics/heatmap`      | Weighted grid points                            |

**`POST /ai/query`**

```json
// request
{ "question": "Which recent incidents share an electromagnetic signature?" }

// response
{
  "answer": "Reports R-101 and R-108 describe a similar electromagnetic signature ...",
  "citations": [
    {"report_id": "R-101", "snippet": "..."},
    {"report_id": "R-108", "snippet": "..."}
  ],
  "evidence_strength": "moderate",
  "uncertainty": "No confirmed link to a specific villain.",
  "route": "search"
}
```

Interactive docs are served at `/docs`.

---

## 15. Local Setup

**Requirements:** Python 3.11+, Node.js 18+, PostgreSQL 16 with the `pgvector` and `pg_trgm` extensions, a Groq API key.

**Backend**

```bash
git clone https://github.com/<your-org>/flame-threat-database.git
cd flame-threat-database/backend

python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
```

```env
DATABASE_URL=postgresql+psycopg://user:password@host:5432/flame
JWT_SECRET=change_me_to_a_long_random_string
JWT_EXPIRE_MINUTES=60
GROQ_API_KEY=your_groq_api_key
LLM_MODEL=a_model_available_on_your_groq_account
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
FRONTEND_ORIGIN=http://localhost:5173
```

```bash
python scripts/init_db.py            # extensions, tables, indexes
python ../data/generate_dataset.py   # synthetic data
python scripts/seed_db.py            # load into PostgreSQL
python scripts/ingest_reports.py     # chunk, embed, index
python -m app.ml.train               # fit, calibrate, save classifier
python scripts/create_admin.py       # first administrator

uvicorn app.main:app --reload
```

Backend: `http://localhost:8000` · Docs: `http://localhost:8000/docs`

**Frontend**

```bash
cd frontend
cp .env.example .env                 # VITE_API_URL=http://localhost:8000
npm install
npm run dev
```

Frontend: `http://localhost:5173`

---

## 16. Deployment

| Component | Platform                       | Configuration                                                             |
| --------- | ------------------------------ | ------------------------------------------------------------------------- |
| Frontend  | Vercel                         | Root `frontend`; env `VITE_API_URL`                                       |
| Backend   | Render web service             | Root `backend`; start `uvicorn app.main:app --host 0.0.0.0 --port $PORT`; all backend env vars set in the dashboard |
| Database  | Managed Postgres with pgvector | Run `init_db.py`, `seed_db.py`, `ingest_reports.py` once against it       |

- Set `FRONTEND_ORIGIN` to the deployed frontend URL for CORS.
- Commit the trained model artifacts, or run `python -m app.ml.train` during the build.
- The embedding model loads once at startup. If the host's memory limit is too small for PyTorch, swap the loader for an ONNX runtime build of the same model; the 384-d vectors stay compatible with the existing index.

---

## 17. Testing

```bash
cd backend && pytest
```

| Suite | Assertions |
| ----- | ---------- |
| `test_auth` | Valid/invalid login, expired and tampered tokens, role guards, rate limit |
| `test_clearance` | For each list, search and retrieval path, a lower-clearance user receives zero restricted records and zero restricted chunks |
| `test_crud` | Villain, incident and report lifecycle; unique `(incident_id, villain_id)`; allowed and rejected status transitions |
| `test_rag` | Recall@8 and MRR on the evaluation questions; cited IDs ⊆ retrieved IDs; refusal on unanswerable questions |
| `test_ml` | Classifier macro-F1 on holdout, probability outputs sum to 1, similarity excludes the source incident |

---

## 18. Known Limitations

- All data is synthetic. Reported metrics measure behaviour on the simulated scenario and do not indicate real-world predictive accuracy.
- The classifier can recover the generator's labelling rule; the campaign-grouped holdout reduces but does not remove this.
- Similarity thresholds are calibrated on synthetic pairs and must be recalibrated for other data.
- `all-MiniLM-L6-v2` is a general-purpose English model with a 256-token input limit, and it has no domain tuning.
- LLM output remains non-deterministic across model versions; citation validation bounds the failure mode but does not guarantee correct reasoning.
- Free hosting tiers cold-start after idle periods.

---

## 19. Acknowledgements

Thanks to the DEV OR DIE Web Development team at Culrav-Avishkar, MNNIT Allahabad, for the problem statement. Characters and setting belong to their respective owners and are used only within the context of the contest.
