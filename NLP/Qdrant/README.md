# Article Embedding Pipeline

A FastAPI service that ingests articles from Elasticsearch, optionally translates Indic-language content to English via GPT-4o-mini, embeds the text using OpenAI's `text-embedding-3-small` model, and stores the resulting vectors in Qdrant for semantic search.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Pipeline Flow](#pipeline-flow)
- [Configuration](#configuration)
- [API Endpoints](#api-endpoints)
- [Key Components](#key-components)
  - [Text Cleaning](#text-cleaning)
  - [Indic Language Detection & Translation](#indic-language-detection--translation)
  - [Token-Aware Batching](#token-aware-batching)
  - [Embedding](#embedding)
  - [Qdrant Upsert](#qdrant-upsert)
- [Data Models](#data-models)
- [Error Handling & Resilience](#error-handling--resilience)
- [Dependencies](#dependencies)
- [Environment Variables](#environment-variables)
- [Running the Service](#running-the-service)

---

## Overview

This pipeline solves a specific problem: Reddit/ASK-style article corpora often contain **code-mixed Indic language content** (e.g. Hindi mixed with English). Embedding such text directly produces poor semantic vectors because embedding models are primarily trained on English. The pipeline detects Indic scripts, translates only those articles, and then embeds everything in a unified English vector space — enabling high-quality multilingual semantic search.

**Tech stack:**

| Layer | Technology |
|---|---|
| API framework | FastAPI + Uvicorn |
| Source data store | Elasticsearch (async) |
| Translation | OpenAI GPT-4o-mini |
| Embeddings | OpenAI `text-embedding-3-small` (1536-dim) |
| Vector store | Qdrant |
| Tokenisation | tiktoken (`cl100k_base`) |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      FastAPI Service                     │
│                                                         │
│  POST /full_ingest   POST /today_ingest   POST /search  │
│        │                   │                   │        │
│        └──────┬────────────┘                   │        │
│               │                                │        │
│        run_pipeline()                  search()         │
└───────────────┼────────────────────────────────┼────────┘
                │                                │
        ┌───────┴────────┐               ┌───────┴──────┐
        │  Elasticsearch │               │    Qdrant    │
        │  (source data) │◄──────────────│  (ANN index) │
        └────────────────┘               └──────────────┘
                                                │
                                       OpenAI Embeddings API
                                       OpenAI Chat API (GPT-4o-mini)
```

---

## Pipeline Flow

### Ingest Pipeline (`/full_ingest` and `/today_ingest`)

```
Trigger endpoint
     │
     ▼
Elasticsearch scroll
  (articleBigContent + articleHashId, source=ask)
     │
     ▼
clean_text()
  - Remove emojis
  - Strip URLs
  - Remove control characters (preserve Indic Unicode)
  - Collapse whitespace
     │
     ▼
Indic script detection [needs_translation()]
  ├── YES → translate_text() via GPT-4o-mini
  │           (bounded concurrency: 8 parallel calls)
  └── NO  → pass through unchanged
     │
     ▼
clean_text_post_translation()
  - Strip any remaining non-ASCII artifacts
  - Collapse whitespace
     │
     ▼
_token_aware_batches()
  - Truncate each doc to ≤ 8,192 tokens
  - Group into batches ≤ 200,000 tokens and ≤ 2,048 docs
     │
     ▼
embed_texts()
  - Call OpenAI text-embedding-3-small per batch
  - Retry up to 3× with exponential back-off on failure
     │
     ▼
upsert_to_qdrant()
  - Deterministic UUID5 from articleHashId (idempotent re-runs)
  - Upsert in chunks of 200 points
     │
     ▼
IngestResponse
  { fetched, embedded, upserted, duration_seconds }
```

### Search Pipeline (`/search`)

```
POST /search  { query, top_k }
     │
     ▼
translate_if_needed()
  - Detect Indic script in query
  - Translate if detected; otherwise pass through
     │
     ▼
OpenAI embed query → 1536-dim vector
     │
     ▼
Qdrant query_points()
  - Cosine ANN search
  - Filter: articleSource = "ask"
  - Return top-k hits with payload
     │
     ▼
fetch_articles_by_hash_ids()
  - ES terms query for matched articleHashIds
  - Returns full articleBigContent + articleInsertedDate
     │
     ▼
Merge Qdrant scores with ES content
  (preserves Qdrant ranking order)
     │
     ▼
SearchResponse { results: [SearchHit], query_time_ms }
```

---

## Configuration

All tunable constants are defined near the top of `pipeline.py`:

| Constant | Default | Purpose |
|---|---|---|
| `ES_INDEX` | `articles` | Elasticsearch index name |
| `QDRANT_HOST` | `localhost` | Qdrant server hostname |
| `QDRANT_PORT` | `6333` | Qdrant server port |
| `COLLECTION` | `articles_ask` | Qdrant collection name |
| `EMBED_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `EMBED_DIM` | `1536` | Embedding vector dimension |
| `ES_PAGE_SIZE` | `500` | Documents per Elasticsearch scroll page |
| `ARTICLE_SOURCE` | `ask` | Value of `articleSource.keyword` filter |
| `TRANSLATE_CONCURRENCY` | `8` | Max parallel GPT-4o-mini calls |
| `MAX_TOKENS_PER_BATCH` | `200,000` | OpenAI batch token budget |
| `MAX_TOKENS_PER_DOC` | `8,192` | Per-document token hard cap |

---

## API Endpoints

### `GET /health`

Liveness check. Returns `{"status": "ok"}`.

---

### `POST /full_ingest`

One-time full ingest of **all** articles where `articleSource = ask`.

- Safe to re-run — deterministic UUIDs prevent duplicate vectors in Qdrant.
- Intended for initial population or full re-indexing.

**Response:**
```json
{
  "fetched": 12430,
  "embedded": 12430,
  "upserted": 12430,
  "duration_seconds": 184.5
}
```

---

### `POST /today_ingest`

Incremental daily ingest. Fetches only articles whose `articleInsertedDate` falls within the current UTC day (`[00:00:00, 24:00:00)`).

- Designed to be scheduled via cron during low-traffic windows.
- Returns `404` if no articles were inserted today.

**Response:** Same schema as `/full_ingest`.

---

### `POST /search`

Semantic search over embedded articles.

**Request body:**
```json
{
  "query": "best productivity apps for students",
  "top_k": 10
}
```

**Response:**
```json
{
  "results": [
    {
      "articleHashId": "abc123",
      "score": 0.9142,
      "articleBigContent": "...",
      "articleInsertedDate": "1716400000000"
    }
  ],
  "query_time_ms": 212.4
}
```

---

## Key Components

### Text Cleaning

Two-pass cleaning is used deliberately:

**Pass 1 — `clean_text()`** (before translation)
- Removes emojis via the `emoji` library.
- Strips HTTP/HTTPS URLs.
- Removes ASCII control characters (`\x00–\x1F`, `\x7F`) only — **does not** remove non-ASCII, since Indic Unicode must be preserved for detection and translation.
- Collapses excess whitespace.

**Pass 2 — `clean_text_post_translation()`** (after translation)
- Now safe to strip all non-printable non-ASCII (`[^\x20-\x7E\n]`).
- Collapses whitespace again to handle any artifacts introduced by the translation model.

This two-pass design avoids accidentally destroying Indic content before it can be detected and translated.

---

### Indic Language Detection & Translation

**Detection** uses a compiled regex over nine Unicode ranges:

| Script | Range |
|---|---|
| Hindi / Marathi | `\u0900–\u097F` |
| Bengali | `\u0980–\u09FF` |
| Punjabi | `\u0A00–\u0A7F` |
| Gujarati | `\u0A80–\u0AFF` |
| Odia | `\u0B00–\u0B7F` |
| Tamil | `\u0B80–\u0BFF` |
| Telugu | `\u0C00–\u0C7F` |
| Kannada | `\u0C80–\u0CFF` |
| Malayalam | `\u0D00–\u0D7F` |

**Translation** via GPT-4o-mini is only triggered when detection returns `True`. Pure-English articles skip the API call entirely, saving cost and latency.

Batch translation uses `asyncio.Semaphore(TRANSLATE_CONCURRENCY)` to limit parallel calls to 8, preventing OpenAI rate limit violations during large ingests. On failure, the original text is returned as a graceful fallback — the pipeline does not abort.

---

### Token-Aware Batching

The `_token_aware_batches()` function uses exact `tiktoken` (`cl100k_base`) token counts to build batches that satisfy all three OpenAI limits simultaneously:

1. **Per-document:** ≤ 8,192 tokens (hard model limit for `text-embedding-3-small`)
2. **Per-batch total:** ≤ 200,000 tokens (conservative budget below the 1M-token API limit)
3. **Per-batch count:** ≤ 2,048 documents (OpenAI hard request limit)

Documents exceeding the per-doc limit are truncated token-exactly — the truncated token list is decoded back to a string to avoid mid-codepoint splits.

---

### Embedding

`embed_texts()` sends each batch to `text-embedding-3-small` and retries on failure:

- Up to **3 attempts** per batch.
- **Exponential back-off:** 1s, 2s, 4s between retries.
- Raises `RuntimeError` if all 3 attempts fail, halting the pipeline.
- Logs per-batch progress including batch index, doc count, and token count.

---

### Qdrant Upsert

`upsert_to_qdrant()` generates point IDs using `uuid.uuid5(uuid.NAMESPACE_DNS, articleHashId)`. This means:

- The same article always maps to the same UUID.
- Re-running ingest upserts (overwrites) rather than duplicating.
- No need to track which articles have been processed.

Points are upserted in chunks of 200 to avoid large single requests to Qdrant.

Each point's payload stores:
```json
{
  "articleHashId": "<original hash>",
  "articleSource": "ask"
}
```

The `articleSource` payload field enables filtered ANN search in the `/search` endpoint.

---

## Data Models

### `IngestResponse`
```python
class IngestResponse(BaseModel):
    fetched: int           # docs pulled from Elasticsearch
    embedded: int          # docs successfully embedded
    upserted: int          # points written to Qdrant
    duration_seconds: float
```

### `SearchRequest`
```python
class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
```

### `SearchHit`
```python
class SearchHit(BaseModel):
    articleHashId: str
    score: float
    articleBigContent: str | None = None
    articleInsertedDate: str | None = None
```

### `SearchResponse`
```python
class SearchResponse(BaseModel):
    results: list[SearchHit]
    query_time_ms: float
```

---

## Error Handling & Resilience

| Scenario | Behaviour |
|---|---|
| Translation API failure | Logs warning, returns original (untranslated) text — pipeline continues |
| Embedding batch failure | Retries up to 3× with exponential back-off; raises `RuntimeError` after all attempts |
| No articles found | Endpoint returns `HTTP 404` with descriptive message |
| Duplicate ingest | Deterministic UUID5 ensures Qdrant upsert is idempotent |
| Qdrant collection missing | Created automatically at startup with cosine distance metric |
| Elasticsearch scroll | `clear_scroll` called in a `finally` block to avoid cursor leaks |

---

## Dependencies

```
fastapi
uvicorn
elasticsearch[async]
openai
qdrant-client
tiktoken
emoji
python-dotenv
pydantic
```

Install with:
```bash
pip install fastapi uvicorn "elasticsearch[async]" openai qdrant-client tiktoken emoji python-dotenv
```

---

## Environment Variables

Create a `.env` file in the project root:

```env
# Elasticsearch
ES_ENDPOINT=https://your-es-host:9200
ES_API_KEY=your_es_api_key
ES_INDEX=articles
ELASTIC_CERT=/path/to/ca.crt

# OpenAI
OPENAI_API_KEY=sk-...

# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

---

## Running the Service

```bash
# Development
uvicorn pipeline:app --reload --host 0.0.0.0 --port 8000

# Production
python pipeline.py
```

### Typical usage sequence

```bash
# 1. One-time full ingest (can take several minutes for large corpora)
curl -X POST http://localhost:8000/full_ingest

# 2. Daily incremental ingest (schedule via cron)
# Example cron: run at 01:00 UTC every day
# 0 1 * * * curl -X POST http://localhost:8000/today_ingest

# 3. Semantic search
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "best Python frameworks for async APIs", "top_k": 5}'

# 4. Health check
curl http://localhost:8000/health
```
