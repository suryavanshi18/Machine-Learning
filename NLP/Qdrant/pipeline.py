"""
Pipeline:
  Elasticsearch (articleBigContent + articleHashId, source=reddit)
      ↓ clean text
      ↓ translate to English if Indic script detected (GPT-4o-mini)
      ↓ OpenAI embeddings (text-embedding-3-small, 1536-dim)
      ↓ Qdrant (vector + articleHashId payload)

FastAPI endpoints:
  POST /full_ingest   — one-time: all articles for source=reddit
  POST /today_ingest  — daily: only today's ingested articles
  POST /search        — embed query → Qdrant ANN → similar articleHashIds
  GET  /health        — liveness
"""

import os
import re
import time
import logging
import uuid
import asyncio
import uvicorn
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta

import tiktoken
import emoji
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from elasticsearch import AsyncElasticsearch
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue,
    NamedVector,
)

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────

ES_INDEX       = os.getenv("ES_INDEX", "articles")
QDRANT_HOST    = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT    = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION     = "articles_ask"
EMBED_MODEL    = "text-embedding-3-small"
EMBED_DIM      = 1536
ES_PAGE_SIZE          = 500      # docs per ES scroll page
ARTICLE_SOURCE        = "ask"
TRANSLATE_CONCURRENCY = 8        # max parallel GPT-4o-mini translation calls
MAX_TOKENS_PER_BATCH  = 200_000  # conservative — OpenAI adds per-item overhead on top of token count
MAX_TOKENS_PER_DOC    = 8192     # text-embedding-3-small/large hard per-doc limit

# ── Indic language detection ───────────────────────────────────────────────────

INDIC_PATTERN = re.compile(
    r'[\u0900-\u097F]|'   # Hindi / Marathi
    r'[\u0980-\u09FF]|'   # Bengali
    r'[\u0A00-\u0A7F]|'   # Punjabi
    r'[\u0A80-\u0AFF]|'   # Gujarati
    r'[\u0B00-\u0B7F]|'   # Odia
    r'[\u0B80-\u0BFF]|'   # Tamil
    r'[\u0C00-\u0C7F]|'   # Telugu
    r'[\u0C80-\u0CFF]|'   # Kannada
    r'[\u0D00-\u0D7F]'    # Malayalam
)

def needs_translation(text: str) -> bool:
    """Return True if the text contains any Indic script characters."""
    return bool(INDIC_PATTERN.search(text))

# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.es = AsyncElasticsearch(
        hosts=[os.getenv("ES_ENDPOINT")],
        api_key=os.getenv("ES_API_KEY"),
        ca_certs=os.getenv("ELASTIC_CERT"),
        request_timeout=30,
    )
    app.state.openai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    app.state.qdrant = AsyncQdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    existing = [c.name for c in (await app.state.qdrant.get_collections()).collections]
    if COLLECTION not in existing:
        await app.state.qdrant.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )
        log.info("Created Qdrant collection '%s'", COLLECTION)

    log.info("All clients ready")
    yield
    await app.state.es.close()
    await app.state.qdrant.close()

app = FastAPI(title="Article Embedding Pipeline", lifespan=lifespan)

# ── Text cleaning ──────────────────────────────────────────────────────────────

def clean_text(raw: str) -> str:
    """
    Clean noise while preserving Indic characters (needed for translation).
      1. Remove emojis
      2. Remove URLs
      3. Remove control characters only (NOT full non-ASCII — Indic scripts are non-ASCII)
      4. Collapse extra spaces and newlines
    """
    if not raw:
        return ""
    text = emoji.replace_emoji(raw, replace="")
    text = re.sub(r"https?://\S+|www\.\S+", "", text)  # URLs
    text = re.sub(r"[\x00-\x1F\x7F]", " ", text)       # control chars only
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def clean_text_post_translation(text: str) -> str:
    """
    After translation the text is pure English/ASCII.
    Now safe to remove any leftover non-printable artifacts.
    """
    text = re.sub(r"[^\x20-\x7E\n]", " ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()

# ── Translation (GPT-4o-mini) ──────────────────────────────────────────────────

async def translate_text(openai: AsyncOpenAI, text: str) -> str:
    """
    Translate text to English using GPT-4o-mini.
    Handles code-mixed text well (e.g. "आज Bangalore की flight late है").
    Only called when needs_translation() returns True — saves cost and latency.
    """
    try:
        resp = await openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Translate the following text to English. Return only the translated text, nothing else."
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            temperature=0,
            max_tokens=2048,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        log.warning("Translation failed (%s) — using original text", exc)
        return text  # fall back gracefully, don't break the pipeline


async def translate_if_needed(openai: AsyncOpenAI, text: str) -> str:
    """Translate only if Indic script is detected — skip API call for English text."""
    if needs_translation(text):
        return await translate_text(openai, text)
    return text


async def translate_batch(openai: AsyncOpenAI, texts: list[str]) -> list[str]:
    """
    Translate a list of texts concurrently, bounded by TRANSLATE_CONCURRENCY
    to avoid overwhelming the OpenAI rate limit.
    Texts already in English are returned as-is (no API call made).
    """
    sem = asyncio.Semaphore(TRANSLATE_CONCURRENCY)

    async def _limited(text: str) -> str:
        async with sem:
            return await translate_if_needed(openai, text)

    results = await asyncio.gather(*[_limited(t) for t in texts])
    return list(results)

# ── Elasticsearch helpers ──────────────────────────────────────────────────────

def _build_query(extra_filters: list | None = None) -> dict:
    """Build ES bool query — always filtered to ARTICLE_SOURCE."""
    must = [{"term": {"articleSource.keyword": ARTICLE_SOURCE}}]
    if extra_filters:
        must.extend(extra_filters)
    return {
        "query": {"bool": {"must": must}},
        "_source": ["articleHashId", "articleBigContent", "articleInsertedDate"],
        "size": ES_PAGE_SIZE,
    }


async def _scroll_docs(es: AsyncElasticsearch, body: dict) -> list[dict]:
    """Generic scroll loop — cleans text, skips near-empty docs."""
    docs      = []
    resp      = await es.search(index=ES_INDEX, body=body, scroll="5m")
    scroll_id = resp["_scroll_id"]

    try:
        while True:
            hits = resp["hits"]["hits"]
            if not hits:
                break
            for hit in hits:
                src   = hit["_source"]
                clean = clean_text(src.get("articleBigContent") or "")
                if len(clean) < 20:
                    continue
                docs.append({
                    "articleHashId":       src.get("articleHashId", hit["_id"]),
                    "articleInsertedDate": src.get("articleInsertedDate"),
                    "text":                clean,
                })
            log.info("Scrolled %d docs so far...", len(docs))
            resp      = await es.scroll(scroll_id=scroll_id, scroll="5m")
            scroll_id = resp["_scroll_id"]
    finally:
        await es.clear_scroll(scroll_id=scroll_id)

    log.info("Fetch complete: %d usable articles", len(docs))
    return docs


async def fetch_all_articles(es: AsyncElasticsearch) -> list[dict]:
    """Fetch ALL articles for source=reddit (one-time full ingest)."""
    return await _scroll_docs(es, _build_query())


async def fetch_articles_today(es: AsyncElasticsearch) -> list[dict]:
    """Fetch only articles inserted today UTC (daily incremental ingest)."""
    now          = datetime.now(timezone.utc)
    start        = now.replace(hour=0, minute=0, second=0, microsecond=0)
    end          = start + timedelta(days=1)
    start_epoch  = int(start.timestamp() * 1000)
    end_epoch    = int(end.timestamp() * 1000)
    date_filter  = {"range": {"articleInsertedDate": {"gte": start_epoch, "lt": end_epoch}}}
    return await _scroll_docs(es, _build_query(extra_filters=[date_filter]))

# ── OpenAI batch embed ─────────────────────────────────────────────────────────

# tiktoken encoder — cl100k_base is used by all text-embedding-3-* models
_enc = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_enc.encode(text, disallowed_special=()))


def _truncate_doc(text: str) -> str:
    """
    Hard-truncate a single document to MAX_TOKENS_PER_DOC using exact token counts.
    Called on every text AFTER translation so translated content is also checked.
    Decodes the truncated token list back to a string to avoid mid-token splits.
    """
    tokens = _enc.encode(text, disallowed_special=())
    if len(tokens) <= MAX_TOKENS_PER_DOC:
        return text
    log.debug("Truncating doc from %d tokens to %d", len(tokens), MAX_TOKENS_PER_DOC)
    return _enc.decode(tokens[:MAX_TOKENS_PER_DOC])


def _token_aware_batches(texts: list[str]) -> list[list[str]]:
    """
    Build batches that satisfy BOTH OpenAI limits:
      1. Per-doc  : each text ≤ MAX_TOKENS_PER_DOC  (8 192 tokens)
      2. Per-batch: total tokens ≤ MAX_TOKENS_PER_BATCH (250 000 tokens)
      3. Per-batch: ≤ 2048 texts (OpenAI hard request limit)

    Uses exact tiktoken counts — no estimation.
    """
    batches      = []
    current      = []
    current_toks = 0

    for text in texts:
        text      = _truncate_doc(text)          # enforce per-doc limit first
        doc_toks  = _count_tokens(text)

        if current and (
            current_toks + doc_toks > MAX_TOKENS_PER_BATCH
            or len(current) >= 2048
        ):
            batches.append(current)
            current      = []
            current_toks = 0

        current.append(text)
        current_toks += doc_toks

    if current:
        batches.append(current)

    return batches


async def embed_texts(openai: AsyncOpenAI, texts: list[str]) -> list[list[float]]:
    """
    Embed all texts in token-aware batches (max 250k tokens per request).
    Each batch retried up to 3 times with exponential back-off.
    """
    all_vecs = []
    batches  = _token_aware_batches(texts)
    total    = len(texts)
    done     = 0

    log.info("Embedding %d texts across %d token-aware batches", total, len(batches))

    for batch_idx, batch in enumerate(batches):
        for attempt in range(3):
            try:
                resp = await openai.embeddings.create(model=EMBED_MODEL, input=batch)
                all_vecs.extend([d.embedding for d in resp.data])
                done += len(batch)
                log.info(
                    "Embedded batch %d/%d (%d texts, %d tokens) — %d/%d total",
                    batch_idx + 1, len(batches),
                    len(batch),
                    sum(_count_tokens(t) for t in batch),
                    done, total,
                )
                break
            except Exception as exc:
                wait = 2 ** attempt
                log.warning(
                    "Embed batch %d attempt %d failed (%s) — retry in %ds",
                    batch_idx + 1, attempt + 1, exc, wait,
                )
                await asyncio.sleep(wait)
        else:
            raise RuntimeError(
                f"Embedding batch {batch_idx + 1} failed after 3 attempts"
            )

    return all_vecs

# ── Qdrant upsert ──────────────────────────────────────────────────────────────

async def upsert_to_qdrant(
    qdrant: AsyncQdrantClient,
    docs: list[dict],
    vectors: list[list[float]],
) -> int:
    """Deterministic UUIDs from articleHashId — safe to re-run without duplicates."""
    points = [
        PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_DNS, str(doc["articleHashId"]))),
            vector=vec,
            payload={
                "articleHashId": doc["articleHashId"],
                "articleSource": ARTICLE_SOURCE,
            },
        )
        for doc, vec in zip(docs, vectors)
    ]
    for i in range(0, len(points), 200):
        await qdrant.upsert(collection_name=COLLECTION, points=points[i : i + 200])
        log.info("Upserted %d / %d", min(i + 200, len(points)), len(points))
    return len(points)

# ── Shared pipeline logic ──────────────────────────────────────────────────────

async def run_pipeline(req: Request, docs: list[dict]) -> "IngestResponse":
    """
    Shared by both ingest endpoints:
      clean → translate (if Indic) → post-clean → embed → upsert
    """
    t0 = time.perf_counter()
    openai = req.app.state.openai

    # Count how many actually need translation (for logging)
    needs_tr = sum(1 for d in docs if needs_translation(d["text"]))
    log.info("%d / %d articles need translation", needs_tr, len(docs))

    # Translate (skips API call for pure-English articles)
    translated = await translate_batch(openai, [d["text"] for d in docs])

    # Post-translation clean — strip any leftover non-ASCII artifacts
    final_texts = [clean_text_post_translation(t) for t in translated]

    # Embed
    vectors = await embed_texts(openai, final_texts)

    # Upsert
    n = await upsert_to_qdrant(req.app.state.qdrant, docs, vectors)

    return IngestResponse(
        fetched=len(docs),
        embedded=len(vectors),
        upserted=n,
        duration_seconds=round(time.perf_counter() - t0, 2),
    )

# ── Pydantic schemas ───────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    fetched: int
    embedded: int
    upserted: int
    duration_seconds: float

class SearchRequest(BaseModel):
    query: str
    top_k: int = 10

class SearchHit(BaseModel):
    articleHashId: str
    score: float
    articleBigContent: str | None = None
    articleInsertedDate: str | None = None

class SearchResponse(BaseModel):
    results: list[SearchHit]
    query_time_ms: float

# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/full_ingest", response_model=IngestResponse)
async def full_ingest(req: Request):
    """
    ONE-TIME full pipeline: fetch all source=reddit articles,
    translate Indic content to English, embed, upsert to Qdrant.
    Safe to re-run — deterministic IDs prevent duplicates.
    """
    docs = await fetch_all_articles(req.app.state.es)
    if not docs:
        raise HTTPException(404, "No articles found for source=reddit")
    return await run_pipeline(req, docs)


@app.post("/today_ingest", response_model=IngestResponse)
async def today_ingest(req: Request):
    """
    DAILY pipeline: only articles inserted today (UTC).
    Schedule via cron during downtime.
    """
    docs = await fetch_articles_today(req.app.state.es)
    if not docs:
        raise HTTPException(404, "No articles ingested today for source=reddit")
    return await run_pipeline(req, docs)


async def fetch_articles_by_hash_ids(
    es: AsyncElasticsearch,
    hash_ids: list[str],
) -> dict[str, dict]:
    """
    Fetch full article details from Elasticsearch for a list of articleHashIds.
    Returns a dict keyed by articleHashId for O(1) lookup.
    """
    if not hash_ids:
        return {}

    resp = await es.search(
        index=ES_INDEX,
        body={
            "query": {
                "terms": {"articleHashId.keyword": hash_ids}
            },
            "_source": ["articleHashId", "articleBigContent", "articleInsertedDate"],
            "size": len(hash_ids),
        }
    )

    return {
        hit["_source"]["articleHashId"]: hit["_source"]
        for hit in resp["hits"]["hits"]
        if "articleHashId" in hit["_source"]
    }


@app.post("/search", response_model=SearchResponse)
async def search(req: Request, body: SearchRequest):
    """
    1. Translate query if Indic script detected
    2. Embed query → Qdrant ANN → top-k articleHashIds + scores
    3. Fetch full article details from Elasticsearch for those IDs
    4. Return merged results
    """
    t0     = time.perf_counter()
    openai = req.app.state.openai
    es     = req.app.state.es

    # Step 1: translate query if needed
    query_text = await translate_if_needed(openai, body.query)

    # Step 2: embed + Qdrant ANN search
    resp = await openai.embeddings.create(model=EMBED_MODEL, input=[query_text])
    query_vec = resp.data[0].embedding

    hits = await req.app.state.qdrant.query_points(
        collection_name=COLLECTION,
        query=query_vec,
        limit=body.top_k,
        query_filter=Filter(
            must=[FieldCondition(
                key="articleSource",
                match=MatchValue(value=ARTICLE_SOURCE),
            )]
        ),
        with_payload=True,
    )

    # Step 3: fetch ES details for matched IDs
    hash_ids    = [h.payload["articleHashId"] for h in hits.points]
    es_articles = await fetch_articles_by_hash_ids(es, hash_ids)

    # Step 4: merge Qdrant score with ES content, preserve Qdrant ranking order
    results = []
    for h in hits.points:
        article_id = h.payload["articleHashId"]
        es_data    = es_articles.get(article_id, {})
        results.append(SearchHit(
            articleHashId=article_id,
            score=round(h.score, 4),
            articleBigContent=es_data.get("articleBigContent"),
            articleInsertedDate=str(es_data.get("articleInsertedDate", "")),
        ))

    return SearchResponse(
        results=results,
        query_time_ms=round((time.perf_counter() - t0) * 1000, 2),
    )


if __name__ == "__main__":
    uvicorn.run("pipeline:app", host="0.0.0.0", port=8000)