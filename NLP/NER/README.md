# 🔍 GLiNER NER Inference Service

A production-ready **Named Entity Recognition (NER)** microservice built on [GLiNER](https://github.com/urchade/GLiNER), FastAPI, and Docker. Designed for real-world social media text — multilingual, noisy, hashtag-heavy, emoji-filled.

Extracts **PERSON · ORG · LOCATION · EVENT · PRODUCT** entities with zero-shot flexibility: define any entity type in plain English at inference time, no retraining required.

---

## 📐 Architecture Overview

```
POST /predict or /predict/batch
        │
        ▼
┌─────────────────────────────────────┐
│         FastAPI (async)             │
│   ThreadPoolExecutor (4 workers)    │  ← event loop never blocks
└────────────────┬────────────────────┘
                 │ run_in_executor
                 ▼
┌─────────────────────────────────────┐
│         NERPipeline                 │
│                                     │
│  1. Metadata extraction             │  mentions · hashtags · quotes · URLs
│  2. Text cleaning                   │  strip URLs, @handles, #tags, emoji
│  3. Caps normalization              │  ALL-CAPS → Title Case
│  4. Auto-translation (EN)          │  chunked at 4500 chars
│  5. Hashtag reconstruction          │  CamelCase → readable tokens
│                                     │
│  → Sliding window NER               │  1000-char window, 200-char overlap
│  → Deduplication                    │  highest-confidence entity wins
│  → Grouped output                   │  per entity type
└─────────────────────────────────────┘
```

**Model:** `urchade/gliner_small-v2.1` — loaded once at startup, GPU-resident  
**Stack:** GLiNER · FastAPI · PyTorch · Docker · Elasticsearch (dense_vector index)

---

## 🚀 Quick Start

### Run with Docker

```bash
docker build -t ner-service .
docker run -p 7562:7562 --gpus all ner-service
```

### Run locally

```bash
pip install -r requirements.txt
python main.py
```

Service starts on `http://0.0.0.0:7562`

---

## 📡 API Endpoints

### `POST /predict` — single document

```bash
curl -X POST http://localhost:7562/predict \
  -H "Content-Type: application/json" \
  -d '{
    "title": "PM Modi inaugurates new airport in Pune",
    "text": "Prime Minister Narendra Modi today inaugurated the new Pune International Airport. #ModiInPune @BJP2024"
  }'
```

**Response:**
```json
{
  "original_text": "...",
  "translated_text": "...",
  "mentions": ["BJP2024"],
  "hashtags": ["ModiInPune"],
  "entities": [
    {"text": "Narendra Modi", "label": "PERSON", "score": 0.94},
    {"text": "Pune International Airport", "label": "LOC", "score": 0.88}
  ],
  "ner_persons":   ["Narendra Modi"],
  "ner_orgs":      [],
  "ner_locations": ["Pune International Airport"],
  "ner_events":    [],
  "ner_products":  []
}
```

### `POST /predict/batch` — multiple documents

```bash
curl -X POST http://localhost:7562/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      {"title": "Apple event", "text": "Apple launched the new iPhone 16 at their Cupertino campus."},
      {"title": "Cricket news", "text": "Virat Kohli smashes century against Australia at Eden Gardens."}
    ]
  }'
```

### `GET /health`

```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cuda",
  "gpu_name": "Quadro GV100"
}
```

---

## ⚙️ Key Engineering Decisions

### 1. Async-safe inference

FastAPI is async; PyTorch inference is synchronous and blocking. Calling the model directly in a route handler freezes the event loop under concurrent load.

**Fix:** `ThreadPoolExecutor` + `asyncio.run_in_executor()` — inference runs in a thread pool, the event loop stays free.

```python
result = await loop.run_in_executor(
    executor,
    lambda: model_pipeline.predict(request.title, request.text)
)
```

### 2. Model-as-singleton, one Uvicorn worker

The GLiNER model is loaded once at startup via FastAPI's `lifespan` context manager and kept resident in GPU memory. Multiple Uvicorn workers would fork the process — causing CUDA memory duplication or sharing errors.

**Architecture:** 1 worker + thread pool = safe concurrency without GPU conflicts.

### 3. Sliding window with overlap

GLiNER's BiLM encoder has a fixed context window. Long texts silently truncate — no error, just missing entities.

**Fix:** 1000-char sliding window with 200-char overlap. Span offsets are corrected back to the original document coordinate space. Deduplication by `(text.lower(), label)` keeps the highest-confidence extraction.

```
| window 1 (0–1000)     |
              | window 2 (800–1800)   |  ← 200-char overlap captures boundary entities
```

### 4. Hashtag entity recovery

`#BJP2024Rally` is a single OOV token to a subword tokenizer — NER extracts nothing. CamelCase splitting reconstructs it into recognizable words before inference.

```
#ModiInauguratesAirport  →  "Modi Inaugurates Airport"
                             ↑PERSON  ↑EVENT      ↑LOC
```

### 5. ALL-CAPS normalization

Twitter content like `NARENDRA MODI VISITS PUNE` disrupts GLiNER's subword tokenization patterns, significantly reducing recall. Title-casing before inference restores the expected input distribution.

```python
def normalize_caps(self, text):
    def cap_word(m):
        word = m.group(0)
        return word.title() if word.isupper() and len(word) > 1 else word
    return re.sub(r'\b\w+\b', cap_word, text)
```

### 6. Multilingual preprocessing

Indian social media mixes Hindi, Hinglish, Tamil, and regional scripts in the same post. GLiNER (trained on English Pile-NER) performs poorly on non-English input.

**Fix:** Auto-detect language → translate to English in 4500-char chunks → run NER on normalized text.

---

## 📂 Project Structure

```
.
├── main.py            # FastAPI app, lifespan, /predict, /predict/batch, /health
├── model.py           # NERPipeline — preprocessing + GLiNER inference
├── schemes.py         # Pydantic request/response models
├── indexMapping.py    # Elasticsearch dense_vector index mapping (768-dim, l2_norm)
├── requirements.txt
└── Dockerfile
```

---

## 🗃️ Elasticsearch Integration

The service output is designed for indexing into Elasticsearch alongside semantic embeddings:

```python
# indexMapping.py
"DescriptionVector": {
    "type": "dense_vector",
    "dims": 768,
    "index": True,
    "similarity": "l2_norm"
}
```

NER entities enriched at ingestion time enable faceted search, entity filtering, and hybrid keyword + semantic retrieval.

---

## ⚙️ Configuration

| Constant | Default | Description |
|---|---|---|
| `LABELS` | `PERSON, ORG, PRODUCT, EVENT, LOC` | Entity types to extract |
| `WINDOW_SIZE` | `1000` | Sliding window size (chars) |
| `OVERLAP` | `200` | Overlap between windows (chars) |
| `NER_THRESHOLD` | `0.5` | Minimum confidence score |
| `MAX_TEXT_LEN` | `4500` | Hard truncation before inference |
| `TRANSLATE_LIMIT` | `4500` | Max chars per translation API call |
| `ThreadPoolExecutor` | `max_workers=4` | Tune to available CPU cores |

---

## 🐳 Dockerfile Notes

```dockerfile
# requirements installed before code copy — rebuild after code changes
# skips dependency reinstall (layer cache hit)
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
```

For GPU support, use `--gpus all` with Docker or configure `nvidia-container-toolkit`.

To open the firewall port on the host:
```bash
sudo ufw allow 7562
```

---

## ⚠️ Known Limitations

- **Noisy/informal text** — slang, abbreviations, and non-standard grammar reduce precision. SpaCy rule-based matchers may complement GLiNER for such input.
- **Complex financial entities** — structured identifiers like `NASDAQ: AAPL` and multi-clause spans are frequently missed.
- **Translation quality** — Hinglish and code-switched text translates imperfectly; downstream NER on translated output inherits those errors.
- **Low-resource languages** — languages absent from DeBERTa-v3 pretraining see reduced recall.
- **Span length** — enumerating all candidate spans for very long documents is memory-intensive without chunking.

---

## 📚 References

- [GLiNER: Generalist Model for NER using Bidirectional Transformer](https://arxiv.org/abs/2311.08526) — Zaratiana et al., NAACL 2024
- [urchade/gliner_small-v2.1](https://huggingface.co/urchade/gliner_small-v2.1) — Hugging Face model card
- [Pile-NER dataset](https://huggingface.co/datasets/Universal-NER/Pile-NER-type) — training data
- Docker repo: https://hub.docker.com/repository/docker/abhs9/systemdgli/general

