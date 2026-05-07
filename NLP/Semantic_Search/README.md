
# Myntra Semantic Search with Elasticsearch & Sentence Transformers

A semantic search pipeline built on the Myntra products catalog. Instead of traditional keyword matching, this project generates dense vector embeddings from product descriptions and stores them in Elasticsearch, enabling meaning-based search using kNN (k-Nearest Neighbors).

---

## How It Works

```
CSV Data
   │
   ▼
Pandas DataFrame  ──► fillna / clean
   │
   ▼
SentenceTransformer ('all-mpnet-base-v2')
   │  encodes Description column → 768-dim float vector
   ▼
Elasticsearch Index ('all_products')
   │  each document stores ProductID, ProductName, Description, DescriptionVector
   ▼
kNN Search query
   │  user query → encode → find top-k nearest vectors
   ▼
Results: ProductName + Description
```

---

## Tech Stack

| Component | Tool |
|---|---|
| Data | `myntra_products_catalog.csv` |
| DataFrame processing | `pandas` |
| Embedding model | `sentence-transformers` — `all-mpnet-base-v2` |
| Vector store & search | `Elasticsearch` (v8+) |
| Search type | kNN (Approximate Nearest Neighbor) |

---

## Setup

### Prerequisites

```bash
pip install elasticsearch pandas sentence-transformers
```

Elasticsearch must be running with HTTPS and basic auth. You will need:
- Elasticsearch endpoint (e.g. `https://192.168.x.x:9200`)
- Username and password
- CA certificate PEM file (`elasticsearch-ca.pem`)

### Connect to Elasticsearch

```python
from elasticsearch import Elasticsearch

es = Elasticsearch(
    "https://<your-host>:9200",
    basic_auth=("your_user", "password"),
    ca_certs="path_to_cert/elasticsearch-ca.pem",
    max_retries=10,
    retry_on_timeout=True
)

es.ping()  # should return True
```

---

## Pipeline Steps

### 1. Load & Clean Data

```python
df = pd.read_csv("myntra_products_catalog.csv")
df.fillna("None", inplace=True)
```

### 2. Generate Embeddings

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-mpnet-base-v2')
df["DescriptionVector"] = list(model.encode(df["Description"].tolist(), show_progress_bar=True))
```

Each product description is encoded into a **768-dimensional float vector** that captures semantic meaning.

### 3. Create Elasticsearch Index

```python
from indexMapping import indexMapping

es.indices.create(index="all_products", mappings=indexMapping)
```

The `indexMapping` (defined in `indexMapping.py`) must declare `DescriptionVector` as a `dense_vector` field with `dims: 768` and `index: true` to support kNN search.

Example mapping for the vector field:

```json
{
  "DescriptionVector": {
    "type": "dense_vector",
    "dims": 768,
    "index": true,
    "similarity": "cosine"
  }
}
```

### 4. Index Documents

```python
record_list = df.to_dict("records")

for record in record_list:
    try:
        es.index(index="all_products", document=record, id=record["ProductID"])
    except Exception as e:
        print(e)
```

---

## Search

### Encode the Query

```python
input_keyword = "Blue Shoes"
vector_of_input = model.encode(input_keyword)
```

### Run kNN Search

```python
query = {
    "field": "DescriptionVector",
    "query_vector": vector_of_input,
    "k": 2,
    "num_candidates": 500
}

res = es.knn_search(index="all_products", knn=query, source=["ProductName", "Description"])
print(res["hits"]["hits"])
```

`k=2` returns the 2 most semantically similar products. `num_candidates=500` controls the ANN candidate pool — higher values improve recall at the cost of speed.

---

## BM25 vs kNN — When to Use What

| Feature | BM25 (Keyword Search) | kNN (Semantic Search) |
|---|---|---|
| Type | Exact token matching | Meaning-based matching |
| Input | Text tokens | Dense vectors |
| Speed | Very fast | Slower |
| Accuracy | Exact match | Handles synonyms, paraphrases |
| Infrastructure | Simple | Requires embedding model |
| Best for | SKU / ID lookup, filters | Natural language queries |

---

## Limitations of Semantic Search

| Limitation | Why It Matters |
|---|---|
| Token limit on encoder | Long descriptions lose tail context |
| Semantic fuzziness | Poor for exact match queries (e.g. product codes) |
| Vector compression | Quantization introduces information loss |
| No term interaction | Weaker than BM25 for precise field ranking |
| Compute cost | Encoding at scale requires GPU / batching |

---

## Advanced: Nested Document Search

For documents with nested array fields (e.g. product images), use Elasticsearch nested queries:

```json
{
  "query": {
    "nested": {
      "path": "images",
      "query": {
        "match": {
          "images.image_hash_name": "0ca7bd077a062ce9.jpg"
        }
      }
    }
  }
}
```

This is required when fields like `images` are mapped as `nested` type rather than plain `object`.

---

## File Structure

```
.
├── myntra.ipynb              # Main notebook — full pipeline
├── indexMapping.py           # Elasticsearch index mapping definition
├── myntra_products_catalog.csv  # Source data
└── README.md
```

---

## Notes

- Replace `192.168.yyy.XX`, credentials, and cert path with your actual Elasticsearch cluster details before running.
- The embedding step can be slow on CPU for large catalogs — consider batching or running on GPU.
- `all-mpnet-base-v2` produces 768-dim vectors and is one of the stronger general-purpose sentence embedding models from SBERT.
