from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
import asyncio
import uvicorn
import torch
from schemes import NERRequest, NERResponse, BatchNERRequest, BatchNERResponse

# ============================================================
# GLOBAL STATE
# ============================================================
model_pipeline = None
executor       = ThreadPoolExecutor(max_workers=4)  # tune to your CPU cores

# ============================================================
# LIFESPAN — load model ONCE at startup
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_pipeline
    print("Loading model at startup...")
    from model import NERPipeline
    model_pipeline = NERPipeline()   # loads GLiNER once
    print("Model ready. Service started.")
    yield
    print("Shutting down...")
    executor.shutdown(wait=False)

# ============================================================
# APP
# ============================================================
app = FastAPI(
    title="NER Inference Service",
    description="GLiNER-based NER for PERSON, ORG, LOC, EVENT, PRODUCT",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# SINGLE — async so server never blocks
# ============================================================
@app.post("/predict", response_model=NERResponse)
async def run_ner(request: NERRequest):
    try:
        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor,
            lambda: model_pipeline.predict(request.title or "", request.text)
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================
# BATCH — async
# ============================================================
@app.post("/predict/batch", response_model=BatchNERResponse)
async def run_ner_batch(request: BatchNERRequest):
    try:
        items  = [{"title": r.title or "", "text": r.text} for r in request.texts]
        loop   = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            executor,
            lambda: model_pipeline.predict_batch(items)
        )
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================
# HEALTH
# ============================================================
@app.get("/health")
async def health():
    return {
        "status":      "ok",
        "model_loaded": model_pipeline is not None,
        "device":      "cuda" if torch.cuda.is_available() else "cpu",
        "gpu_name":    torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    }

# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=7562,
        reload=False,
        workers=1        # keep 1 worker — model is GPU resident
    )
    

# sudo ufw allow 7562

