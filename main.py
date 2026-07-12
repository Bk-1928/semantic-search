"""
Semantic Search -- Top-K Ranking API (Local & Free)
POST /rank

Run locally:
    uvicorn main:app --host 0.0.0.0 --port 8000

Submit: http://localhost:8000/rank
"""

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI(title="Local Semantic Ranking API")

# Load the model once when the server starts.
# This model runs locally and entirely for free!
MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)


class RankRequest(BaseModel):
    query_id: str
    query: str
    candidates: list[str]


def _cosine_sim(query_vec: np.ndarray, cand_vecs: np.ndarray) -> np.ndarray:
    # Normalize query vector
    q = query_vec / np.linalg.norm(query_vec)
    # Normalize candidate vectors along rows
    c = cand_vecs / np.linalg.norm(cand_vecs, axis=1, keepdims=True)
    # Matrix multiplication to get cosine similarities
    return c @ q


@app.post("/rank")
def rank(req: RankRequest):
    if not req.candidates:
        raise HTTPException(400, "candidates must be a non-empty list")

    # 1. Encode query and candidates using the local transformer model
    # Convert them to numpy arrays explicitly for the matrix operations
    query_vec = np.array(model.encode(req.query), dtype=np.float64)
    cand_vecs = np.array(model.encode(req.candidates), dtype=np.float64)

    # 2. Calculate similarities
    sims = _cosine_sim(query_vec, cand_vecs)
    
    # 3. Sort indices in descending order and pick top 3
    top3 = np.argsort(-sims)[:3].tolist()

    return {"ranking": [int(i) for i in top3]}


@app.get("/")
def health():
    return {"status": "ok", "model": MODEL_NAME, "endpoint": "/rank"}