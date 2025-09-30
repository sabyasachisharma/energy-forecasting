
import os, json
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import faiss

EMBEDDINGS_MODEL = os.environ.get("EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
INDEX_PATH = os.environ.get("INDEX_PATH", "data/faiss.index")
META_PATH = os.environ.get("META_PATH", "data/meta.json")

_model = None
_index = None
_meta = None

def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDINGS_MODEL)
    return _model

def build_corpus_from_prices(df: pd.DataFrame, horizon_hours: int = 168) -> List[str]:
    lines = [f"{ts.isoformat()} price €{round(float(p),2)} per MWh" for ts, p in df["price"].items()]
    return lines

def build_index(df: pd.DataFrame, out_dir: str = "data"):
    os.makedirs(out_dir, exist_ok=True)
    corpus = build_corpus_from_prices(df)
    model = _get_model()
    embeddings = model.encode(corpus, convert_to_numpy=True, normalize_embeddings=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({"corpus": corpus}, f)
    faiss.write_index(index, os.path.join(out_dir, "faiss.index"))
    return True

def load_index(out_dir: str = "data"):
    global _index, _meta
    try:
        _index = faiss.read_index(os.path.join(out_dir, "faiss.index"))
        with open(os.path.join(out_dir, "meta.json"), "r") as f:
            _meta = json.load(f)
        return True
    except (FileNotFoundError, RuntimeError) as e:
        _index = None
        _meta = None
        return False

def retrieve(query: str, k: int = 5, out_dir: str = "data") -> List[str]:
    global _index, _meta
    if _index is None or _meta is None:
        if not load_index(out_dir):
            return [f"No index found. Please build the index first using POST /index/build endpoint."]
    
    model = _get_model()
    q = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    D, I = _index.search(q, k)
    corpus = _meta["corpus"]
    return [corpus[i] for i in I[0] if i < len(corpus)]
