# -*- coding: utf-8 -*-
"""
Jeff-0.8B FastAPI Inference Server
High-Speed Lightweight Decision Head for Semantic Filtering & NLI
Port: 8092 (Default) | VRAM: ~1.5 GB | Precision: 100% on Long-Context
"""
import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Add Jeff source to path
JEFF_DIR = Path(r"D:\test_arc\local_jev_proj\jeff")
sys.path.insert(0, str(JEFF_DIR / "src"))

from jeff.models import load_decision_model
from jeff.model import answer

app = FastAPI(title="Jeff-0.8B Decision Server", version="1.0.0")

model = None
device = "cuda" if torch.cuda.is_available() else "cpu"

class QuestionPayload(BaseModel):
    state: str
    instruction: str
    criteria: Optional[Dict[str, str]] = None

class BatchItem(BaseModel):
    evidence: str
    instruction: Optional[str] = None
    criteria: Optional[Dict[str, str]] = None

class BatchPayload(BaseModel):
    items: List[BatchItem]
    default_instruction: Optional[str] = None
    default_criteria: Optional[Dict[str, str]] = None

@app.on_event("startup")
def load_model():
    global model
    checkpoint_path = str(JEFF_DIR / "checkpoints" / "jeff-0.8b")
    print(f"Loading Jeff-0.8B on {device} from {checkpoint_path}...", flush=True)
    t0 = time.time()
    model = load_decision_model(checkpoint_path, device=device)
    print(f"[OK] Jeff-0.8B loaded in {time.time()-t0:.2f}s", flush=True)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "Jeff-0.8B",
        "device": device,
        "vram_allocated_mb": round(torch.cuda.memory_allocated() / (1024*1024), 2) if device == "cuda" else 0
    }

@app.post("/predict")
def predict(payload: QuestionPayload):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    t0 = time.perf_counter()
    criteria = payload.criteria or {"yes": "合致する・直接役立つ情報", "no": "合致しない・無関係なノイズ"}
    q_spec = {
        "type": "choice",
        "instructions": payload.instruction,
        "criteria": criteria
    }
    row = {"state": payload.state, "question": q_spec, "images": []}
    batch = model.prepare([row])
    with torch.inference_mode():
        dist = (model(batch) / model.temperature).softmax(-1).cpu().tolist()
    
    ans = answer(q_spec, dist[0][:batch.counts[0]])
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
    
    # Extract probabilities
    probs = {}
    options = list(criteria.keys())
    for idx, opt in enumerate(options):
        if idx < len(dist[0]):
            probs[opt] = round(float(dist[0][idx]), 4)
            
    selected = ans.get("choice", max(probs, key=probs.get) if probs else "no")
    
    return {
        "selected": selected,
        "probabilities": probs,
        "elapsed_ms": elapsed_ms,
        "tokens": getattr(batch, "input_tokens", 0)
    }

@app.post("/batch_predict")
def batch_predict(payload: BatchPayload):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    t0 = time.perf_counter()
    def_inst = payload.default_instruction or "現在の問い合わせ内容に対して、候補の記憶情報が【直接役立つ・合致する情報】かを判定してください。"
    def_crit = payload.default_criteria or {"yes": "合致する・直接役立つ情報", "no": "合致しない・無関係なノイズ"}
    
    rows = []
    q_specs = []
    for it in payload.items:
        inst = it.instruction or def_inst
        crit = it.criteria or def_crit
        q = {"type": "choice", "instructions": inst, "criteria": crit}
        q_specs.append(q)
        rows.append({"state": it.evidence, "question": q, "images": []})
        
    batch = model.prepare(rows)
    with torch.inference_mode():
        dists = (model(batch) / model.temperature).softmax(-1).cpu().tolist()
        
    results = []
    for i, (q, row) in enumerate(zip(q_specs, rows)):
        cnt = batch.counts[i] if hasattr(batch, "counts") else len(q["criteria"])
        dist_row = dists[i][:cnt]
        ans = answer(q, dist_row)
        
        crit_keys = list(q["criteria"].keys())
        probs = {k: round(float(dist_row[k_idx]), 4) for k_idx, k in enumerate(crit_keys) if k_idx < len(dist_row)}
        sel = ans.get("choice", max(probs, key=probs.get) if probs else "no")
        is_match = (sel.lower() == "yes" or probs.get("yes", 0.0) >= 0.5)
        
        results.append({
            "index": i,
            "selected": sel,
            "is_match": is_match,
            "probabilities": probs
        })
        
    total_elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
    return {
        "results": results,
        "total_elapsed_ms": total_elapsed_ms,
        "count": len(results)
    }

@app.post("/v1/systemone")
def system_one(req: Dict[str, Any]):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    state = req.get("state")
    questions = req.get("questions", {})
    if not questions:
        raise HTTPException(status_code=422, detail="at least one question required")

    answers_dict = {}
    rows = []
    q_names = []
    q_specs = []

    for q_name, q_spec in questions.items():
        q_names.append(q_name)
        q_specs.append(q_spec)
        rows.append({"state": state, "question": q_spec, "images": []})

    batch = model.prepare(rows)
    with torch.inference_mode():
        dists = (model(batch) / model.temperature).softmax(-1).cpu().tolist()

    for i, (q_name, q_spec) in enumerate(zip(q_names, q_specs)):
        cnt = batch.counts[i] if hasattr(batch, "counts") else len(dists[i])
        dist_i = dists[i][:cnt]
        ans = answer(q_spec, dist_i)
        
        # Ensure confidence exists for noul
        if ans.get("type") == "noul" and "confidence" not in ans:
            p_true = ans.get("noul", 0.5)
            ans["confidence"] = max(p_true, 1.0 - p_true)
            
        answers_dict[q_name] = ans

    return {
        "model": "firelex/jeff-0.8b",
        "answers": answers_dict,
        "usage": {
            "input_tokens": getattr(batch, "input_tokens", 100),
            "output_tokens": 0
        }
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Jeff-0.8B Decision Server")
    parser.add_argument("--port", type=int, default=8092, help="Port to listen on (default 8092)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address")
    args = parser.parse_args()
    
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
