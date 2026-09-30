# -*- coding: utf-8 -*-
"""
Kev-4B FastAPI Inference Server
High-Speed Decision Head for Tactical Evaluation
Thread-Safe Architecture with Model Mutex Lock
"""
import os
import sys
import time
import json
import torch
import uvicorn
import threading
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

# Environment configuration
os.environ["HF_HOME"] = "/mnt/d/Models/hf_cache"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, "/mnt/d/test_arc/local_jev_proj/kev")

from kev.checkpoint import Checkpoint, LoadOptions

app = FastAPI(title="Kev-4B Decision Head Server", version="1.0.0")

# Global model state & Mutex Lock
tok = None
model = None
model_lock = threading.Lock()

class QuestionItem(BaseModel):
    id: str
    instr: str
    options: List[str]
    label: int = 0

class EvaluateRequest(BaseModel):
    state: str
    questions: List[QuestionItem]

@app.on_event("startup")
def load_model():
    global tok, model
    print("=" * 60)
    print("Initializing Kev-4B model on CUDA (RTX 3090)...")
    print("=" * 60)
    t0 = time.perf_counter()
    ck = Checkpoint("jaredpalmer/kev-4b")
    opts = LoadOptions(dtype=torch.bfloat16, merge=True)
    tok, model = ck.load("cuda", opts)
    load_time = time.perf_counter() - t0
    print(f"Kev-4B loaded successfully in {load_time:.2f}s!")

    # Warmup inference
    print("Running warmup inference...")
    warmup_rec = {
        "state": "テスト状況。ボールは敵陣中央にある。",
        "questions": [
            {
                "instr": "味方は攻撃を継続すべきか？",
                "options": ["攻撃継続", "守備後退"],
                "label": 0
            }
        ]
    }
    with model_lock:
        enc = model.encode(tok, warmup_rec)
        for _ in range(3):
            _ = model.probs(enc)
            torch.cuda.synchronize()
    print("Warmup complete. Server is ready!")

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "jaredpalmer/kev-4b",
        "device": "cuda:0",
        "vram_allocated_gb": round(torch.cuda.memory_allocated() / (1024 ** 3), 2)
    }

@app.post("/evaluate")
def evaluate(req: EvaluateRequest):
    if not req.questions:
        raise HTTPException(status_code=400, detail="No questions provided")

    t0 = time.perf_counter()

    rec = {
        "state": req.state,
        "questions": [
            {
                "instr": q.instr,
                "options": q.options,
                "label": q.label
            }
            for q in req.questions
        ]
    }

    try:
        # Thread-safe GPU inference using Mutex Lock
        with model_lock:
            enc = model.encode(tok, rec)
            torch.cuda.synchronize()
            infer_t0 = time.perf_counter()
            raw_probs = model.probs(enc)
            torch.cuda.synchronize()
            infer_ms = (time.perf_counter() - infer_t0) * 1000.0

        results = {}
        for i, q in enumerate(req.questions):
            q_probs = [float(p) for p in raw_probs[i]]
            max_idx = max(range(len(q_probs)), key=lambda idx: q_probs[idx])
            results[q.id] = {
                "instr": q.instr,
                "options": q.options,
                "probs": q_probs,
                "best_idx": max_idx,
                "best_option": q.options[max_idx],
                "confidence": q_probs[max_idx]
            }

        total_latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "results": results,
            "latency_ms": round(total_latency_ms, 2),
            "infer_ms": round(infer_ms, 2),
            "num_questions": len(req.questions)
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok", "model": "jaredpalmer/kev-4b", "backend": "kev-4b"}

@app.post("/v1/systemone")
def system_one(req: Dict[str, Any]):
    state = req.get("state")
    questions = req.get("questions", {})
    if not questions:
        raise HTTPException(status_code=422, detail="at least one question required")

    state_str = json.dumps(state, ensure_ascii=False) if isinstance(state, (dict, list)) else str(state)
    answers = {}

    for q_name, q_spec in questions.items():
        q_type = q_spec.get("type", "noul")
        instr = q_spec.get("instructions") or ""
        criteria = q_spec.get("criteria", {})

        if q_type == "noul":
            options_list = ["false", "true"]
        elif q_type == "choice":
            options_list = list(criteria.keys())
        elif q_type == "score":
            options_list = [c if isinstance(c, str) else json.dumps(c, ensure_ascii=False) for c in criteria]
        else:
            options_list = ["false", "true"]

        rec = {
            "state": state_str,
            "questions": [{"instr": instr, "options": options_list, "label": 0}]
        }

        with model_lock:
            enc = model.encode(tok, rec)
            torch.cuda.synchronize()
            raw_probs = model.probs(enc)[0]
            torch.cuda.synchronize()

        probs_list = [float(p) for p in raw_probs]
        best_idx = max(range(len(probs_list)), key=lambda i: probs_list[i])
        confidence = probs_list[best_idx]

        if q_type == "noul":
            answers[q_name] = {
                "type": "noul",
                "noul": probs_list[1] if len(probs_list) > 1 else 0.5,
                "confidence": confidence
            }
        elif q_type == "choice":
            prob_map = {options_list[i]: round(p, 4) for i, p in enumerate(probs_list)}
            answers[q_name] = {
                "type": "choice",
                "choice": str(options_list[best_idx]),
                "probabilities": prob_map,
                "confidence": confidence
            }
        elif q_type == "score":
            prob_map = {i: round(p, 4) for i, p in enumerate(probs_list)}
            exp_score = sum(i * p for i, p in enumerate(probs_list))
            answers[q_name] = {
                "type": "score",
                "score": round(exp_score, 2),
                "probabilities": prob_map,
                "legend": {i: f"level {i}" for i in range(len(options_list))},
                "confidence": confidence
            }

    return {
        "model": "jaredpalmer/kev-4b",
        "answers": answers,
        "usage": {"input_tokens": 100, "output_tokens": 0}
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8090, log_level="info")
