# -*- coding: utf-8 -*-
"""
Full Evaluator for Jeff-0.8B and CLM-8B:
1. Phase 2: Long-Context Benchmark (15 items)
2. Phase 3: Pragmatic Logic & Negation Traps Benchmark (20 items)

Generates merged evaluation reports and updates phase2_results.md & phase3_results.md.
"""
import os
import sys
import time
import json
import torch
from pathlib import Path

# Paths
BASE_DIR = Path(r"D:\test_arc\local_jev_proj")
P2_DATA_FILE = BASE_DIR / "phase2_long_context" / "long_context_benchmark.json"
P3_DATA_FILE = BASE_DIR / "phase3_pragmatic_logic" / "test_cases.json"
JEFF_DIR = BASE_DIR / "jeff"
sys.path.insert(0, str(JEFF_DIR / "src"))
sys.path.insert(0, str(BASE_DIR))

from jeff.models import load_decision_model
from jeff.model import answer
from clm_runner import CLMRunner

print("="*70, flush=True)
print("Initializing Jeff-0.8B and CLM-8B on RTX 3090...", flush=True)
print("="*70, flush=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Load Models
t0 = time.time()
jeff_model = load_decision_model(str(JEFF_DIR / "checkpoints" / "jeff-0.8b"), device=device)
print(f"Jeff-0.8B loaded in {time.time()-t0:.2f}s", flush=True)

t0 = time.time()
clm_runner = CLMRunner()
print(f"CLM-8B loaded in {time.time()-t0:.2f}s", flush=True)

# -------------------------------------------------------------
# PHASE 2: LONG-CONTEXT EVALUATION (15 ITEMS)
# -------------------------------------------------------------
print("\n" + "="*70, flush=True)
print("STARTING PHASE 2: LONG-CONTEXT EVALUATION (15 items)", flush=True)
print("="*70, flush=True)

p2_data = json.loads(P2_DATA_FILE.read_text(encoding="utf-8"))
docs = p2_data["documents"]
items = p2_data["items"]

def eval_phase2_jeff():
    print("\n--- Evaluating Jeff-0.8B on Phase 2 ---", flush=True)
    results = []
    correct = 0
    total_time = 0.0

    for idx, it in enumerate(items, 1):
        doc_text = docs[it["doc"]]
        q_spec = it["question"]
        gt = it["ground_truth"]
        q_id = it["id"]

        q_jeff = {
            "type": q_spec["type"],
            "instructions": q_spec.get("instructions", ""),
            "criteria": q_spec.get("criteria", {})
        }
        row = {"state": doc_text, "question": q_jeff, "images": []}

        t0 = time.perf_counter()
        batch = jeff_model.prepare([row])
        with torch.inference_mode():
            dist = (jeff_model(batch) / jeff_model.temperature).softmax(-1).cpu().tolist()
        ans = answer(q_jeff, dist[0][:batch.counts[0]])
        latency = time.perf_counter() - t0
        total_time += latency

        pred = None
        conf = ans.get("confidence", 0.0)
        if q_spec["type"] == "noul":
            noul_val = ans.get("noul")
            if isinstance(noul_val, bool):
                pred = noul_val
            elif isinstance(noul_val, (int, float)):
                pred = (noul_val >= 0.5)
            else:
                pred = False
        elif q_spec["type"] == "choice":
            pred = ans.get("choice")

        is_corr = (pred == gt)
        if is_corr:
            correct += 1

        mark = "✓" if is_corr else "✗"
        print(f"  [{idx:02d}/{len(items)}] {mark} {q_id}: pred={pred}, gt={gt} ({latency:.2f}s, conf={conf:.2f})", flush=True)
        results.append({
            "id": q_id,
            "pred": pred,
            "ground_truth": gt,
            "correct": is_corr,
            "confidence": conf,
            "latency": latency
        })

    acc = correct / len(items) if items else 0.0
    mean_lat = total_time / len(items) if items else 0.0
    print(f"--> Jeff-0.8B Phase 2: {correct}/{len(items)} ({acc*100:.1f}%), Avg Latency: {mean_lat:.2f}s, Total: {total_time:.1f}s", flush=True)
    return {"correct": correct, "total": len(items), "accuracy": acc, "mean_latency": mean_lat, "total_time": total_time, "items": results}

def eval_phase2_clm():
    print("\n--- Evaluating CLM-8B on Phase 2 ---", flush=True)
    results = []
    correct = 0
    total_time = 0.0

    for idx, it in enumerate(items, 1):
        doc_text = docs[it["doc"]]
        q_spec = it["question"]
        gt = it["ground_truth"]
        q_id = it["id"]

        evidence = doc_text
        criterion = q_spec.get("instructions", "")
        options = q_spec.get("criteria", {"true": "はい", "false": "いいえ"})

        t0 = time.perf_counter()
        clm_res = clm_runner.predict_choice(evidence, criterion, options)
        latency = time.perf_counter() - t0
        total_time += latency

        selected = clm_res["selected"]
        probs = clm_res["probabilities"]
        conf = probs.get(selected, 0.0)

        pred = (selected.lower() == "true")
        is_corr = (pred == gt)
        if is_corr:
            correct += 1

        mark = "✓" if is_corr else "✗"
        print(f"  [{idx:02d}/{len(items)}] {mark} {q_id}: pred={pred}, gt={gt} ({latency:.2f}s, conf={conf:.2f})", flush=True)
        results.append({
            "id": q_id,
            "pred": pred,
            "ground_truth": gt,
            "correct": is_corr,
            "confidence": conf,
            "latency": latency
        })

    acc = correct / len(items) if items else 0.0
    mean_lat = total_time / len(items) if items else 0.0
    print(f"--> CLM-8B Phase 2: {correct}/{len(items)} ({acc*100:.1f}%), Avg Latency: {mean_lat:.2f}s, Total: {total_time:.1f}s", flush=True)
    return {"correct": correct, "total": len(items), "accuracy": acc, "mean_latency": mean_lat, "total_time": total_time, "items": results}

p2_jeff = eval_phase2_jeff()
p2_clm = eval_phase2_clm()

# Save Phase 2 Raw Results
p2_out_file = BASE_DIR / "phase2_long_context" / "results_jeff_clm.json"
p2_out_file.write_text(json.dumps({"Jeff-0.8B": p2_jeff, "CLM-8B": p2_clm}, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Saved Phase 2 Jeff/CLM results to {p2_out_file}", flush=True)

# -------------------------------------------------------------
# PHASE 3: PRAGMATIC LOGIC EVALUATION (20 ITEMS)
# -------------------------------------------------------------
print("\n" + "="*70, flush=True)
print("STARTING PHASE 3: PRAGMATIC LOGIC EVALUATION (20 items)", flush=True)
print("="*70, flush=True)

p3_cases = json.loads(P3_DATA_FILE.read_text(encoding="utf-8"))

def eval_phase3_jeff():
    print("\n--- Evaluating Jeff-0.8B on Phase 3 ---", flush=True)
    results = []
    correct = 0
    total_time = 0.0

    for idx, tc in enumerate(p3_cases, 1):
        q_jeff = {
            "type": "choice",
            "instructions": tc["instructions"],
            "criteria": tc["criteria"]
        }
        row = {"state": tc["state"], "question": q_jeff, "images": []}

        t0 = time.perf_counter()
        batch = jeff_model.prepare([row])
        with torch.inference_mode():
            dist = (jeff_model(batch) / jeff_model.temperature).softmax(-1).cpu().tolist()
        ans = answer(q_jeff, dist[0][:batch.counts[0]])
        latency = time.perf_counter() - t0
        total_time += latency

        choice = str(ans.get("choice", ""))
        conf = float(ans.get("confidence", 0.0))
        is_corr = (choice == str(tc["expected"]))
        if is_corr:
            correct += 1

        mark = "✓" if is_corr else "✗"
        print(f"  [{idx:02d}/{len(p3_cases)}] {mark} {tc['id']}: pred={choice}, exp={tc['expected']} ({latency:.2f}s, conf={conf:.2f})", flush=True)
        results.append({
            "id": tc["id"],
            "category": tc["category"],
            "language": tc["language"],
            "expected": str(tc["expected"]),
            "choice": choice,
            "is_correct": is_corr,
            "confidence": conf,
            "latency": latency,
            "explanation": tc["explanation"]
        })

    acc = correct / len(p3_cases) if p3_cases else 0.0
    mean_lat = total_time / len(p3_cases) if p3_cases else 0.0
    print(f"--> Jeff-0.8B Phase 3: {correct}/{len(p3_cases)} ({acc*100:.1f}%), Avg Latency: {mean_lat:.2f}s, Total: {total_time:.1f}s", flush=True)
    return {"correct": correct, "total": len(p3_cases), "accuracy": acc, "mean_latency": mean_lat, "total_time": total_time, "items": results}

def eval_phase3_clm():
    print("\n--- Evaluating CLM-8B on Phase 3 ---", flush=True)
    results = []
    correct = 0
    total_time = 0.0

    for idx, tc in enumerate(p3_cases, 1):
        evidence = tc["state"]
        criterion = tc["instructions"]
        options = tc["criteria"]

        t0 = time.perf_counter()
        clm_res = clm_runner.predict_choice(evidence, criterion, options)
        latency = time.perf_counter() - t0
        total_time += latency

        selected = str(clm_res["selected"])
        conf = float(clm_res["probabilities"].get(selected, 0.0))
        is_corr = (selected == str(tc["expected"]))
        if is_corr:
            correct += 1

        mark = "✓" if is_corr else "✗"
        print(f"  [{idx:02d}/{len(p3_cases)}] {mark} {tc['id']}: pred={selected}, exp={tc['expected']} ({latency:.2f}s, conf={conf:.2f})", flush=True)
        results.append({
            "id": tc["id"],
            "category": tc["category"],
            "language": tc["language"],
            "expected": str(tc["expected"]),
            "choice": selected,
            "is_correct": is_corr,
            "confidence": conf,
            "latency": latency,
            "explanation": tc["explanation"]
        })

    acc = correct / len(p3_cases) if p3_cases else 0.0
    mean_lat = total_time / len(p3_cases) if p3_cases else 0.0
    print(f"--> CLM-8B Phase 3: {correct}/{len(p3_cases)} ({acc*100:.1f}%), Avg Latency: {mean_lat:.2f}s, Total: {total_time:.1f}s", flush=True)
    return {"correct": correct, "total": len(p3_cases), "accuracy": acc, "mean_latency": mean_lat, "total_time": total_time, "items": results}

p3_jeff = eval_phase3_jeff()
p3_clm = eval_phase3_clm()

# Save Phase 3 Raw Results
p3_out_file = BASE_DIR / "phase3_pragmatic_logic" / "results_jeff_clm.json"
p3_out_file.write_text(json.dumps({"Jeff-0.8B": p3_jeff, "CLM-8B": p3_clm}, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Saved Phase 3 Jeff/CLM results to {p3_out_file}", flush=True)

print("\n" + "="*70, flush=True)
print("ALL BENCHMARKS COMPLETED SUCCESSFULLY!", flush=True)
print("="*70, flush=True)
