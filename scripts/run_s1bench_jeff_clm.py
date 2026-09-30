# -*- coding: utf-8 -*-
"""
Evaluate Jeff-0.8B and CLM-8B on S1Bench (6 subsets, 50 items each, 300 items total)
Subsets:
- boolq (noul / reading comprehension)
- paws (noul / paraphrase adversarial)
- vitaminc-dev (choice / fact verification)
- massive-en-US (choice / 60 intents classification)
- aegis2 (noul / safety guardrail)
- helpsteer2 (score / response quality ranking)
"""
import os
import sys
import time
import json
import torch
from pathlib import Path

BASE_DIR = Path(r"D:\test_arc\local_jev_proj")
DATA_DIR = BASE_DIR / "s1bench_data"
JEFF_DIR = BASE_DIR / "jeff"
sys.path.insert(0, str(JEFF_DIR / "src"))
sys.path.insert(0, str(BASE_DIR))

from jeff.models import load_decision_model
from jeff.model import answer
from clm_runner import CLMRunner

SUBSETS = [
    "boolq",
    "paws",
    "vitaminc-dev",
    "massive-en-US",
    "aegis2",
    "helpsteer2",
]

LIMIT_PER_SUBSET = 50

print("="*70, flush=True)
print("Loading Models for S1Bench Evaluation...", flush=True)
print("="*70, flush=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
t0 = time.time()
jeff_model = load_decision_model(str(JEFF_DIR / "checkpoints" / "jeff-0.8b"), device=device)
print(f"Jeff-0.8B loaded in {time.time()-t0:.2f}s", flush=True)

t0 = time.time()
clm_runner = CLMRunner()
print(f"CLM-8B loaded in {time.time()-t0:.2f}s", flush=True)

def eval_subset(name, model_type):
    file_path = DATA_DIR / f"{name}.json"
    data = json.loads(file_path.read_text(encoding="utf-8"))
    q_dict = data["question"]
    q_type = data["question_type"]
    items = data["items"][:LIMIT_PER_SUBSET]

    correct = 0
    total_time = 0.0
    results = []

    for idx, it in enumerate(items, 1):
        state = it["state"]
        gt = it.get("truth", it.get("target"))

        # Local per-item question spec if present
        cur_q = it.get("question", q_dict)
        cur_type = it.get("question_type", q_type)

        t0 = time.perf_counter()
        pred = None
        conf = 0.0

        if model_type == "jeff":
            q_jeff = {
                "type": cur_type,
                "instructions": cur_q.get("instructions"),
                "criteria": cur_q.get("criteria", {})
            }
            row = {"state": state, "question": q_jeff, "images": []}
            try:
                batch = jeff_model.prepare([row])
                with torch.inference_mode():
                    dist = (jeff_model(batch) / jeff_model.temperature).softmax(-1).cpu().tolist()
                ans = answer(q_jeff, dist[0][:batch.counts[0]])
                conf = float(ans.get("confidence", 0.0))

                if cur_type == "noul":
                    noul_val = ans.get("noul")
                    if isinstance(noul_val, bool):
                        pred = noul_val
                    elif isinstance(noul_val, (int, float)):
                        pred = (noul_val >= 0.5)
                    else:
                        pred = False
                elif cur_type == "choice":
                    pred = ans.get("choice")
                elif cur_type == "score":
                    pred = ans.get("score")
            except Exception as e:
                pred = None

        elif model_type == "clm":
            crit = cur_q.get("instructions", "")
            criteria = cur_q.get("criteria")
            if cur_type == "noul":
                options = criteria if isinstance(criteria, dict) else {"true": "True / Yes", "false": "False / No"}
            elif cur_type == "choice":
                options = criteria if isinstance(criteria, dict) else {str(i): str(c) for i, c in enumerate(criteria or [])}
            elif cur_type == "score":
                options = {str(i): str(c) for i, c in enumerate(criteria or [1,2,3,4,5])}
            else:
                options = {"1": "yes", "2": "no"}

            try:
                clm_res = clm_runner.predict_choice(state, crit, options)
                selected = clm_res["selected"]
                conf = float(clm_res["probabilities"].get(selected, 0.0))
                if cur_type == "noul":
                    pred = (str(selected).lower() in ("true", "1", "yes"))
                else:
                    pred = selected
            except Exception as e:
                pred = None

        lat = time.perf_counter() - t0
        total_time += lat

        # Check correctness
        is_corr = False
        if cur_type == "noul":
            is_corr = (bool(pred) == bool(gt))
        elif cur_type == "choice":
            is_corr = (str(pred) == str(gt))
        elif cur_type == "score":
            is_corr = (abs(float(pred or 0.0) - float(gt or 0.0)) < 0.5)

        if is_corr:
            correct += 1

        results.append({"pred": pred, "target": gt, "correct": is_corr, "confidence": conf, "latency": lat})

    acc = correct / len(items) if items else 0.0
    mean_lat = total_time / len(items) if items else 0.0
    return {"correct": correct, "total": len(items), "accuracy": acc, "mean_latency": mean_lat, "total_time": total_time}

all_results = {"jeff": {}, "clm": {}}

for model_key in ["jeff", "clm"]:
    m_name = "Jeff-0.8B" if model_key == "jeff" else "CLM-8B"
    print(f"\n{'='*60}\nEvaluating {m_name} on S1Bench...\n{'='*60}", flush=True)
    m_tot_corr = 0
    m_tot_items = 0
    m_tot_time = 0.0

    for s_name in SUBSETS:
        res = eval_subset(s_name, model_key)
        all_results[model_key][s_name] = res
        m_tot_corr += res["correct"]
        m_tot_items += res["total"]
        m_tot_time += res["total_time"]
        print(f"  [{s_name:15s}] {res['correct']:02d}/{res['total']:02d} ({res['accuracy']*100:.1f}%), avg lat: {res['mean_latency']*1000:.1f}ms", flush=True)

    macro_acc = m_tot_corr / m_tot_items if m_tot_items else 0.0
    all_results[model_key]["overall"] = {
        "correct": m_tot_corr,
        "total": m_tot_items,
        "accuracy": macro_acc,
        "total_time": m_tot_time,
        "mean_latency": m_tot_time / m_tot_items
    }
    print(f"\n--> {m_name} Overall: {m_tot_corr}/{m_tot_items} ({macro_acc*100:.2f}%), Total Time: {m_tot_time:.1f}s", flush=True)

out_file = BASE_DIR / "results_s1bench_jeff_clm.json"
out_file.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\nSaved S1Bench results to {out_file}", flush=True)
