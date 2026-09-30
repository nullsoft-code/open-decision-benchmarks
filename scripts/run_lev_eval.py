# -*- coding: utf-8 -*-
"""
Phase 2: Evaluate Lev (interfaze-ai/lev on RTX 3090) on S1Bench
"""
import os
import sys
import json
import time
import torch
from pathlib import Path

os.environ["HF_HOME"] = r"D:\Models\hf_cache"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
sys.path.insert(0, r"D:\test_arc\local_jev_proj\lev_repo\packages\lev\src")

import lev
from lev.types import Noul, Choice, Score

DATA_DIR = Path(r"D:\test_arc\local_jev_proj\s1bench_data")
OUTPUT_PATH = Path(r"D:\test_arc\local_jev_proj\results_lev.json")
CHECKPOINT_PATH = r"D:\Models\hf_cache\models--interfaze-ai--lev\snapshots\f8ef71157ec06a7d3b6435bc0756f9d735c33748"

SUBSETS = [
    "boolq",
    "paws",
    "vitaminc-dev",
    "massive-en-US",
    "aegis2",
    "helpsteer2",
]

LIMIT_PER_SUBSET = 50

print("=======================================================")
print(f"Loading Lev model from {CHECKPOINT_PATH} ...")
print("=======================================================")
t_load_0 = time.perf_counter()
model = lev.load(CHECKPOINT_PATH)
t_load = time.perf_counter() - t_load_0
print(f"Lev model loaded successfully in {t_load:.2f}s!")

def parse_question_obj(q_dict, q_type):
    if q_type == "noul":
        return Noul(instructions=q_dict.get("instructions"), criteria=q_dict.get("criteria"))
    elif q_type == "choice":
        return Choice(instructions=q_dict.get("instructions"), criteria=q_dict["criteria"])
    elif q_type == "score":
        return Score(instructions=q_dict.get("instructions"), criteria=q_dict["criteria"])
    raise ValueError(f"Unknown question type: {q_type}")

all_results = {}

try:
    for name in SUBSETS:
        file_path = DATA_DIR / f"{name}.json"
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        q_dict = data["question"]
        q_type = data["question_type"]
        items = data["items"][:LIMIT_PER_SUBSET]
        q_obj = parse_question_obj(q_dict, q_type)

        print(f"\n=======================================================")
        print(f"Evaluating Lev on '{name}' ({len(items)} items, type: {q_type})...")
        print(f"=======================================================")

        correct_count = 0
        latencies = []
        item_results = []
        t0 = time.perf_counter()

        for idx, item in enumerate(items):
            it_t0 = time.perf_counter()
            resp = model.system_one(item["state"], {"decision": q_obj})
            it_dt = (time.perf_counter() - it_t0) * 1000.0
            latencies.append(it_dt)

            ans = resp.answers["decision"]
            truth = item["truth"]
            pred = None
            is_correct = False

            if q_type == "noul":
                pred = (ans.noul >= 0.5)
                is_correct = (pred == truth)
            elif q_type == "choice":
                pred = ans.choice
                is_correct = (pred == truth)
            elif q_type == "score":
                probs = ans.probabilities or {}
                if probs:
                    pred = int(max(probs.items(), key=lambda kv: kv[1])[0])
                else:
                    pred = round(ans.score)
                is_correct = (pred == truth)

            if is_correct:
                correct_count += 1

            item_results.append({
                "id": item["id"],
                "truth": truth,
                "pred": pred,
                "is_correct": is_correct,
                "latency_ms": round(it_dt, 2),
            })

            if (idx + 1) % 10 == 0 or (idx + 1) == len(items):
                current_acc = correct_count / (idx + 1)
                print(f"  [{idx+1}/{len(items)}] Current Acc: {current_acc*100:.1f}%, Mean Lat: {sum(latencies)/len(latencies):.1f}ms")

        total_time = time.perf_counter() - t0
        acc = correct_count / len(items) if items else 0.0
        sorted_lat = sorted(latencies)
        median_lat = sorted_lat[len(sorted_lat) // 2] if sorted_lat else 0.0
        mean_lat = sum(latencies) / len(latencies) if latencies else 0.0

        print(f"  Done '{name}' in {total_time:.2f}s | Acc: {acc*100:.1f}% ({correct_count}/{len(items)}) | Median Lat: {median_lat:.1f}ms | Mean Lat: {mean_lat:.1f}ms")

        all_results[name] = {
            "subset": name,
            "question_type": q_type,
            "total": len(items),
            "correct": correct_count,
            "accuracy": round(acc, 4),
            "total_time_s": round(total_time, 2),
            "mean_latency_ms": round(mean_lat, 2),
            "median_latency_ms": round(median_lat, 2),
            "items": item_results
        }

    macro_acc = sum(r["accuracy"] for r in all_results.values()) / len(all_results)
    summary = {
        "model": "interfaze-ai/lev",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "limit_per_subset": LIMIT_PER_SUBSET,
        "load_time_s": round(t_load, 2),
        "macro_accuracy": round(macro_acc, 4),
        "subsets": all_results
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n=======================================================")
    print(f"Lev Evaluation Complete! Macro Accuracy: {macro_acc*100:.2f}%")
    print(f"Saved to {OUTPUT_PATH}")
    print(f"=======================================================")

finally:
    print("Releasing GPU memory...")
    del model
    torch.cuda.empty_cache()
    print("GPU memory released successfully!")
