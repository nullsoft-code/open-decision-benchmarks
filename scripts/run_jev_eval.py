# -*- coding: utf-8 -*-
"""
Phase 1: High-Speed Async Evaluate Jev (TypeSafe Cloud API) on S1Bench
"""
import os
import sys
import json
import time
import asyncio
import aiohttp
from pathlib import Path

# Load API Key
env_path = Path(r"C:\Users\kei51\code\GAME\rocketleague_mod\.env")
api_key = None
for line in env_path.read_text(encoding="utf-8").splitlines():
    if line.startswith("TYPESAFE_API_KEY="):
        api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
        break

if not api_key:
    raise ValueError("TYPESAFE_API_KEY not found in .env")

DATA_DIR = Path(r"D:\test_arc\local_jev_proj\s1bench_data")
OUTPUT_PATH = Path(r"D:\test_arc\local_jev_proj\results_jev.json")

SUBSETS = [
    "boolq",
    "paws",
    "vitaminc-dev",
    "massive-en-US",
    "aegis2",
    "helpsteer2",
]

LIMIT_PER_SUBSET = 50
CONCURRENCY = 10

async def evaluate_item(session, sem, item, question, q_type):
    async with sem:
        payload = {
            "model": "jev-latest",
            "state": item["state"],
            "questions": {
                "decision": question
            }
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        t0 = time.perf_counter()
        raw_ans = None
        for attempt in range(3):
            try:
                async with session.post("https://api.typesafe.ai/v1/systemone", json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status == 200:
                        res = await resp.json()
                        raw_ans = res.get("answers", {}).get("decision", {})
                        break
                    else:
                        text = await resp.text()
                        if attempt == 2:
                            print(f"  [HTTP {resp.status}] Item {item['id']}: {text[:100]}")
                        await asyncio.sleep(1.0)
            except Exception as e:
                if attempt == 2:
                    print(f"  [Error] Item {item['id']}: {e}")
                await asyncio.sleep(1.0)

        dt = (time.perf_counter() - t0) * 1000.0

        truth = item["truth"]
        pred = None
        is_correct = False
        if raw_ans:
            if q_type == "noul":
                noul_prob = raw_ans.get("noul", 0.0)
                pred = (noul_prob >= 0.5)
                is_correct = (pred == truth)
            elif q_type == "choice":
                pred = raw_ans.get("choice")
                is_correct = (pred == truth)
            elif q_type == "score":
                pred = raw_ans.get("score")
                is_correct = (pred == truth)

        return {
            "id": item["id"],
            "truth": truth,
            "pred": pred,
            "is_correct": is_correct,
            "latency_ms": round(dt, 2),
            "raw_ans": raw_ans
        }

async def evaluate_subset(session, sem, name):
    file_path = DATA_DIR / f"{name}.json"
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    question = data["question"]
    q_type = data["question_type"]
    items = data["items"][:LIMIT_PER_SUBSET]

    print(f"\n=======================================================")
    print(f"Evaluating Jev on '{name}' ({len(items)} items, type: {q_type})...")
    print(f"=======================================================")

    t0 = time.perf_counter()
    tasks = [evaluate_item(session, sem, item, question, q_type) for item in items]
    results = await asyncio.gather(*tasks)
    total_time = time.perf_counter() - t0

    correct_count = sum(1 for r in results if r["is_correct"])
    acc = correct_count / len(items) if items else 0.0
    latencies = [r["latency_ms"] for r in results]
    sorted_lat = sorted(latencies)
    median_lat = sorted_lat[len(sorted_lat) // 2] if sorted_lat else 0.0
    mean_lat = sum(latencies) / len(latencies) if latencies else 0.0

    print(f"  Done in {total_time:.2f}s | Acc: {acc*100:.1f}% ({correct_count}/{len(items)}) | Median Lat: {median_lat:.1f}ms | Mean Lat: {mean_lat:.1f}ms")

    return {
        "subset": name,
        "question_type": q_type,
        "total": len(items),
        "correct": correct_count,
        "accuracy": round(acc, 4),
        "total_time_s": round(total_time, 2),
        "mean_latency_ms": round(mean_lat, 2),
        "median_latency_ms": round(median_lat, 2),
        "items": results
    }

async def main():
    sem = asyncio.Semaphore(CONCURRENCY)
    connector = aiohttp.TCPConnector(limit=CONCURRENCY)
    all_results = {}

    async with aiohttp.ClientSession(connector=connector) as session:
        for name in SUBSETS:
            res = await evaluate_subset(session, sem, name)
            all_results[name] = res

    macro_acc = sum(r["accuracy"] for r in all_results.values()) / len(all_results)
    summary = {
        "model": "jev-latest",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "limit_per_subset": LIMIT_PER_SUBSET,
        "macro_accuracy": round(macro_acc, 4),
        "subsets": all_results
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n=======================================================")
    print(f"Jev Evaluation Complete! Macro Accuracy: {macro_acc*100:.2f}%")
    print(f"Saved to {OUTPUT_PATH}")
    print(f"=======================================================")

if __name__ == "__main__":
    asyncio.run(main())
