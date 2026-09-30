# -*- coding: utf-8 -*-
"""
Phase 4: High-Speed Async Evaluate Qwen3.5-4B (SGLang RadixAttention) on S1Bench
Concurrency: 20, max_new_tokens: 1, logprobs: True
"""
import os
import sys
import json
import time
import asyncio
import aiohttp
from pathlib import Path

DATA_DIR = Path(r"D:\test_arc\local_jev_proj\s1bench_data")
OUTPUT_PATH = Path(r"D:\test_arc\local_jev_proj\results_qwen_sglang.json")
SGLANG_URL = "http://localhost:30000/generate"

SUBSETS = [
    "boolq",
    "paws",
    "vitaminc-dev",
    "massive-en-US",
    "aegis2",
    "helpsteer2",
]

LIMIT_PER_SUBSET = 50
CONCURRENCY = 20

LETTERS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z"
]
# For >26 options (e.g. massive-en-US has up to 60 options)
def get_letter(idx):
    if idx < 26:
        return LETTERS[idx]
    d1 = idx // 26 - 1
    d2 = idx % 26
    return LETTERS[d1] + LETTERS[d2]

def build_sglang_prompt(state, question_dict, q_type):
    state_str = json.dumps(state, ensure_ascii=False, sort_keys=True) if isinstance(state, (dict, list)) else str(state)
    instr = question_dict.get("instructions") or ""

    lines = [
        "Context:",
        state_str,
        "",
        f"Question: {instr}",
        "Options:"
    ]

    options_map = {} # Letter -> value
    if q_type == "noul":
        # Binary false/true
        lines.append("  A: false")
        lines.append("  B: true")
        options_map = {"A": False, "B": True}
    elif q_type == "choice":
        criteria = question_dict.get("criteria", {})
        for idx, (k, v) in enumerate(criteria.items()):
            code = get_letter(idx)
            options_map[code] = k
            desc_str = f" - {v}" if v else ""
            lines.append(f"  {code}: {k}{desc_str}")
    elif q_type == "score":
        criteria = question_dict.get("criteria", [])
        for idx, desc in enumerate(criteria):
            code = get_letter(idx)
            options_map[code] = idx
            desc_str = desc if isinstance(desc, str) else json.dumps(desc, ensure_ascii=False)
            lines.append(f"  {code}: level {idx} - {desc_str}")

    lines.append("Answer: <think>\n</think>\n")
    prompt = "\n".join(lines)
    return prompt, options_map

async def evaluate_item(session, sem, item, question_dict, q_type):
    async with sem:
        prompt, options_map = build_sglang_prompt(item["state"], question_dict, q_type)
        payload = {
            "model": "Qwen/Qwen3.5-4B",
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": 10
        }

        t0 = time.perf_counter()
        raw_text = ""
        top_logprobs = {}
        for attempt in range(3):
            try:
                async with session.post("http://localhost:30000/v1/completions", json=payload, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status == 200:
                        res = await resp.json()
                        choice = res["choices"][0]
                        raw_text = choice.get("text", "")
                        lp_dict = choice.get("logprobs", {})
                        if lp_dict and "top_logprobs" in lp_dict and lp_dict["top_logprobs"]:
                            top_logprobs = lp_dict["top_logprobs"][0]
                        break
                    else:
                        text = await resp.text()
                        if attempt == 2:
                            print(f"  [HTTP {resp.status}] Item {item['id']}: {text[:100]}")
                        await asyncio.sleep(0.5)
            except Exception as e:
                if attempt == 2:
                    print(f"  [Error] Item {item['id']}: {e}")
                await asyncio.sleep(0.5)

        dt = (time.perf_counter() - t0) * 1000.0

        # Parse generated token
        gen_token = raw_text.strip().upper()
        # Fallback to logprobs if direct text isn't a known letter
        pred = options_map.get(gen_token)

        if pred is None and top_logprobs:
            # Check logprobs top candidates (dict of token -> logprob)
            for cand_token in top_logprobs:
                cand = cand_token.strip().upper()
                if cand in options_map:
                    pred = options_map[cand]
                    break

        # Fallback to default
        if pred is None:
            pred = list(options_map.values())[0]

        truth = item["truth"]
        is_correct = (pred == truth)

        return {
            "id": item["id"],
            "truth": truth,
            "pred": pred,
            "gen_token": gen_token,
            "is_correct": is_correct,
            "latency_ms": round(dt, 2),
            "prompt_tokens": 0
        }

async def evaluate_subset(session, sem, name):
    file_path = DATA_DIR / f"{name}.json"
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    question_dict = data["question"]
    q_type = data["question_type"]
    items = data["items"][:LIMIT_PER_SUBSET]

    print(f"\n=======================================================")
    print(f"Evaluating Qwen3.5-4B SGLang on '{name}' ({len(items)} items, type: {q_type})...")
    print(f"=======================================================")

    t0 = time.perf_counter()
    tasks = [evaluate_item(session, sem, item, question_dict, q_type) for item in items]
    results = await asyncio.gather(*tasks)
    total_time = time.perf_counter() - t0

    correct_count = sum(1 for r in results if r["is_correct"])
    acc = correct_count / len(items) if items else 0.0
    latencies = [r["latency_ms"] for r in results]
    sorted_lat = sorted(latencies)
    median_lat = sorted_lat[len(sorted_lat) // 2] if sorted_lat else 0.0
    mean_lat = sum(latencies) / len(latencies) if latencies else 0.0

    print(f"  Done '{name}' in {total_time:.2f}s | Acc: {acc*100:.1f}% ({correct_count}/{len(items)}) | Median Lat: {median_lat:.1f}ms | Mean Lat: {mean_lat:.1f}ms")

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
        "model": "Qwen/Qwen3.5-4B (SGLang RadixAttention bfloat16)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "limit_per_subset": LIMIT_PER_SUBSET,
        "concurrency": CONCURRENCY,
        "macro_accuracy": round(macro_acc, 4),
        "subsets": all_results
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n=======================================================")
    print(f"Qwen3.5-4B SGLang Complete! Macro Accuracy: {macro_acc*100:.2f}%")
    print(f"Saved to {OUTPUT_PATH}")
    print(f"=======================================================")

if __name__ == "__main__":
    asyncio.run(main())
