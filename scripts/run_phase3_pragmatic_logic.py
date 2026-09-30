# -*- coding: utf-8 -*-
"""
Phase 3 Evaluator: Pragmatic Logic, Grammatical Negation & Contradiction Benchmark
Evaluates:
1. Jev (TypeSafe Cloud API)
2. Lev (Local Port 8000)
3. Kev-4B (Local Port 8090)
4. Qwen3.5-4B (WSL2 SGLang Adapter Port 8002)

Records:
- Accuracy (Overall, Language-wise JA/EN, Category-wise)
- Mean Confidence & Calibration
- Median Latency per decision
- Error Breakdown & Pragmatic Fallacies
"""
import os
import sys
import json
import time
import requests
from pathlib import Path

BASE_DIR = Path(r"D:\test_arc\local_jev_proj\phase3_pragmatic_logic")
DATA_FILE = BASE_DIR / "test_cases.json"
RESULTS_JSON = BASE_DIR / "phase3_results.json"
RESULTS_MD = BASE_DIR / "phase3_results.md"

# Load Jev API Key
env_path = Path(r"C:\Users\kei51\code\GAME\rocketleague_mod\.env")
jev_api_key = os.environ.get("TYPESAFE_API_KEY", "")
if not jev_api_key and env_path.exists():
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("TYPESAFE_API_KEY="):
            jev_api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
            break

ENDPOINTS = {
    "Jev (Cloud)": {
        "url": "https://api.typesafe.ai/v1/systemone",
        "headers": {
            "Authorization": f"Bearer {jev_api_key}",
            "Content-Type": "application/json"
        },
        "timeout": 15
    },
    "Lev (Local :8000)": {
        "url": "http://127.0.0.1:8000/v1/systemone",
        "headers": {"Content-Type": "application/json"},
        "timeout": 30
    },
    "Kev-4B (Local :8090)": {
        "url": "http://127.0.0.1:8090/v1/systemone",
        "headers": {"Content-Type": "application/json"},
        "timeout": 15
    },
    "Qwen3.5-4B (SGLang :8002)": {
        "url": "http://127.0.0.1:8002/v1/systemone",
        "headers": {"Content-Type": "application/json"},
        "timeout": 15
    }
}

def evaluate_backend(backend_name, config, test_cases):
    url = config["url"]
    headers = config["headers"]
    timeout = config["timeout"]

    print(f"\n{'='*70}", flush=True)
    print(f"Evaluating {backend_name} ({len(test_cases)} items)...", flush=True)
    print(f"{'='*70}", flush=True)

    results = []
    correct_count = 0
    total_time = 0.0

    for idx, tc in enumerate(test_cases):
        q_key = tc["question_key"]
        payload = {
            "model": "jev-latest",
            "state": tc["state"],
            "questions": {
                q_key: {
                    "type": "choice",
                    "instructions": tc["instructions"],
                    "criteria": tc["criteria"]
                }
            }
        }

        t0 = time.time()
        choice = None
        conf = 0.0
        probs = {}
        status = "ERROR"

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            elapsed = time.time() - t0
            total_time += elapsed

            if resp.status_code == 200:
                data = resp.json()
                ans = data.get("answers", {}).get(q_key, {})
                choice = str(ans.get("choice", ""))
                conf = float(ans.get("confidence", 0.0))
                probs = ans.get("probabilities", {})
                is_correct = (choice == str(tc["expected"]))
                if is_correct:
                    correct_count += 1
                    status = "PASS"
                else:
                    status = "FAIL"
            else:
                elapsed = time.time() - t0
                total_time += elapsed
                status = f"HTTP_{resp.status_code}"
                print(f"  [{status}] item {tc['id']}: {resp.text[:100]}", flush=True)

        except Exception as e:
            elapsed = time.time() - t0
            total_time += elapsed
            status = f"EXC_{type(e).__name__}"
            print(f"  [{status}] item {tc['id']}: {e}", flush=True)

        is_corr = (choice == str(tc["expected"]))
        results.append({
            "id": tc["id"],
            "category": tc["category"],
            "language": tc["language"],
            "expected": str(tc["expected"]),
            "choice": choice,
            "is_correct": is_corr,
            "confidence": conf,
            "probabilities": probs,
            "latency": elapsed,
            "status": status,
            "explanation": tc["explanation"]
        })

        mark = "✓" if is_corr else "✗"
        print(f"  [{idx+1:02d}/{len(test_cases):02d}] {mark} {tc['id']} ({tc['category']}) -> got {choice}, exp {tc['expected']} (conf: {conf:.2f}, {elapsed:.2f}s)", flush=True)

    acc = correct_count / len(test_cases) if test_cases else 0.0
    mean_lat = total_time / len(test_cases) if test_cases else 0.0
    print(f"--> {backend_name} Result: {correct_count}/{len(test_cases)} ({acc*100:.1f}%), Avg Latency: {mean_lat:.2f}s, Total: {total_time:.1f}s", flush=True)

    return {
        "correct": correct_count,
        "total": len(test_cases),
        "accuracy": acc,
        "total_time": total_time,
        "mean_latency": mean_lat,
        "items": results
    }

def main():
    test_cases = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    print(f"Loaded {len(test_cases)} test cases from {DATA_FILE}", flush=True)

    all_results = {}
    if RESULTS_JSON.exists():
        try:
            all_results = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
        except Exception:
            all_results = {}

    target_filter = sys.argv[1].lower() if len(sys.argv) > 1 else None

    for backend_name, config in ENDPOINTS.items():
        if target_filter and target_filter not in backend_name.lower():
            print(f"Skipping {backend_name} (filtered)", flush=True)
            continue
        all_results[backend_name] = evaluate_backend(backend_name, config, test_cases)

    # Save JSON
    RESULTS_JSON.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved raw results to {RESULTS_JSON}", flush=True)

    # Generate Markdown Report
    generate_markdown_report(all_results, test_cases)

def generate_markdown_report(all_results, test_cases):
    md = []
    md.append("# Phase 3: 文法・語用論的矛盾（Pragmatic Logic & Negation Traps）ベンチマーク結果\n")
    md.append(f"**実施日時**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    md.append(f"**総問題数**: {len(test_cases)} 問（日本語: 10問, 英語: 10問）\n")

    # Overall Summary Table
    md.append("## 1. 総合スコア（正解率・レイテンシ・確信度）\n")
    md.append("| モデル / バックエンド | 正解率 (%) | 正解数 | 平均レイテンシ | 合計時間 | JA正解率 | EN正解率 |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

    for name, res in all_results.items():
        items = res["items"]
        ja_items = [it for it in items if it["language"] == "ja"]
        en_items = [it for it in items if it["language"] == "en"]
        ja_acc = sum(1 for it in ja_items if it["is_correct"]) / len(ja_items) * 100 if ja_items else 0
        en_acc = sum(1 for it in en_items if it["is_correct"]) / len(en_items) * 100 if en_items else 0

        md.append(f"| **{name}** | **{res['accuracy']*100:.1f}%** | {res['correct']}/{res['total']} | {res['mean_latency']:.2f}s | {res['total_time']:.1f}s | {ja_acc:.1f}% ({sum(1 for it in ja_items if it['is_correct'])}/{len(ja_items)}) | {en_acc:.1f}% ({sum(1 for it in en_items if it['is_correct'])}/{len(en_items)}) |")

    # Category Breakdown
    categories = sorted(list(set(tc["category"] for tc in test_cases)))
    md.append("\n## 2. カテゴリ別正解率（語用論・文法トラップの内訳）\n")
    cat_header = "| カテゴリ | 問題数 | " + " | ".join(all_results.keys()) + " |"
    md.append(cat_header)
    md.append("| :--- | :---: | " + " | ".join([":---:"] * len(all_results)) + " |")

    for cat in categories:
        cat_items_ref = [tc for tc in test_cases if tc["category"] == cat]
        row = [f"**{cat}**", str(len(cat_items_ref))]
        for name, res in all_results.items():
            cat_res = [it for it in res["items"] if it["category"] == cat]
            c_corr = sum(1 for it in cat_res if it["is_correct"])
            row.append(f"{c_corr}/{len(cat_res)} ({c_corr/len(cat_res)*100:.0f}%)")
        md.append("| " + " | ".join(row) + " |")

    # Detailed Item Breakdown Table
    md.append("\n## 3. 全問詳細マトリクス（正誤比較）\n")
    item_header = "| ID | 言語 | カテゴリ | 正解 | " + " | ".join(all_results.keys()) + " |"
    md.append(item_header)
    md.append("| :--- | :---: | :--- | :---: | " + " | ".join([":---:"] * len(all_results)) + " |")

    for tc in test_cases:
        row = [f"`{tc['id']}`", tc["language"].upper(), tc["category"], tc["expected"]]
        for name, res in all_results.items():
            it = next((x for x in res["items"] if x["id"] == tc["id"]), None)
            if it:
                mark = "✓" if it["is_correct"] else f"✗ (got {it['choice']})"
                row.append(mark)
            else:
                row.append("—")
        md.append("| " + " | ".join(row) + " |")

    # Failure Analysis & Theoretical Takeaways
    md.append("\n## 4. 各モデルの失敗パターンと語用論的分析\n")
    for name, res in all_results.items():
        failed = [it for it in res["items"] if not it["is_correct"]]
        md.append(f"### {name} (失敗: {len(failed)} 問)")
        if not failed:
            md.append("- 全問正解。表面的な否定バイアスや前提・反事実の論理トラップを完全に克服しています。\n")
        else:
            for it in failed:
                md.append(f"- **`{it['id']}`** ({it['category']}): 予測 `{it['choice']}` / 正解 `{it['expected']}`")
                md.append(f"  - 解説: {it['explanation']}")
            md.append("")

    RESULTS_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"Generated Markdown report at {RESULTS_MD}", flush=True)

if __name__ == "__main__":
    main()
