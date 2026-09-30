# -*- coding: utf-8 -*-
"""
Phase 2 Evaluator: Run Long-Context Benchmark across 4 backends:
1. Jev (TypeSafe Cloud API)
2. Lev (Local Port 8000)
3. Kev-4B (Local Port 8090)
4. Qwen3.5-4B (WSL2 SGLang Port 8002)

Outputs full accuracy, latency, and confidence metrics to markdown and json.
"""
import os
import sys
import json
import time
import requests
from pathlib import Path

DATA_FILE = Path(r"D:\test_arc\local_jev_proj\phase2_long_context\long_context_benchmark.json")
OUTPUT_MD = Path(r"D:\test_arc\local_jev_proj\phase2_long_context\phase2_results.md")

ENDPOINTS = {
    "Jev (Cloud)": {
        "url": "https://api.typesafe.ai/v1/systemone",
        "headers": {
            "Authorization": f"Bearer {os.environ.get('TYPESAFE_API_KEY', '')}",
            "Content-Type": "application/json"
        },
        "type": "cloud"
    },
    "Lev (Local :8000)": {
        "url": "http://127.0.0.1:8000/v1/systemone",
        "headers": {"Content-Type": "application/json"},
        "type": "local"
    },
    "Kev-4B (Local :8090)": {
        "url": "http://127.0.0.1:8090/v1/systemone",
        "headers": {"Content-Type": "application/json"},
        "type": "local"
    },
    "Qwen3.5-4B (SGLang :8002)": {
        "url": "http://127.0.0.1:8002/v1/systemone",
        "headers": {"Content-Type": "application/json"},
        "type": "local"
    }
}

def evaluate_backend(backend_name, config, benchmark_data):
    docs = benchmark_data["documents"]
    items = benchmark_data["items"]
    url = config["url"]
    headers = config["headers"]

    print(f"\n{'='*60}")
    print(f"Evaluating {backend_name} on {len(items)} items...")
    print(f"{'='*60}")

    results = []
    correct_count = 0
    total_time = 0.0

    for idx, item in enumerate(items, 1):
        doc_key = item["doc"]
        state_text = docs[doc_key]
        q_spec = item["question"]
        gt = item["ground_truth"]
        q_id = item["id"]

        req_body = {
            "state": state_text,
            "questions": {
                "decision": q_spec
            }
        }
        if config["type"] == "cloud":
            req_body["model"] = "jev-latest"

        t0 = time.perf_counter()
        try:
            resp = requests.post(url, headers=headers, json=req_body, timeout=180)
            latency = time.perf_counter() - t0
            total_time += latency

            if resp.status_code != 200:
                print(f"  [{idx:02d}/{len(items)}] {q_id}: HTTP {resp.status_code} - {resp.text[:80]}")
                results.append({
                    "id": q_id,
                    "correct": False,
                    "error": f"HTTP {resp.status_code}",
                    "latency": latency
                })
                continue

            res_json = resp.json()
            ans = res_json.get("answers", {}).get("decision", {})
            q_type = q_spec.get("type")

            pred = None
            conf = ans.get("confidence", 0.0)

            if q_type == "noul":
                noul_val = ans.get("noul")
                if isinstance(noul_val, bool):
                    pred = noul_val
                elif isinstance(noul_val, (int, float)):
                    pred = (noul_val >= 0.5)
                else:
                    pred = False
            elif q_type == "choice":
                pred = ans.get("choice")
            elif q_type == "score":
                pred = ans.get("score")

            is_correct = (pred == gt)
            if is_correct:
                correct_count += 1

            status_str = "OK" if is_correct else f"NG (pred={pred}, gt={gt})"
            print(f"  [{idx:02d}/{len(items)}] {q_id} ({latency:.2f}s, conf={conf:.2f}): {status_str}")

            results.append({
                "id": q_id,
                "q_type": q_type,
                "pred": pred,
                "truth": gt,
                "correct": is_correct,
                "confidence": conf,
                "latency": latency
            })

        except Exception as e:
            latency = time.perf_counter() - t0
            total_time += latency
            print(f"  [{idx:02d}/{len(items)}] {q_id}: Exception {e}")
            results.append({
                "id": q_id,
                "correct": False,
                "error": str(e),
                "latency": latency
            })

    acc = correct_count / len(items) if items else 0.0
    mean_lat = total_time / len(items) if items else 0.0
    print(f"--> {backend_name} Finished: Accuracy {acc*100:.1f}% ({correct_count}/{len(items)}), Avg Latency: {mean_lat:.2f}s, Total: {total_time:.1f}s")

    return {
        "accuracy": acc,
        "correct": correct_count,
        "total": len(items),
        "total_time": total_time,
        "mean_latency": mean_lat,
        "items": results
    }

def main():
    if not DATA_FILE.exists():
        print(f"Data file not found: {DATA_FILE}")
        return

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        bench_data = json.load(f)

    # Check Jev API key
    if not os.environ.get("TYPESAFE_API_KEY"):
        env_path = Path(r"C:\Users\kei51\code\GAME\rocketleague_mod\.env")
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("TYPESAFE_API_KEY="):
                    os.environ["TYPESAFE_API_KEY"] = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    if os.environ.get("TYPESAFE_API_KEY"):
        ENDPOINTS["Jev (Cloud)"]["headers"]["Authorization"] = f"Bearer {os.environ['TYPESAFE_API_KEY']}"

    all_results = {}
    for name, cfg in ENDPOINTS.items():
        try:
            res = evaluate_backend(name, cfg, bench_data)
            all_results[name] = res
        except Exception as e:
            print(f"Error evaluating {name}: {e}")

    # Generate Markdown Report
    lines = [
        "# Phase 2 実務長文（契約書・仕様書）ロングコンテキスト評価レポート",
        "",
        "> **評価対象**: 秘密保持契約書（NDA）、システム開発業務委託仕様書（全15問）",
        "> **コンテキスト長**: 各文書 約 2,500〜2,800 文字（約 1,800〜2,000 トークン）",
        "> **情報ソース**: 経済産業省『秘密情報の保護ハンドブック（参考例）』および『モデルIT契約書』",
        "",
        "---",
        "",
        "## 1. 総合サマリー比較表",
        "",
        "| モデル / バックエンド | 正解率 (Accuracy) | 正解数 | 合計走査時間 | 1問平均レイテンシ | 確信度>=0.9時 正解率 |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |"
    ]

    for name, res in all_results.items():
        conf90_items = [it for it in res["items"] if it.get("confidence", 0) >= 0.9]
        conf90_acc = (sum(1 for it in conf90_items if it.get("correct")) / len(conf90_items) * 100) if conf90_items else 0.0
        conf90_str = f"{conf90_acc:.1f}% ({len(conf90_items)}件)" if conf90_items else "-"
        lines.append(f"| **{name}** | **{res['accuracy']*100:.1f}%** | {res['correct']}/{res['total']} | {res['total_time']:.1f} 秒 | {res['mean_latency']:.2f} 秒 | {conf90_str} |")

    lines.extend([
        "",
        "---",
        "",
        "## 2. 設問別 正誤判定一覧",
        "",
        "| 設問ID | タスク内容 | 正解 (Ground Truth) | Jev | Lev | Kev-4B | Qwen3.5 |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |"
    ])

    items_meta = bench_data["items"]
    for it in items_meta:
        qid = it["id"]
        gt = it["ground_truth"]
        q_label = it["question"]["instructions"][:35] + "..."
        row = [f"`{qid}`", q_label, f"`{gt}`"]
        for bname in ["Jev (Cloud)", "Lev (Local :8000)", "Kev-4B (Local :8090)", "Qwen3.5-4B (SGLang :8002)"]:
            b_res = all_results.get(bname, {}).get("items", [])
            match = next((m for m in b_res if m.get("id") == qid), None)
            if match:
                is_c = match.get("correct")
                p = match.get("pred")
                c = match.get("confidence", 0.0)
                mark = f"✅ ({c:.2f})" if is_c else f"❌ `{p}`"
                row.append(mark)
            else:
                row.append("-")
        lines.append("| " + " | ".join(row) + " |")

    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nPhase 2 Full Report written to: {OUTPUT_MD}")

if __name__ == "__main__":
    main()
