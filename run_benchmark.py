# -*- coding: utf-8 -*-
"""
Open Decision Models Benchmark Suite
Evaluates typed decision capabilities across proprietary and open models.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List

# Load environment variables if .env exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from models import JevRunner, QwenRunner, KevRunner, CLMRunner

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"

def load_json(filepath: Path) -> Any:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def run_boundary_benchmark(runners: List[Any], prompts: Dict[str, Any], cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print(" TASK: Boundary & Deception Contrast Benchmark (13 Cases)")
    print("=" * 80)

    patterns = ["boundary_legacy", "boundary_reframed"]
    benchmark_data = {}

    for p_id in patterns:
        p_info = prompts[p_id]
        print(f"\n[Prompt Pattern: {p_info['title']}]")
        print(f"Criterion: {p_info['criterion']}")
        benchmark_data[p_id] = {}

        for runner in runners:
            print(f"\n  ▶ Evaluating: {runner.name}")
            runner_results = []
            for case in cases:
                try:
                    res = runner.predict_choice(
                        evidence=case["text"],
                        criterion=p_info["criterion"],
                        options=p_info["options"]
                    )
                    d_prob = res["probabilities"].get("disguised", 0.0)
                    nd_prob = res["probabilities"].get("not_disguised", 0.0)
                    is_correct = res["selected"] == case["expected"]

                    runner_results.append({
                        "id": case["id"],
                        "name": case["name"],
                        "expected": case["expected"],
                        "selected": res["selected"],
                        "disguised_prob": d_prob,
                        "not_disguised_prob": nd_prob,
                        "latency_ms": res["latency_ms"],
                        "is_correct": is_correct
                    })
                    mark = "✓" if is_correct else "✗"
                    print(f"    [{mark}] {case['name']:<22} | P(disguised): {d_prob*100:6.2f}% | Latency: {res['latency_ms']:5.1f}ms")
                except Exception as e:
                    print(f"    [ERR] {case['name']}: {e}")

            benchmark_data[p_id][runner.name] = runner_results

    return benchmark_data

def format_summary_markdown(boundary_data: Dict[str, Any], cases: List[Dict[str, Any]]) -> str:
    md = "# Benchmark Summary: Decision Models Boundary Test\n\n"

    for p_id, p_results in boundary_data.items():
        title = "旧命題（現行プロンプト / 疑問形フラグ）" if "legacy" in p_id else "新命題（改善プロンプト / 対称な身分・正体判定）"
        md += f"## {title}\n\n"
        
        runner_names = list(p_results.keys())
        headers = ["No", "テストケース", "期待判定"] + runner_names
        md += "| " + " | ".join(headers) + " |\n"
        md += "| " + " | ".join([":---:" if i == 0 or i == 2 else ":---" for i in range(len(headers))]) + " |\n"

        for idx, case in enumerate(cases, 1):
            exp_str = "変装 (○)" if case["expected"] == "disguised" else "非変装 (×)"
            row = [str(idx), case["name"], exp_str]
            for r_name in runner_names:
                r_items = {item["id"]: item for item in p_results[r_name]}
                item = r_items.get(case["id"])
                if item:
                    prob = item["disguised_prob"] * 100
                    mark = "◎" if item["is_correct"] and (prob >= 80 or prob <= 20) else ("○" if item["is_correct"] else "❌")
                    row.append(f"**{prob:.1f}% ({mark})**")
                else:
                    row.append("N/A")
            md += "| " + " | ".join(row) + " |\n"
        md += "\n"

    return md

def main():
    parser = argparse.ArgumentParser(description="Open Decision Models Benchmark CLI")
    parser.add_argument("--model", type=str, default="all", choices=["all", "jev", "qwen", "kev", "clm"],
                        help="Model to test (default: all)")
    parser.add_argument("--task", type=str, default="boundary", choices=["boundary", "dice", "all"],
                        help="Benchmark task (default: boundary)")
    parser.add_argument("--qwen-url", type=str, default="http://127.0.0.1:8089/v1/completions",
                        help="Endpoint URL for llama-server")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize runners
    runners = []
    if args.model in ["all", "jev"]:
        try:
            runners.append(JevRunner())
            print("[Init] JevRunner initialized.")
        except Exception as e:
            print(f"[Skip] JevRunner could not be initialized: {e}")

    if args.model in ["all", "qwen"]:
        try:
            runners.append(QwenRunner(endpoint_url=args.qwen_url))
            print("[Init] QwenRunner initialized.")
        except Exception as e:
            print(f"[Skip] QwenRunner could not be initialized: {e}")

    if args.model in ["all", "kev"]:
        if KevRunner is not None:
            try:
                runners.append(KevRunner())
                print("[Init] KevRunner initialized.")
            except Exception as e:
                print(f"[Skip] KevRunner could not be initialized: {e}")
        else:
            print("[Skip] KevRunner module not available in this environment.")

    if args.model in ["all", "clm"]:
        if CLMRunner is not None:
            try:
                runners.append(CLMRunner())
                print("[Init] CLMRunner initialized.")
            except Exception as e:
                print(f"[Skip] CLMRunner could not be initialized: {e}")
        else:
            print("[Skip] CLMRunner module not available in this environment.")

    if not runners:
        print("Error: No models could be initialized.")
        sys.exit(1)

    # Load datasets
    prompts = load_json(DATA_DIR / "prompts.json")
    boundary_cases = load_json(DATA_DIR / "boundary_cases.json")

    # Run boundary task
    if args.task in ["boundary", "all"]:
        results = run_boundary_benchmark(runners, prompts, boundary_cases)
        
        # Save raw JSON
        out_json = RESULTS_DIR / "boundary_benchmark_results.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n[Saved] Raw results written to {out_json}")

        # Save Markdown summary
        md_summary = format_summary_markdown(results, boundary_cases)
        out_md = RESULTS_DIR / "summary_table.md"
        with open(out_md, "w", encoding="utf-8") as f:
            f.write(md_summary)
        print(f"[Saved] Markdown summary written to {out_md}")

    print("\nAll benchmark tasks completed successfully.")

if __name__ == "__main__":
    main()
