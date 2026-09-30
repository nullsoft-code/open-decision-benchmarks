# -*- coding: utf-8 -*-
"""
SGLang to /v1/systemone Proxy Server
Exposes Qwen3.5-4B running on SGLang (:30000) as a standard SystemOne endpoint (:8002)
Supports Noul, Choice, and Score questions for levbench compatibility
"""
import json
import time
import math
import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

app = FastAPI(title="Qwen3.5-4B SGLang SystemOne Adapter", version="1.0.0")

SGLANG_URL = "http://localhost:30000/v1/completions"

def get_letter(idx):
    if idx < 26:
        return chr(ord('A') + idx)
    first = chr(ord('A') + (idx // 26) - 1)
    second = chr(ord('A') + (idx % 26))
    return f"{first}{second}"

def build_prompt_and_options(state, question_dict):
    q_type = question_dict.get("type", "noul")
    instructions = question_dict.get("instructions", "")
    criteria = question_dict.get("criteria", {})

    lines = []
    if isinstance(state, dict):
        state_str = "\n".join(f"{k}: {v}" for k, v in state.items())
    elif isinstance(state, list):
        state_str = "\n".join(str(s) for s in state)
    else:
        state_str = str(state)

    lines.append(f"State:\n{state_str}\n")
    if instructions:
        lines.append(f"Instructions:\n{instructions}\n")

    options_map = {} # Letter -> truth value
    lines.append("Options:")

    if q_type == "noul":
        options_map = {"A": False, "B": True}
        crit_false = criteria.get("false", "No / False") if isinstance(criteria, dict) else "No / False"
        crit_true = criteria.get("true", "Yes / True") if isinstance(criteria, dict) else "Yes / True"
        lines.append(f"  A: false - {crit_false}")
        lines.append(f"  B: true - {crit_true}")
    elif q_type == "choice":
        for idx, (opt_key, desc) in enumerate(criteria.items()):
            code = get_letter(idx)
            options_map[code] = opt_key
            desc_str = desc if isinstance(desc, str) else json.dumps(desc, ensure_ascii=False)
            lines.append(f"  {code}: {opt_key} - {desc_str}")
    elif q_type == "score":
        for idx, desc in enumerate(criteria):
            code = get_letter(idx)
            options_map[code] = idx
            desc_str = desc if isinstance(desc, str) else json.dumps(desc, ensure_ascii=False)
            lines.append(f"  {code}: level {idx} - {desc_str}")

    lines.append("Answer: <think>\n</think>\n")
    prompt = "\n".join(lines)
    return prompt, options_map, q_type

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "Qwen/Qwen3.5-4B",
        "backend": "sglang-proxy",
        "calibrated": False
    }

@app.post("/v1/systemone")
def system_one(req: Dict[str, Any]):
    state = req.get("state")
    questions = req.get("questions", {})
    if not questions:
        raise HTTPException(status_code=422, detail="No questions provided")

    answers = {}
    total_prompt_tokens = 0

    for q_name, q_spec in questions.items():
        prompt, options_map, q_type = build_prompt_and_options(state, q_spec)
        payload = {
            "model": "Qwen/Qwen3.5-4B",
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": 20
        }

        try:
            resp = requests.post(SGLANG_URL, json=payload, timeout=120)
            if resp.status_code != 200:
                raise HTTPException(status_code=500, detail=f"SGLang error: {resp.text}")
            res = resp.json()
            choice = res["choices"][0]
            gen_text = choice.get("text", "").strip().upper()
            lp_dict = choice.get("logprobs", {})
            top_logprobs = lp_dict.get("top_logprobs", [{}])[0] if lp_dict else {}

            # Map logits to options
            option_logprobs = {}
            for cand, lp in top_logprobs.items():
                cand_clean = cand.strip().upper()
                if cand_clean in options_map:
                    # Keep highest logprob per option letter
                    if cand_clean not in option_logprobs or lp > option_logprobs[cand_clean]:
                        option_logprobs[cand_clean] = lp

            # If an option wasn't in top 20, assign small fallback logprob
            for code in options_map:
                if code not in option_logprobs:
                    option_logprobs[code] = -15.0

            # Softmax normalize over the candidate options
            max_lp = max(option_logprobs.values())
            exps = {code: math.exp(lp - max_lp) for code, lp in option_logprobs.items()}
            sum_exps = sum(exps.values())
            probs = {code: (v / sum_exps) for code, v in exps.items()}

            best_code = max(probs, key=probs.get)
            best_val = options_map[best_code]
            confidence = probs[best_code]

            # Construct Answer object
            if q_type == "noul":
                p_true = probs.get("B", 0.5) # B is true
                answers[q_name] = {
                    "type": "noul",
                    "noul": p_true,
                    "confidence": confidence
                }
            elif q_type == "choice":
                prob_map = {options_map[code]: round(p, 4) for code, p in probs.items()}
                answers[q_name] = {
                    "type": "choice",
                    "choice": str(best_val),
                    "probabilities": prob_map,
                    "confidence": confidence
                }
            elif q_type == "score":
                prob_map = {int(options_map[code]): round(p, 4) for code, p in probs.items()}
                # Expectation score
                exp_score = sum(int(options_map[code]) * p for code, p in probs.items())
                answers[q_name] = {
                    "type": "score",
                    "score": round(exp_score, 2),
                    "probabilities": prob_map,
                    "legend": {i: f"level {i}" for i in range(len(options_map))},
                    "confidence": confidence
                }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return {
        "model": "Qwen/Qwen3.5-4B",
        "answers": answers,
        "usage": {
            "input_tokens": 100,
            "output_tokens": 1
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8002, log_level="warning")
