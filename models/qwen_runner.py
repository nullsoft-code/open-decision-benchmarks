# -*- coding: utf-8 -*-
import json
import time
import math
import urllib.request
from typing import Dict, Any, List
from .base import BaseDecisionRunner

SYSTEM_PROMPT = (
    "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)

class QwenRunner(BaseDecisionRunner):
    """
    Runner for Qwen3.5-4B Direct Logit extraction via llama-server or OpenAI-compatible completion server.
    Extracts top logprobs for option tokens (A, B, C...) and normalizes via Softmax.
    """

    def __init__(self, endpoint_url: str = "http://127.0.0.1:8089/v1/completions", model_name: str = "Qwen3.5-4B"):
        super().__init__(name=f"Qwen ({model_name})")
        self.endpoint_url = endpoint_url

    def predict_choice(
        self,
        evidence: str,
        criterion: str,
        options: Dict[str, str]
    ) -> Dict[str, Any]:
        opt_keys: List[str] = list(options.keys())
        # Map option keys to letters (A, B, C...)
        letters = ["A", "B", "C", "D", "E", "F"][:len(opt_keys)]
        letter_to_key = {l: k for l, k in zip(letters, opt_keys)}

        formatted_opts = [{"letter": l, "description": options[k]} for l, k in zip(letters, opt_keys)]
        user_content = json.dumps({"evidence": evidence, "criterion": criterion, "options": formatted_opts}, ensure_ascii=False)
        raw_prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n"

        payload = {
            "prompt": raw_prompt,
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": 20
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(self.endpoint_url, data=data, headers={"Content-Type": "application/json"})

        t0 = time.perf_counter()
        with urllib.request.urlopen(req) as resp:
            res_json = json.loads(resp.read().decode("utf-8"))
        latency_ms = (time.perf_counter() - t0) * 1000

        choice = res_json["choices"][0]
        logprobs_content = choice.get("logprobs", {}).get("content", [])

        # Extract logprobs for target letters
        raw_probs = {}
        first_token = ""
        if logprobs_content:
            first_token = logprobs_content[0].get("token", "").strip()
            top_list = logprobs_content[0].get("top_logprobs", [])
            for l in letters:
                raw_probs[l] = 0.0
            for item in top_list:
                tok = item.get("token", "").strip()
                lp = item.get("logprob", -999.0)
                if tok in raw_probs:
                    raw_probs[tok] += math.exp(lp)

        total_prob = sum(raw_probs.values())
        norm_probs = {}
        for l in letters:
            k = letter_to_key[l]
            if total_prob > 0:
                norm_probs[k] = raw_probs[l] / total_prob
            else:
                norm_probs[k] = 1.0 if first_token == l else 0.0

        selected = max(norm_probs, key=norm_probs.get)

        return {
            "probabilities": norm_probs,
            "selected": selected,
            "latency_ms": latency_ms
        }
