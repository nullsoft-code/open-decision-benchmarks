# -*- coding: utf-8 -*-
import os
import time
from typing import Dict, Any
from .base import BaseDecisionRunner

class JevRunner(BaseDecisionRunner):
    """
    Runner for proprietary Jev API (TypeSafe AI).
    Requires TYPESAFE_API_KEY environment variable.
    """

    def __init__(self, model: str = "jev-latest", api_key: str = None):
        super().__init__(name=f"Jev ({model})")
        self.model = model
        self.api_key = api_key or os.getenv("TYPESAFE_API_KEY")
        if not self.api_key:
            raise ValueError("TYPESAFE_API_KEY environment variable is required to run JevRunner.")

        from typesafe_sdk import TypeSafeClient
        self.client = TypeSafeClient(api_key=self.api_key)

    def predict_choice(
        self,
        evidence: str,
        criterion: str,
        options: Dict[str, str]
    ) -> Dict[str, Any]:
        questions = {
            "decision": {
                "type": "choice",
                "instructions": criterion,
                "criteria": options
            }
        }
        t0 = time.perf_counter()
        resp = self.client.system_one(
            state=evidence,
            questions=questions,
            model=self.model
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        ans = resp.answers["decision"]
        probs = ans.probabilities # e.g. {'opt_a': 0.95, 'opt_b': 0.05}
        selected = max(probs, key=probs.get)

        return {
            "probabilities": probs,
            "selected": selected,
            "latency_ms": latency_ms
        }
