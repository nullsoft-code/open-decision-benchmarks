# -*- coding: utf-8 -*-
import time
from typing import Dict, Any, List
from .base import BaseDecisionRunner

class KevRunner(BaseDecisionRunner):
    """
    Runner for Kev-4B (Jared Palmer / DeltaNet Decision Model).
    Executes directly via PyTorch with Pointer Head softmax scoring.
    """

    def __init__(self, model_id: str = "jaredpalmer/kev-4b", device: str = "cuda"):
        super().__init__(name=f"Kev ({model_id})")
        self.device = device
        self.model_id = model_id

        import torch
        from kev.checkpoint import Checkpoint, LoadOptions

        print(f"Loading {model_id} on {device}...")
        ck = Checkpoint(model_id)
        opts = LoadOptions(dtype=torch.bfloat16, merge=True)
        self.tok, self.model = ck.load(device, opts)
        print("Kev model loaded successfully.")

    def predict_choice(
        self,
        evidence: str,
        criterion: str,
        options: Dict[str, str]
    ) -> Dict[str, Any]:
        opt_keys: List[str] = list(options.keys())
        opt_texts: List[str] = [options[k] for k in opt_keys]

        rec = {
            "state": evidence,
            "questions": [{
                "instr": criterion,
                "options": opt_texts,
                "label": 0
            }]
        }

        t0 = time.perf_counter()
        enc = self.model.encode(self.tok, rec)
        raw_probs = [float(x) for x in self.model.probs(enc)[0]]
        latency_ms = (time.perf_counter() - t0) * 1000

        # Map back to option keys
        probs_dict = {k: p for k, p in zip(opt_keys, raw_probs)}
        selected = max(probs_dict, key=probs_dict.get)

        return {
            "probabilities": probs_dict,
            "selected": selected,
            "latency_ms": latency_ms
        }
