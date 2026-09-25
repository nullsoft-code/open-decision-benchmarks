# -*- coding: utf-8 -*-
"""
CLM-8B (Contrastive Language Model) Runner for Open Decision Benchmarks
Loads Qwen3-8B base encoder and CLM projection heads for zero-latency typed decisions.
"""

import os
import time
from typing import Dict, Any, List
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download

from .base import BaseDecisionRunner

HIDDEN_DIM = 4096
PROJ_DIM = 512

def make_head(width: int = 1536, depth: int = 3, proj: int = PROJ_DIM,
              activation: str = "gelu", layernorm: bool = True, residual: bool = False,
              hidden: int = HIDDEN_DIM):
    """3-layer MLP projection head matching CLM v0.1 checkpoint."""
    act_fn = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}[activation]

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Linear(hidden, width)
            self.hidden = nn.ModuleList(nn.Linear(width, width) for _ in range(depth - 2))
            self.norms = nn.ModuleList((nn.LayerNorm(width) if layernorm else nn.Identity())
                                       for _ in range(depth - 2))
            self.out = nn.Linear(width, proj)
            self.act = act_fn()
            self.residual = residual

        def forward(self, x):
            x = self.act(self.inp(x))
            for lin, nrm in zip(self.hidden, self.norms):
                h = self.act(nrm(lin(x)))
                x = x + h if self.residual else h
            return self.out(x)

    return Head()


class CLMRunner(BaseDecisionRunner):
    """
    Runner for CLM-8B (Contrastive LM / Qwen3-8B).
    Inherits from BaseDecisionRunner for unified benchmarking.
    """
    def __init__(
        self,
        base_model_path: str = r"D:\Models\Qwen3-8B",
        head_ckpt_path: str | None = None,
        device: str = "cuda"
    ):
        super().__init__(name="CLM-8B (Contrastive LM / Qwen3-8B)")
        self.device = device
        print(f"[CLMRunner] Loading tokenizer & base encoder from {base_model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            trust_remote_code=True
        )
        self.base_model.eval()

        if not head_ckpt_path or not os.path.exists(head_ckpt_path):
            print("[CLMRunner] Fetching CLM-v0.1-8B.pt from Hugging Face Hub...")
            head_ckpt_path = hf_hub_download("Contrastive-LM/CLM-v0.1-8B", "CLM_v0.1-8B.pt")

        print(f"[CLMRunner] Loading projection heads from {head_ckpt_path}...")
        ckpt = torch.load(head_ckpt_path, map_location="cpu")
        cfg = ckpt.get("cfg", {})
        kw = dict(
            width=cfg.get("width", 1536),
            depth=cfg.get("depth", 3),
            proj=ckpt.get("projection_dim", cfg.get("projection_dim", PROJ_DIM)),
            activation=cfg.get("activation", "gelu"),
            layernorm=cfg.get("layernorm", True),
            residual=cfg.get("residual", False),
            hidden=cfg.get("hidden_size", HIDDEN_DIM)
        )
        self.state_head = make_head(**kw)
        self.action_head = make_head(**kw)
        self.state_head.load_state_dict(ckpt["state_head"])
        self.action_head.load_state_dict(ckpt["action_head"])
        self.state_head.eval().to(self.device)
        self.action_head.eval().to(self.device)
        self.scale = float(torch.as_tensor(ckpt["logit_scale"]).float().exp().clamp(max=100.0))
        print(f"[CLMRunner] Successfully loaded CLM-8B (Scale: {self.scale:.2f})")

    @torch.no_grad()
    def _get_embedding(self, texts: List[str]) -> torch.Tensor:
        """Extracts L2-normalized last-token embeddings from Qwen3-8B."""
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=2048,
            return_tensors="pt"
        ).to(self.device)

        outputs = self.base_model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[-1] # [B, T, 4096]
        
        # Last non-padding token
        seq_lengths = inputs["attention_mask"].sum(dim=1) - 1
        last_embs = hidden[torch.arange(hidden.size(0)), seq_lengths] # [B, 4096]
        
        return nn.functional.normalize(last_embs.float(), p=2, dim=-1)

    def predict_choice(
        self,
        evidence: str,
        criterion: str,
        options: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Executes a typed decision using CLM-8B contrastive scoring.
        """
        t0 = time.time()
        
        s_text = f"{evidence}\n\n{criterion}".strip() if evidence and criterion else (evidence or criterion)
        opt_keys = list(options.keys())
        opt_texts = [options[k] if options[k] not in (None, "") else k for k in opt_keys]
        
        with torch.no_grad():
            s_emb = self._get_embedding([s_text])
            s_proj = self.state_head(s_emb)
            s_proj = nn.functional.normalize(s_proj, p=2, dim=-1)
            
            a_emb = self._get_embedding(opt_texts)
            a_proj = self.action_head(a_emb)
            a_proj = nn.functional.normalize(a_proj, p=2, dim=-1)
            
            sim = torch.sum(s_proj * a_proj, dim=-1)
            logits = sim * self.scale
            probs = torch.softmax(logits, dim=-1).cpu().numpy().tolist()

        prob_dict = {k: float(p) for k, p in zip(opt_keys, probs)}
        selected = max(prob_dict, key=prob_dict.get)
        latency = (time.time() - t0) * 1000.0

        return {
            "probabilities": prob_dict,
            "selected": selected,
            "latency_ms": latency
        }
