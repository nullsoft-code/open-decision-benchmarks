# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseDecisionRunner(ABC):
    """
    Abstract base class for typed decision model runners.
    Ensures uniform interface across Cloud Jev, Kev-4B, and Qwen Direct Logit.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def predict_choice(
        self,
        evidence: str,
        criterion: str,
        options: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Executes a single typed decision.

        Args:
            evidence: Context or state text.
            criterion: Decision question or instructions.
            options: Dictionary of {option_key: option_description}.

        Returns:
            Dictionary with:
                - 'probabilities': Dict[str, float] normalized probabilities for each option.
                - 'selected': str (key of highest probability option)
                - 'latency_ms': float (inference time in milliseconds)
        """
        pass
