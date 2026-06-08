"""Text preprocessing helper for the Dp-VLA model.

Owns the tokenizer side of the Florence-2 processor; kept separate from
:class:`DpVlaModel` because it is not an ``nn.Module`` (no parameters) and is
shared by both the base and the RL agents.
"""

from __future__ import annotations

from typing import List

import torch
from transformers import AutoProcessor


DEFAULT_MAX_LENGTH: int = 40


class DpVlaLanguagePreprocessor:
    """Tokenise the natural-language driving instruction."""

    def __init__(
        self,
        encoder_name: str = "microsoft/Florence-2-base",
        device: str = "cuda",
        max_length: int = DEFAULT_MAX_LENGTH,
    ) -> None:
        self.preprocessor = AutoProcessor.from_pretrained(
            encoder_name, trust_remote_code=True
        )
        self.device = device
        self.max_length = max_length

    @torch.no_grad()
    def encode_language(self, language_instruction: List[str]) -> torch.Tensor:
        inputs = self.preprocessor.tokenizer(
            language_instruction,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )
        return inputs["input_ids"].to(self.device)
