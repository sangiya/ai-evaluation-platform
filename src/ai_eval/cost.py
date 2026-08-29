from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import tiktoken

_PRICES: Dict[str, Dict[str, float]] = {
    "gpt-4o":                      {"in": 5.00,   "out": 15.00,  "tok": "o200k"},
    "gpt-4o-mini":                 {"in": 0.15,   "out": 0.60,   "tok": "o200k"},
    "gpt-4-turbo":                 {"in": 10.00,  "out": 30.00,  "tok": "cl100k"},
    "gpt-3.5-turbo":               {"in": 0.50,   "out": 1.50,   "tok": "cl100k"},
    "claude-3-5-sonnet-20241022":  {"in": 3.00,   "out": 15.00,  "tok": "approx"},
    "claude-3-5-haiku-20241022":   {"in": 0.80,   "out": 4.00,   "tok": "approx"},
    "claude-3-opus-20240229":      {"in": 15.00,  "out": 75.00,  "tok": "approx"},
    "claude-3-haiku-20240307":     {"in": 0.25,   "out": 1.25,   "tok": "approx"},
    "gemini-1.5-pro":              {"in": 3.50,   "out": 10.50,  "tok": "approx"},
    "gemini-1.5-flash":            {"in": 0.075,  "out": 0.30,   "tok": "approx"},
}


@dataclass
class CostEstimate:
    model_id: str
    input_tokens: int
    output_tokens: int
    input_cost_usd: float
    output_cost_usd: float
    total_cost_usd: float


def _token_count(text: str, tokenizer_family: str) -> int:
    if tokenizer_family in ("cl100k", "o200k"):
        enc_name = "cl100k_base" if tokenizer_family == "cl100k" else "o200k_base"
        return len(tiktoken.get_encoding(enc_name).encode(text))
    return max(1, round(len(text) / 4.0))


def cost_estimate(model_id: str, input_text: str, output_text: str = "", *, output_tokens: int | None = None) -> CostEstimate:
    if model_id not in _PRICES:
        raise ValueError(f"Unknown model '{model_id}'. Available: {sorted(_PRICES)}")

    pricing = _PRICES[model_id]
    in_tok = _token_count(input_text, pricing["tok"])
    out_tok = output_tokens if output_tokens is not None else _token_count(output_text, pricing["tok"])

    in_cost = (in_tok / 1_000_000) * pricing["in"]
    out_cost = (out_tok / 1_000_000) * pricing["out"]

    return CostEstimate(
        model_id=model_id,
        input_tokens=in_tok,
        output_tokens=out_tok,
        input_cost_usd=in_cost,
        output_cost_usd=out_cost,
        total_cost_usd=in_cost + out_cost,
    )
