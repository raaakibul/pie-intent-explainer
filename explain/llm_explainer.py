from __future__ import annotations
import os
from abc import ABC, abstractmethod
from explain.prompt_templates import ManeuverContext, build_messages, recommend_maneuver

class Explainer(ABC):
    @abstractmethod
    def explain(self, ctx: ManeuverContext, maneuver: dict) -> str:
        ...
    def explain_from_model_output(self, intent_label: str, intent_confidence: float,
                                   pred_traj_xy, ego_speed_mps: float, frame_dt_s: float,
                                   cfg: dict) -> dict:
        """Convenience wrapper: build context + maneuver + explanation in one call."""
        ctx = ManeuverContext(
            intent_label=intent_label,
            intent_confidence=intent_confidence,
            pred_traj_xy=pred_traj_xy,
            ego_speed_mps=ego_speed_mps,
            frame_dt_s=frame_dt_s,
        )
        maneuver = recommend_maneuver(ctx, cfg)
        text = self.explain(ctx, maneuver)
        return {"context": ctx, "maneuver": maneuver, "explanation": text}
    
class HFExplainer(Explainer):
    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
                 load_in_4bit: bool = True, max_new_tokens: int = 160,
                 temperature: float = 0.3):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        quant_kwargs = {}
        if load_in_4bit and torch.cuda.is_available():
            from transformers import BitsAndBytesConfig
            quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            **quant_kwargs,
        )
        self.model.eval()