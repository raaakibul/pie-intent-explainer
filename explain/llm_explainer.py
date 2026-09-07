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
        
    def explain(self, ctx: ManeuverContext, maneuver: dict) -> str:
        import torch

        messages = build_messages(ctx, maneuver)
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()
    
class OpenAIExplainer(Explainer):
    def __init__(self, model_name: str = "gpt-4o", max_tokens: int = 200,
                 temperature: float = 0.3):
        from openai import OpenAI

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not set. Export it, or use --llm_backend hf instead."
            )
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature