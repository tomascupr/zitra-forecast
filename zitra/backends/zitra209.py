"""Zitra-209M-v2: Pure single-backbone 209,312,256-parameter Timestral forecaster
with algebraically folded 24-layer FullLoRA-v2 weights (1.00x stock latency) and
optional 84-step nonnegative integer support snapping."""

import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from timestral import TimestralForecaster
from timestral.config import TimestralConfig
from timestral.model import TimestralModel


class Zitra209Forecaster(TimestralForecaster):
    def __init__(self, model: TimestralModel, policy: dict | None = None):
        super().__init__(model)
        self.policy = policy or {"integer_snap": True}

    @classmethod
    def from_pretrained(cls, path, device="cuda"):
        p = Path(path)
        cfg = TimestralConfig.load(p / "config.json")
        model = TimestralModel(cfg)
        sd = load_file(str(p / "model.safetensors"))
        model.load_state_dict(sd, strict=True)
        model = model.to(device).eval()
        model.transformer.rope.rebuild_inv_freq()
        pol_path = p / "policy.json"
        policy = json.loads(pol_path.read_text()) if pol_path.exists() else {"integer_snap": True}
        return cls(model, policy=policy)

    @torch.inference_mode()
    def forecast(self, context, prediction_length, observation_mask=None, *args, **kwargs):
        base = super().forecast(context, prediction_length, observation_mask, *args, **kwargs)
        if not self.policy.get("integer_snap", True):
            return base
        x, observed = self._as_context(context, observation_mask)
        if x.shape[-1] < 84:
            return base
        tail = x[..., -84:]
        eligible = (observed[..., -84:] & torch.isfinite(tail) & (tail >= 0) & (tail == tail.round())).all(-1)
        if not eligible.any():
            return base
        q = torch.where(eligible[..., None, None], base.quantiles.round(), base.quantiles)
        q = torch.sort(q, dim=-1).values
        return type(base)(quantiles=q, quantile_levels=base.quantile_levels)
