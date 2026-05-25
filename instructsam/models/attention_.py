import torch
from transformers import AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.integrations.flash_attention import flash_attention_forward

def fused_attention(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask=None,
    **kwargs,
):
    if getattr(module, "_is_in_language_model", False):
        return sdpa_attention_forward(module, query, key, value, attention_mask, **kwargs)
    else:
        try:
            return flash_attention_forward(module, query, key, value, attention_mask, **kwargs)
        except Exception:
            return sdpa_attention_forward(module, query, key, value, attention_mask, **kwargs)

AttentionInterface.register("fused_attention", fused_attention)
