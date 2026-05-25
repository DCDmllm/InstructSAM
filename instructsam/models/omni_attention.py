import numpy as np
import torch
from torch.nn.attention.flex_attention import BlockMask
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

if torch.cuda.is_available():
    flex_attention = torch.compile(flex_attention)


def omni_attn_mask_naive(attention_mask, modalities, inverted=False):
    # print('omni_attn_mask_naive')
    B, L = attention_mask.shape
    causal_lm_mask = torch.tril(torch.ones((L, L), dtype=torch.bool, device=attention_mask.device))
    padding_mask = attention_mask.unsqueeze(1).unsqueeze(2)
    attention_mask = causal_lm_mask.unsqueeze(0).unsqueeze(1) & padding_mask

    if len(modalities)>0:
        for b in range(B):
            modality_batch = modalities[b]
            if len(modality_batch)>0:
                for start, end in modality_batch:
                    attention_mask[b, :, start:end, start:end] = True
    attention_mask = attention_mask.bool()
    if inverted:
        attention_mask = attention_mask.to(torch.long)
        inverted_attention_mask = 1 - attention_mask
        inverted_attention_mask = inverted_attention_mask.masked_fill(
            inverted_attention_mask.to(torch.bool), torch.iinfo(torch.long).min
        )
        return inverted_attention_mask
    else:
        return attention_mask


def full_attn_mask(L1, L2, attention_mask, inverted=False):
    # print('full_attn_mask')
    full_mask = torch.ones((1, 1, L1, L2), dtype=torch.bool, device=attention_mask.device)
    
    return full_mask

def fused_full_attn_mask(L1, L2, attention_mask, inverted=False):
    
    full_mask = torch.ones((1, 1, L1, L2), dtype=torch.bool, device=attention_mask.device)
    full_mask[0, 0, 0,-L1+1:] = False
    full_mask[0, 0, 1,-L1+2:] = False
    return full_mask

