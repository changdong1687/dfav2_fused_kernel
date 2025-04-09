from typing import List
from torch.nn.attention.flex_attention import _mask_mod_signature
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from functools import partial
from time import time
import torch
import math
import random

def generate_partial_sliding_window_per_head(head_list: List, vtok_len: int, block_size=64) -> _mask_mod_signature:
    """Generates a sliding window attention mask with a given window size.
    Args:
        window_size: The size of the sliding window.

    Note:
        We assume that the window size represents the lookback size and we mask out all future tokens
        similar to causal masking.
    """

    def sliding_window_with_offset(b, h, q_idx, kv_idx):
        window_size = head_list[h] 
        return ((torch.abs(q_idx - kv_idx) <= window_size // 2) | (q_idx >= vtok_len) | (kv_idx >= vtok_len)) & (window_size > 0)

    sliding_window_mask = sliding_window_with_offset
    sliding_window_mask.__name__ = f"sliding_window_multiple_size"
    return sliding_window_mask

def generate_tiled_partial_sliding_window_per_head(lefts, rights, vtok_len: int, block_size=64) -> _mask_mod_signature:
    """Generates a sliding window attention mask with a given window size.
    Args:
        window_size: The size of the sliding window.

    Note:
        We assume that the window size represents the lookback size and we mask out all future tokens
        similar to causal masking.
    """

    def sliding_window_with_offset(b, h, q_idx, kv_idx):
        q_tiled_idx = q_idx // block_size
        kv_tiled_idx = kv_idx // block_size
        wl, wr = lefts[h], rights[h]
        window_size_tiled = (wl + wr) // block_size
        # window_size_tiled = head_list[h] // block_size
        return (((q_tiled_idx - kv_tiled_idx) <= wl) | ((kv_tiled_idx - q_tiled_idx) <= wr) | (q_idx >= (vtok_len//block_size) * block_size) | (kv_idx >= (vtok_len//block_size) * block_size)) & (window_size_tiled > 0)
        # return ((torch.abs(q_tiled_idx - kv_tiled_idx) <= window_size_tiled // 2) | (q_idx >= (vtok_len//block_size) * block_size) | (kv_idx >= (vtok_len//block_size) * block_size)) & (window_size_tiled > 0)

    sliding_window_mask = sliding_window_with_offset
    sliding_window_mask.__name__ = f"sliding_window_multiple_size"
    return sliding_window_mask

def generate_partial_sliding_block_window(window_size: int, vtok_len: int, block_size=64) -> _mask_mod_signature:
    """Generates a sliding window attention mask with a given window size.
    Args:
        window_size: The size of the sliding window.

    Note:
        We assume that the window size represents the lookback size and we mask out all future tokens
        similar to causal masking.
    """

    def sliding_window_with_offset(b, h, q_idx, kv_idx):
        blockq_idx = q_idx // block_size * block_size

        return ((kv_idx >= blockq_idx - window_size // 2 * block_size) & (kv_idx <= blockq_idx + (window_size // 2 + 1) * block_size)) | (q_idx >= vtok_len) | (kv_idx >= vtok_len)

    sliding_window_mask = sliding_window_with_offset
    sliding_window_mask.__name__ = f"sliding_window_{window_size}"
    return sliding_window_mask

def flex_attn_wrapper(q, k, v):
    return flex_attention(q, k, v)

def flex_attn_block_mask_wrapper(q, k, v, block_mask):
    return flex_attention(q, k, v, block_mask=block_mask)

flex_attn_compiled = torch.compile(flex_attn_wrapper, dynamic=False)
flex_attn_block_mask_compiled = torch.compile(flex_attn_block_mask_wrapper, dynamic=False)

def transform_dense_block_mask(mask, block_size=(64, 64)):
    _, nhead, H, W = mask.shape
    _H = H // block_size[0] * block_size[0]
    _W = W // block_size[1] * block_size[1]
    _mask = mask[:, :_H, :_W].view(nhead, H // block_size[0], block_size[0], W // block_size[1], block_size[1])
    _mask_max = _mask.amax(dim=[2, 4], keepdim=True)
    _mask |= _mask_max
    mask[:, :_H, :_W] = _mask.reshape(nhead, _H, _W)
    return mask

def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


seqlen_vision, seqlen_text = 16384, 512
B = 4
S = seqlen_vision + seqlen_text
H = 24
D = 64
bs = int(math.sqrt(S - seqlen_text))
query = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
key = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
value = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)



window_sizes = torch.zeros((H, 2), dtype=torch.int32, device="cuda")

# head_list = torch.randint(12, 24, (H,), device="cuda", dtype=torch.int64)
window_sizes[0] = torch.tensor((2, 4), device="cuda", dtype=torch.int32) * 128
window_sizes[1:3] = torch.tensor((2, 5), device="cuda", dtype=torch.int32) * 128
window_sizes[3:] = torch.tensor((5, 8), device="cuda", dtype=torch.int32) * 128


# head_list = torch.tensor(head_list, dtype=torch.int64, device=query.device)

window_size_lefts = window_sizes[:, 0]
window_size_rights = window_sizes[:, 1]

print(f"window-size left shape: {window_size_lefts.shape}")
# print(f"head_list shape: {head_list.shape}")

sliding_window_mask = generate_tiled_partial_sliding_window_per_head(window_size_lefts, window_size_rights, vtok_len=(S - seqlen_text) // 128 * 128)
block_mask = create_block_mask(sliding_window_mask, None, H, S, S, device="cuda", _compile=True, BLOCK_SIZE=128)
print(query.shape)
for _ in range(100):
    output = flex_attn_block_mask_compiled(query, key, value, block_mask=block_mask)

# block_mask_dict = {}
# window_sizes_dict = {}

# candidate = [(S - seqlen_text)//4, (S - seqlen_text) // 8, (S - seqlen_text) // 16]
# # candidate = [(S - seqlen_text)*2, (S - seqlen_text) // 8, (S - seqlen_text) // 16, 0]
# for m in range(100):
#     head_list = []
#     for i in range(H):
#         head_list.append(random.choice(candidate))
#         window_sizes[i, :] = head_list[i] // 2

#     head_list = torch.tensor(head_list, dtype=torch.int64, device=query.device)
#     # print(window_sizes)
#     # print(head_list)
    

#     # tensor([   0, 8192,    0, 8192, 8192, 8192, 8192,    0, 8192, 8192, 8192, 8192,
#     #         8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192,    0, 8192])

#     sliding_window_mask = generate_tiled_partial_sliding_window_per_head(head_list, vtok_len=(S - seqlen_text) // 128 * 128)
#     block_mask = create_block_mask(sliding_window_mask, None, H, S, S, device="cuda", _compile=True, BLOCK_SIZE=128)
#     block_mask_dict[m] = block_mask
#     window_sizes_dict[m] = window_sizes
# # print(block_mask.shape)
# # print(block_mask.to_string(limit=24))
# # block_mask = create_block_mask(causal_mask, B, H, S, S, device="cuda", _compile=True)

# output = flex_attn_block_mask_compiled(query, key, value, block_mask=block_mask_dict[j])
# # output = flex_attn_compiled(query, key, value)
# torch.cuda.synchronize()
# o_headWin = flash_attn_func_headwin(query.permute(0, 2, 1, 3).contiguous(), 
#                                     key.permute(0, 2, 1, 3).contiguous(), 
#                                     value.permute(0, 2, 1, 3).contiguous(), 
#                                     window_sizes=window_sizes_dict[j], 
#                                     seqlen_q_vision=seqlen_vision, 
#                                     seqlen_k_vision=seqlen_vision).permute(0, 2, 1, 3).contiguous()
# print(f"iter {i}-th {j}-th: o_headWin==o_flex: {torch.allclose(o_headWin, output, 1e-3, 1e-3)}")
# print(o_headWin - output)

# # print(window_sizes_dict[0])
# print(block_mask_dict[0].shape)