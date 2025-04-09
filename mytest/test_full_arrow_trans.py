import torch
import os
import math
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from torch import Tensor
import time
import sys
import argparse
from typing import List

import torch.nn.functional as F
from PIL import Image
import numpy as np

import matplotlib.pyplot as plt
from pathlib import Path

from  mask import create_full_arrow_mask, create_half_arrow_mask
from flash_attn_ours import headwise_arrow_attn, headwise_half_arrow_attn, headwise_arrow_attn_trans
from torch.nn.attention.flex_attention import _mask_mod_signature



def sdpa_attn(q, k, v, mask=None):
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)


def generate_tiled_partial_sliding_window_per_head(lefts, rights, ttok_len: int, block_size=64) -> _mask_mod_signature:
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
        wl, wr = lefts[h] // block_size, rights[h] // block_size
        window_size_tiled = (wl + wr) 
        # window_size_tiled = head_list[h] // block_size
        return ((((q_tiled_idx - kv_tiled_idx) <= wl) & ((kv_tiled_idx - q_tiled_idx) <= wr)) | (q_idx < (ttok_len//block_size) * block_size) | (kv_idx < (ttok_len//block_size) * block_size)) & (window_size_tiled >= 0)

    sliding_window_mask = sliding_window_with_offset
    sliding_window_mask.__name__ = f"sliding_window_multiple_size"
    return sliding_window_mask

def flex_attn_wrapper(q, k, v):
    return flex_attention(q, k, v)

def flex_attn_block_mask_wrapper(q, k, v, block_mask):
    return flex_attention(q, k, v, block_mask=block_mask)

flex_attn_compiled = torch.compile(flex_attn_wrapper, dynamic=False)
flex_attn_block_mask_compiled = torch.compile(flex_attn_block_mask_wrapper, dynamic=False)


# def repeat_kv(kv: torch.Tensor, n_rep: int) -> torch.Tensor:
#     """
#     This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
#     num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
#     [1,2,3,4] => [1,1,2,2,3,3,4,4]
#     """
#     bs, nhkv, slen, hdim = kv.shape
#     if n_rep == 1:
#         return kv
#     kv = kv[:, :, None, :, :].expand(bs, nhkv, n_rep, slen, hdim)
#     return kv.reshape(bs, nhkv * n_rep, slen, hdim)

# def draw_mask(window_mask, i, save_dir):
#     window_img = window_mask
#     plt.imshow(window_img, cmap='gray', vmin=0, vmax=1)
#     plt.axis('off')  # 不显示坐标轴
#     plt.savefig(os.path.join(save_dir, f'window_img_{i}.png'), bbox_inches='tight', pad_inches=0)  # 保存图像
#     # plt.show() 

# flex_attention = torch.compile(flex_attention)

# FLEX_MASK_CACHE = {}
# def flex_headwin_attn(q: Tensor, k: Tensor, v: Tensor, n_im_tokens: int, n_text_tokens: int, spatial_mask, block_size=(128, 128)):

#     cache_key = (f"{n_im_tokens}_{n_text_tokens}_{block_size}", spatial_mask)

#     if cache_key in FLEX_MASK_CACHE:
#         block_mask = FLEX_MASK_CACHE[cache_key]
#     else:
#         def mask_function(b, h, q_idx, kv_idx):
#             cond1 = spatial_mask[h, q_idx, kv_idx]
#             return cond1

#         total_tokens = n_im_tokens + n_text_tokens


#         block_mask = create_block_mask(
#             mask_function,
#             B=None,
#             H=q.shape[1],
#             Q_LEN=total_tokens,
#             KV_LEN=total_tokens,
#             BLOCK_SIZE=block_size,
#         )
#         FLEX_MASK_CACHE[cache_key] = block_mask
#     output = flex_attention(q, k, v, block_mask=block_mask)
#     return output


def gen_win_size_uniform(n_head, min_val, max_val, device, dtype=torch.int32):
    win_size_lr = torch.randint(min_val, max_val+1, (n_head,), dtype=dtype, device=device) # * 10
    window_sizes = torch.zeros((n_head, 2), dtype=dtype, device=device)
    for i in range(n_head):
        window_sizes[i, 0] = torch.randint(1, win_size_lr[i], (1,))
        window_sizes[i, 1] = win_size_lr[i] - window_sizes[i, 0]
    
    return window_sizes, win_size_lr

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch", type=int, default=2)
    parser.add_argument("-seq", "--seqlen", type=int, default=2048)
    parser.add_argument("-seqv", "--seqlen-vision", type=int, default=1024)
    parser.add_argument("-seqt", "--seqlen-text", type=int, default=128)
    parser.add_argument("-nh", "--num-head", type=int, default=8)
    parser.add_argument("-d", "--head-dim", type=int, default=64)
    parser.add_argument("-ws1", "--window-size1", type=int, default=128)
    parser.add_argument("-ws2", "--window-size2", type=int, default=256)
    parser.add_argument("--min-val", type=int, default=128)
    parser.add_argument("--max-val", type=int, default=256)
    parser.add_argument("-fn", "--full-num", type=int, default=0)
    parser.add_argument("--using-bench", action="store_true")
    parser.add_argument("--headwin", action="store_true")
    
    args = parser.parse_args(argv)

    # B, N, H, D = args.batch, args.seqlen, args.num_head, args.head_dim                    # 2, 1024, 32, 128
    B, N, H, D = args.batch, args.seqlen_vision + args.seqlen_text, args.num_head, args.head_dim                    # 2, 1024, 32, 128
    device = torch.device("cuda")

    block_sizes = (128, 128)


    q, k, v = torch.randn(B * 3, N, H, D, dtype=torch.float16, device=device).split(B, dim=0)
    
    window_sizes = torch.zeros((H, 2), device="cuda", dtype=torch.int32)
    window_sizes, _ = gen_win_size_uniform(H, 128, 512, device=device)
    # window_sizes[:, :] = -1
    window_sizes[:H//2, :] = args.window_size1

    window_sizes[H//2:, :] = args.window_size2
    window_sizes[:args.full_num, :] = -1

    print(window_sizes)


    window_masks_full_arrow, _ = create_full_arrow_mask(H, args.seqlen_vision, args.seqlen_text, window_sizes, (128, 128))

    window_masks_half_arrow, _ = create_half_arrow_mask(H, args.seqlen_vision, args.seqlen_text, window_sizes, (128, 128))

    
    # window_masks_cpu, block_mask_cpu = create_mask_new(H, args.seqlen_vision, args.seqlen_text, window_sizes, (128, 128), device=torch.device("cpu"))
    save_dir = "/root/surd/flash_attn_mod/flash-attention-2.7.2/mytest/mask_img"

    o_headwise_full_arrow_trans = headwise_arrow_attn_trans(q, k, v, window_sizes=window_sizes, seqlen_k_vision=args.seqlen_vision, seqlen_q_vision=args.seqlen_vision)

    # wh, wtr, wtc = window_masks_full_arrow.shape

    # flex_masks_full_arrow = torch.zeros((wh, (wtr + block_sizes[0] - 1) // block_sizes[0] * block_sizes[0], (wtc + block_sizes[1] - 1) // block_sizes[1] * block_sizes[1]), device=window_masks_full_arrow.device, dtype=window_masks_full_arrow.dtype)
    # flex_masks_full_arrow[:, :wtr, :wtc] = window_masks_full_arrow
    window_size_lefts, window_size_rights = window_sizes[:, 0] , window_sizes[:, 1] 
    print((args.seqlen_text  + 128 - 1)// 128 * 128)
    sliding_window_mask = generate_tiled_partial_sliding_window_per_head(window_size_lefts, window_size_rights, ttok_len=(args.seqlen_text  + 128 - 1)// 128 * 128, block_size=128)
    # print("sliding window no problem")
    block_mask = create_block_mask(sliding_window_mask, None, H, N, N, device="cuda", _compile=True, BLOCK_SIZE=128)
    print(block_mask)
    o_flex_full_arrow_trans = flex_attn_block_mask_compiled(q.permute(0, 2, 1, 3).contiguous(), 
                               k.permute(0, 2, 1, 3).contiguous(), 
                               v.permute(0, 2, 1, 3).contiguous(), 
                               block_mask=block_mask).permute(0, 2, 1, 3).contiguous()
    

    torch.cuda.synchronize()
    
    print(f"shape: batch: {B}, seqlen: {N}, seqlen-vision: {args.seqlen_vision}, seqlen-text: {args.seqlen_text}, num_head: {H}, head_dim: {D}")
    print(f"{'-'*64}\n{'full arrow attn':^21}\n{'-'*64}")
    print("{:<25}=={:>25}: \033[92m{!s:6>}\033[0m".format("o_flex_full_arrow_trans", "o_headwise_full_arrow_trans", torch.allclose(o_flex_full_arrow_trans, o_headwise_full_arrow_trans, 1e-3, 1e-3)))
    # torch.set_printoptions(threshold=float("inf"))
    # print(o_flex_full_arrow_trans[0, :, 0, :] - o_headwise_full_arrow_trans[0, :, 0, :])
    


if __name__=="__main__":
    main(sys.argv[1:])