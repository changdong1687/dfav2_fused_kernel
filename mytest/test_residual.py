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


from flash_attn_ours import headwise_arrow_attn, headwise_arrow_attn_with_residual


from torch.nn.attention.flex_attention import _mask_mod_signature
import sys
import argparse
import torch
import numpy as np

import torch

from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from torch import Tensor
import time

import torch.nn.functional as F
import numpy as np

from typing import List

def generate_tiled_partial_sliding_window_per_head(lefts, rights, vtok_len: int, block_size=64) -> _mask_mod_signature:             # for compute the arrow attention
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
        wl, wr = lefts[h]//block_size, rights[h]//block_size
        window_size_tiled = (wl + wr)
        # window_size_tiled = head_list[h] // block_size
        # return (((q_tiled_idx - kv_tiled_idx) <= wl) | ((kv_tiled_idx - q_tiled_idx) <= wr)) & (window_size_tiled > 0)
        return (((q_tiled_idx - kv_tiled_idx) <= wl) & ((kv_tiled_idx - q_tiled_idx) <= wr) | (q_idx >= (vtok_len//block_size) * block_size) | (kv_idx >= (vtok_len//block_size) * block_size)) & (window_size_tiled > 0)
        # return ((torch.abs(q_tiled_idx - kv_tiled_idx) <= window_size_tiled // 2) | (q_idx >= (vtok_len//block_size) * block_size) | (kv_idx >= (vtok_len//block_size) * block_size)) & (window_size_tiled > 0)

    sliding_window_mask = sliding_window_with_offset
    sliding_window_mask.__name__ = f"sliding_window_multiple_size"
    return sliding_window_mask

def generate_tiled_partial_sliding_window_per_head_residual(lefts, rights, vtok_len: int, block_size=64) -> _mask_mod_signature:        # for compute the residual part

    def sliding_window_with_offset(b, h, q_idx, kv_idx):
        q_tiled_idx = q_idx // block_size
        kv_tiled_idx = kv_idx // block_size
        wl, wr = lefts[h] // block_size, rights[h] // block_size
        window_size_tiled = (wl + wr) 
        # window_size_tiled = head_list[h] // block_size
        return ((((q_tiled_idx - kv_tiled_idx) > wl) | ((kv_tiled_idx - q_tiled_idx) > wr)) & (q_idx < (vtok_len//block_size) * block_size) & (kv_idx < (vtok_len//block_size) * block_size)) & (window_size_tiled >= 0)

    sliding_window_mask = sliding_window_with_offset
    sliding_window_mask.__name__ = f"sliding_window_multiple_size"
    
    return sliding_window_mask

def flex_attn_wrapper(q, k, v):
    return flex_attention(q, k, v)

def flex_attn_block_mask_wrapper(q, k, v, block_mask):
    return flex_attention(q, k, v, block_mask=block_mask)

flex_attn_compiled = torch.compile(flex_attn_wrapper, dynamic=False)
flex_attn_block_mask_compiled = torch.compile(flex_attn_block_mask_wrapper, dynamic=False)


"""
window_size_lefts, window_size_rights = window_sizes[:, 0] , window_sizes[:, 1] 
sliding_window_mask = generate_tiled_partial_sliding_window_per_head_trans(window_size_lefts, window_size_rights, ttok_len=seqlen_text // 128 * 128, block_size=128)
# print("sliding window no problem")
block_mask = create_block_mask(sliding_window_mask, None, H, N, N, device="cuda", _compile=True, BLOCK_SIZE=128)
# print("block mask no problem")
assert block_mask is not None, "Error: create_block_mask() returned None!"
print("block_mask shape:", block_mask.shape)
print(f"mask sparsity: {block_mask.sparsity()}")
qp, kp, vp = q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)

for _ in range(100):
    flex_attn_block_mask_compiled(qp, kp, vp, block_mask=block_mask)
    torch.cuda.synchronize()
"""



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
    parser.add_argument("-d", "--head-dim", type=int, default=128)
    parser.add_argument("-ws1", "--window-size1", type=int, default=256)
    parser.add_argument("-ws2", "--window-size2", type=int, default=128)
    parser.add_argument("--min-val", type=int, default=128)
    parser.add_argument("--max-val", type=int, default=256)
    parser.add_argument("-fn", "--full-num", type=int, default=0)
    parser.add_argument("--res-num", type=int, default=4)
    parser.add_argument("--using-bench", action="store_true")
    parser.add_argument("--headwin", action="store_true")
    
    args = parser.parse_args(argv)

    # B, N, H, D = args.batch, args.seqlen, args.num_head, args.head_dim                    # 2, 1024, 32, 128
    B, N, H, D = args.batch, args.seqlen_vision + args.seqlen_text, args.num_head, args.head_dim                    # 2, 1024, 32, 128
    device = torch.device("cuda")

    window_sizes = torch.zeros((H, 2), device="cuda", dtype=torch.int32)
    use_residual = torch.zeros((H, 1), device="cuda", dtype=torch.int32)
    
    window_sizes[:H//2, :] = args.window_size1
    window_sizes[H//2:, :] = args.window_size2
    
    window_sizes[:args.full_num, :] = -1
    q, k, v = torch.randn(B * 3, N, H, D, dtype=torch.float16, device=device).split(B, dim=0)

    window_size_lefts, window_size_rights = window_sizes[:, 0], window_sizes[:, 1]
    sliding_window_mask = generate_tiled_partial_sliding_window_per_head(window_size_lefts, window_size_rights, vtok_len=args.seqlen_vision // 128 * 128, block_size=128)
    block_mask = create_block_mask(sliding_window_mask, None, H, N, N, device="cuda", _compile=True, BLOCK_SIZE=128)
    
    qp, kp, vp = q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)
    o_flex = flex_attn_block_mask_compiled(qp, kp, vp, block_mask=block_mask).permute(0, 2, 1, 3)

    sliding_window_mask_res =  generate_tiled_partial_sliding_window_per_head_residual(window_size_lefts, window_size_rights, vtok_len=args.seqlen_vision // 128 * 128, block_size=128)
    block_mask_res = create_block_mask(sliding_window_mask_res, None, H, N, N, device="cuda", _compile=True, BLOCK_SIZE=128)
    o_flex_res = flex_attn_block_mask_compiled(qp, kp, vp, block_mask=block_mask_res).permute(0, 2, 1, 3)

    use_residual[:args.full_num, :] = 1
    o_headwise_full_arrow_res, res = headwise_arrow_attn_with_residual(q, k, v, window_sizes=window_sizes, need_residual=use_residual,  seqlen_k_vision=args.seqlen_vision, seqlen_q_vision=args.seqlen_vision)
    o_headwise_full_arrow = headwise_arrow_attn(q, k, v, window_sizes=window_sizes, seqlen_k_vision=args.seqlen_vision, seqlen_q_vision=args.seqlen_vision)
    flex_eq_o_arrow_res = torch.allclose(o_flex, o_headwise_full_arrow_res, 1e-3, 1e-3)
    if flex_eq_o_arrow_res:
        print("\033[32m")
    else:
        print("\033[31m")
    print(f"o_flex == o_hw_fa_res: {flex_eq_o_arrow_res}")
    print("\033[0m")

    flex_res_eq_o_res = torch.allclose(o_flex_res[:, :, :args.full_num, :], res[:, :, :args.full_num, :], 1e-3, 1e-3)
    if flex_res_eq_o_res:
        print("\033[32m")
    else:
        print("\033[31m")
    print(f"o_res == o_flex_res: {flex_res_eq_o_res}")
    print("\033[0m")

    flex_eq_o_hw_fa = torch.allclose(o_flex, o_headwise_full_arrow, 1e-3, 1e-3)
    if flex_eq_o_hw_fa:
        print("\033[32m")
    else:
        print("\033[31m")
    print(f"o_flex == o_hw_fa: {flex_eq_o_hw_fa}")
    print("\033[0m")

    o_res_eq_o_hw_fa = torch.allclose(o_headwise_full_arrow_res, o_headwise_full_arrow, 1e-3, 1e-3)
    if o_res_eq_o_hw_fa:
        print("\033[32m")
    else:
        print("033[31m")
    print(f"o_res == o_fa: {o_res_eq_o_hw_fa}")
    print("\033[0m")

    ##################
    ## measure time ##
    ##################
    for _ in range(100):
        headwise_arrow_attn_with_residual(q, k, v, window_sizes=window_sizes, need_residual=use_residual,  seqlen_k_vision=args.seqlen_vision, seqlen_q_vision=args.seqlen_vision)

    start_res = torch.cuda.Event(True)
    end_res = torch.cuda.Event(True)
    cost_res = []
    for _ in range(100):
        start_res.record()
        headwise_arrow_attn_with_residual(q, k, v, window_sizes=window_sizes, need_residual=use_residual,  seqlen_k_vision=args.seqlen_vision, seqlen_q_vision=args.seqlen_vision)
        end_res.record()
        torch.cuda.synchronize()
        tot_res = start_res.elapsed_time(end_res)
        cost_res.append(tot_res)
    time_res = np.mean(np.array(cost_res))

    for _ in range(100):
        headwise_arrow_attn(q, k, v, window_sizes=window_sizes, seqlen_k_vision=args.seqlen_vision, seqlen_q_vision=args.seqlen_vision)

    start_fa_no_res = torch.cuda.Event(True)
    end_fa_no_res = torch.cuda.Event(True)
    cost_fa_no_res = []
    for _ in range(100):
        start_fa_no_res.record()
        headwise_arrow_attn(q, k, v, window_sizes=window_sizes, seqlen_k_vision=args.seqlen_vision, seqlen_q_vision=args.seqlen_vision)
        end_fa_no_res.record()
        torch.cuda.synchronize()
        tot_no_res = start_fa_no_res.elapsed_time(end_fa_no_res)
        cost_fa_no_res.append(tot_no_res)
    time_fa_no_res = np.mean(np.array(cost_fa_no_res))

    from flash_attn_original import flash_attn_func as flash_attn_func_original

   

    for _ in range(100):
        flash_attn_func_original(q, k, v, window_size=(-1 , -1))
    torch.cuda.synchronize()

    start_ori = torch.cuda.Event(True)
    end_ori =  torch.cuda.Event(True)
    ori_full_times = []
    for _ in range(100): 
        start_ori.record()
        flash_attn_func_original(q, k, v, window_size=(-1 , -1))
        end_ori.record()
        torch.cuda.synchronize()
        tot_original = (start_ori.elapsed_time(end_ori)) 
        ori_full_times.append(tot_original)
    ori_full_mean = np.mean(np.array(ori_full_times))

    print(q.shape)

    print(f"using residual time: {time_res:>16.4f}")
    print(f"without residual time: {time_fa_no_res:>16.4f}")
    print(f"origianl full attn time: {ori_full_mean:>16.4f}")
if __name__=="__main__":
    main(sys.argv[1:])