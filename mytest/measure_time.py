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

from mask import create_full_arrow_mask, create_half_arrow_mask, create_headwise_window_mask

def generate_tiled_partial_sliding_window_per_head_trans(lefts, rights, ttok_len: int, block_size=64) -> _mask_mod_signature:
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
        wl, wr = lefts[h] // block_size, rights[h] // block_size
        window_size_tiled = (wl + wr) 
        # window_size_tiled = head_list[h] // block_size
        return ((((q_tiled_idx - kv_tiled_idx) <= wl) & ((kv_tiled_idx - q_tiled_idx) <= wr)) | (q_idx >= (vtok_len//block_size) * block_size) | (kv_idx >= (vtok_len//block_size) * block_size)) & (window_size_tiled >= 0)

    sliding_window_mask = sliding_window_with_offset
    sliding_window_mask.__name__ = f"sliding_window_multiple_size"
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

# %%

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", "--k-choice", type=int, default=1)
    parser.add_argument("-m", "--token-choice", type=int, default=1)
    parser.add_argument("--ori", action="store_false")
    parser.add_argument("--full-arrow", action="store_false")
    parser.add_argument("--flex-full", action="store_false")
    parser.add_argument("--half-arrow", action="store_false")
    parser.add_argument("--flex-half", action="store_false")
    parser.add_argument("--flex-trans", action="store_false")
    parser.add_argument("--full-trans", action="store_false")
    args = parser.parse_args()

    num_head = 24

    head_dim = 64

    H, D = num_head, head_dim
    device = torch.device("cuda")

    window_sizes = torch.zeros((H, 2), device=device, dtype=torch.int32)
    if args.token_choice == 1:
        batch = 8
        seqlen_vision = 4096        # 16384
        seqlen_text = 333           # 512
        if args.k_choice == 1:
            window_sizes[0] = torch.tensor((0, 0), device=device, dtype=torch.int32) * 128
            window_sizes[1:4] = torch.tensor((1, 3), device=device, dtype=torch.int32) * 128
            window_sizes[4:] = torch.tensor((1, 2), device=device, dtype=torch.int32) * 128

        elif args.k_choice == 2:
            window_sizes[0] = torch.tensor((7, 8), device=device, dtype=torch.int32) * 128
            window_sizes[1:] = torch.tensor((7, 7), device=device, dtype=torch.int32) * 128
        elif args.k_choice == 3:

            window_sizes[0] = torch.tensor((13, 16), device=device, dtype=torch.int32) * 128
            window_sizes[1:4] = torch.tensor((13, 15), device=device, dtype=torch.int32) * 128
            window_sizes[4:8] = torch.tensor((14, 16), device=device, dtype=torch.int32) * 128
            window_sizes[8:] = torch.tensor((14, 14), device=device, dtype=torch.int32) * 128
    elif args.token_choice == 2:
        batch = 8
        seqlen_vision = 4096        # 16384
        seqlen_text = 512
        if args.k_choice == 1:
            window_sizes[:4] = torch.tensor((0, 0), device=device, dtype=torch.int32) * 128
            window_sizes[4:] = torch.tensor((0, 1), device=device, dtype=torch.int32) * 128
            # window_sizes[:9] = torch.tensor((0, 0), device=device, dtype=torch.int32) * 128
            # window_sizes[9:] = torch.tensor((0, 1), device=device, dtype=torch.int32) * 128
        elif args.k_choice == 2:
            window_sizes[0] = torch.tensor((2, 4), device=device, dtype=torch.int32) * 128
            window_sizes[1:3] = torch.tensor((2, 5), device=device, dtype=torch.int32) * 128
            window_sizes[3:] = torch.tensor((5, 8), device=device, dtype=torch.int32) * 128
        elif args.k_choice == 3:
            window_sizes[:] = torch.tensor((13, 14), device=device, dtype=torch.int32) * 128
    elif args.token_choice == 3:
        batch = 1
        seqlen_vision = 16384        # 16384
        seqlen_text = 512           # 512
        if args.k_choice == 1:
            window_sizes[:9] = torch.tensor((12, 15), device=device, dtype=torch.int32) * 128
            window_sizes[9:] = torch.tensor((13, 13), device=device, dtype=torch.int32) * 128
        elif args.k_choice == 2:
            window_sizes[:2] = torch.tensor((34, 34), device=device, dtype=torch.int32) * 128
            window_sizes[2:4] = torch.tensor((36, 36), device=device, dtype=torch.int32) * 128
            window_sizes[4:] = torch.tensor((33, 35), device=device, dtype=torch.int32) * 128
        elif args.k_choice == 3:
            window_sizes[:8] = torch.tensor((36, 65), device=device, dtype=torch.int32) * 128
            window_sizes[8:10] = torch.tensor((42, 57), device=device, dtype=torch.int32) * 128
            window_sizes[10:] = torch.tensor((58, 65), device=device, dtype=torch.int32) * 128
    elif args.token_choice == 4:
        batch = 1
        seqlen_vision = 65536        # 16384
        seqlen_text = 512           # 512
        if args.k_choice == 1:
            window_sizes[:21] = torch.tensor((64, 65), device=device, dtype=torch.int32) * 128
            window_sizes[21:] = torch.tensor((65, 66), device=device, dtype=torch.int32) * 128
        elif args.k_choice == 2:
            window_sizes[:4] = torch.tensor((137, 143), device=device, dtype=torch.int32) * 128
            window_sizes[4] = torch.tensor((147, 148), device=device, dtype=torch.int32) * 128
            window_sizes[5:] = torch.tensor((148, 148), device=device, dtype=torch.int32) * 128

        elif args.k_choice == 3:
            window_sizes[:] = torch.tensor((253, 254), device=device, dtype=torch.int32) * 128
    
    # print(window_sizes)
    print(f"seqlen_vision: {seqlen_vision} seqlen_text: {seqlen_text}")



    B, N, = batch, seqlen_vision + seqlen_text, 
    block_size = (128, 128)
    q, k, v = torch.randn(B * 3, N, H, D, dtype=torch.float16, device=device).split(B, dim=0)

    if args.flex_full:
        
        flex_full_arrow_times= []
        
        # win_flex = window_sizes.to(torch.int64)
        window_size_lefts, window_size_rights = window_sizes[:, 0] , window_sizes[:, 1] 
        sliding_window_mask = generate_tiled_partial_sliding_window_per_head(window_size_lefts, window_size_rights, vtok_len=seqlen_vision // 128 * 128, block_size=128)
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

        
        start_flex_fa = torch.cuda.Event(True)
        end_flex_fa = torch.cuda.Event(True)
        for _ in range(100): 
            start_flex_fa.record()
            o_flex_full_arrow= flex_attn_block_mask_compiled(qp, kp, vp, block_mask=block_mask)
            end_flex_fa.record()
            torch.cuda.synchronize()
            tot_flex_full_arrow = (start_flex_fa.elapsed_time(end_flex_fa)) 
            flex_full_arrow_times.append(tot_flex_full_arrow)
        flex_full_arrow_mean = np.mean(np.array(flex_full_arrow_times))
        print(f"flex full arrow time: {flex_full_arrow_mean}")

        
    if args.full_arrow:
        # from flash_attn_ours import  headwise_arrow_attn
        from dfav2 import  headwise_arrow_attn

        hw_fa_times = []
        flex_full_arrow_times = []
        output_hw_full_arrows_rights = []
        o_ori_vs_o_fa = []

        for _ in range(100):
            headwise_arrow_attn(q, k, v, window_sizes=window_sizes, seqlen_q_vision = seqlen_vision, seqlen_k_vision = seqlen_vision)
        torch.cuda.synchronize()

        start_hw_full_arrow = torch.cuda.Event(True)
        end_hw_full_arrow =  torch.cuda.Event(True)

                
        for _ in range(100): 
            start_hw_full_arrow.record()
            o_hw_full_arrow = headwise_arrow_attn(q, k, v, window_sizes=window_sizes, seqlen_q_vision = seqlen_vision, seqlen_k_vision = seqlen_vision)
            end_hw_full_arrow.record()
            torch.cuda.synchronize()
            tot_hw_fa = (start_hw_full_arrow.elapsed_time(end_hw_full_arrow)) 
            hw_fa_times.append(tot_hw_fa)
        hw_full_arrow_mean = np.mean(np.array(hw_fa_times))
        print(f"ours full arrow time: {hw_full_arrow_mean}")
        if args.flex_full:
            print(f"ours == flex: {torch.allclose(o_flex_full_arrow.permute(0, 2, 1, 3), o_hw_full_arrow, 1e-3, 1e-3)}")
        

    
        if args.ori:
            from flash_attn_original import flash_attn_func as flash_attn_func_original

            ori_full_times = []

            for _ in range(100):
                flash_attn_func_original(q, k, v, window_size=(-1 , -1))
            torch.cuda.synchronize()

            start_ori = torch.cuda.Event(True)
            end_ori =  torch.cuda.Event(True)

            for _ in range(100): 
                start_ori.record()
                flash_attn_func_original(q, k, v, window_size=(-1 , -1))
                end_ori.record()
                torch.cuda.synchronize()
                tot_original = (start_ori.elapsed_time(end_ori)) 
                ori_full_times.append(tot_original)
            ori_full_mean = np.mean(np.array(ori_full_times))
            print(f"original time: {ori_full_mean}")
            print(f"full-arrow/original: {hw_full_arrow_mean/ori_full_mean*100:.3f}%") 
            print(f"speedup ours: {ori_full_mean/hw_full_arrow_mean:.4f}")
            print(f"speedup flex: {ori_full_mean/flex_full_arrow_mean:.4f}")

    
            # print(o_hw_full_arrow - o_flex_full_arrow.permute(0, 2, 1, 3))

        

        # window_masks, _ = create_full_arrow_mask(H, seqlen_vision, seqlen_text, window_sizes, (128, 128))
        # flex_masks = torch.zeros(window_masks.shape[0], 
        #                         (N + block_size[0] - 1) // block_size[0] * block_size[0], 
        #                         (N + block_size[1] - 1) // block_size[1] * block_size[1],
        #                         dtype=window_masks.dtype,
        #                         device=window_masks.device, requires_grad=False)
        # flex_masks[:, :N, :N] = window_masks

        # qp, kp, vp = q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)

        # for _ in range(100):
        #     flex_headwin_attn(qp,
        #                             kp,
        #                             vp, 
        #                             seqlen_vision, seqlen_text, flex_masks, (128, 128))
        #     torch.cuda.synchronize()
        # print("flex warmup done!")
        # start_flex_fa = torch.cuda.Event(True)
        # end_flex_fa = torch.cuda.Event(True)
        # for _ in range(100): 
        #     start_flex_fa.record()
        #     o_flex_full_arrow = flex_headwin_attn(qp,
        #                             kp,
        #                             vp,
        #                             seqlen_vision, seqlen_text, flex_masks, (128, 128))
            
        #     end_flex_fa.record()
        #     torch.cuda.synchronize()
        #     tot_flex_full_arrow = (start_flex_fa.elapsed_time(end_flex_fa)) 
        #     flex_full_arrow_times.append(tot_flex_full_arrow)
        # flex_full_arrow_mean = np.mean(np.array(flex_full_arrow_times))
        # print(f"flex full arrow time: {flex_full_arrow_mean}")
        
    # if args.half_arrow:
    #     from flash_attn_ours import  headwise_half_arrow_attn
    #     hw_half_arrow_times, hw_half_arrow_mean = [], []
    #     flex_half_arrow_times = []
    #     # output_hw_half_arrows_rights = []
    #     # o_ori_vs_o_ha = []

    #     for _ in range(100):
    #         headwise_half_arrow_attn(q, k, v, window_sizes=window_sizes, seqlen_q_vision = seqlen_vision, seqlen_k_vision = seqlen_vision)
    #     torch.cuda.synchronize()

    #     start_hw_ha = torch.cuda.Event(True)
    #     end_hw_ha =  torch.cuda.Event(True)


    #     for _ in range(100): 
    #         start_hw_ha.record()
    #         headwise_half_arrow_attn(q, k, v, window_sizes=window_sizes, seqlen_q_vision = seqlen_vision, seqlen_k_vision = seqlen_vision)
    #         end_hw_ha.record()
    #         torch.cuda.synchronize()
    #         tot_time_ha = (start_hw_ha.elapsed_time(end_hw_ha)) 
    #         hw_half_arrow_times.append(tot_time_ha)
    #     hw_half_arrow_mean = np.mean(np.array(hw_half_arrow_times))
    #     print(f"ours half arrow time: {hw_half_arrow_mean}")

    # if args.flex_half:
    #     window_masks_ha, _ = create_half_arrow_mask(H, seqlen_vision, seqlen_text, window_sizes, (128, 128))
    #     flex_masks_ha = torch.zeros(window_masks.shape[0], 
    #                             (N + block_size[0] - 1) // block_size[0] * block_size[0], 
    #                             (N + block_size[1] - 1) // block_size[1] * block_size[1],
    #                             dtype=window_masks_ha.dtype,
    #                             device=window_masks_ha.device)
    #     flex_masks_ha[:, :N, :N] = window_masks_ha

    #     qp, kp, vp = q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)

    #     for _ in range(100):
    #         flex_headwin_attn(qp,
    #                                 kp,
    #                                 vp, 
    #                                 seqlen_vision, seqlen_text, flex_masks_ha, (128, 128))
    #         torch.cuda.synchronize()


    #     start_flex_ha = torch.cuda.Event(True)
    #     end_flex_ha = torch.cuda.Event(True)
    #     for _ in range(100): 
    #         start_flex_ha.record()
    #         o_flex_half_arrow = flex_headwin_attn(qp,
    #                                 kp,
    #                                 vp,
    #                                 seqlen_vision, seqlen_text, flex_masks_ha, (128, 128))

    #         end_flex_ha.record()
    #         torch.cuda.synchronize()
    #         tot_flex_half_arrow = (start_flex_ha.elapsed_time(end_flex_ha)) 
    #         flex_half_arrow_times.append(tot_flex_half_arrow)
    #     flex_half_arrow_mean = np.mean(np.array(flex_half_arrow_times))
    #     print(f"flex half arrow time: {flex_half_arrow_mean}")

if __name__ == "__main__":
    # for m in range(2, 4):
    #     for k in range(1, 4):
    #         args = ['-m', f'{m}', '-k', f'{k}']
    #         print(args)
    #         main(args)
    main(sys.argv[1:])