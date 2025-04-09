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



def sdpa_attn(q, k, v, mask=None):
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)


# from flash_attn_ours import flash_attn_func as flash_attn_func

# from flash_attn_original import flash_attn_func as flash_attn_func_ori

# from flash_attn import flash_attn_func_headwin

def transform_dense_block_mask(mask: torch.Tensor, block_size=(64, 64)):
    mask = mask.clone()
    nhead, H, W = mask.shape
    mask = mask.view(nhead, H // block_size[0], block_size[0], W // block_size[1], block_size[1])
    mask_max = mask.amax(dim=[2, 4], keepdim=True)
    # print(f"mask_max.shape: {mask_max.shape}")
    mask |= mask_max
    mask = mask.reshape(nhead, H, W)
    return mask

def create_mask_new(n_heads: int, n_im_tokens: int, n_text_tokens: int, windows_size, block_size: tuple[int, int]=None, dtype=torch.int32, device=torch.device("cuda")):
    tot_tokens = n_im_tokens + n_text_tokens
    
    bmask_sizes_h, bmask_sizes_w = (tot_tokens + block_size[0] - 1) // block_size[0], (tot_tokens + block_size[1] - 1) // block_size[1]
    
    if n_text_tokens == 0:
        full_mask = torch.zeros((n_heads, tot_tokens, tot_tokens), device=device, dtype=torch.bool)
        block_mask = torch.zeros((n_heads, bmask_sizes_h, bmask_sizes_w), device=device, dtype=torch.bool)

        img_mask_row_num = (n_im_tokens + block_size[0] - 1) // block_size[0]
        img_mask_col_num = (n_im_tokens + block_size[1] - 1) // block_size[1]
    else:
        full_mask = torch.ones((n_heads, tot_tokens, tot_tokens), device=device, dtype=torch.bool)
        block_mask = torch.ones((n_heads, bmask_sizes_h, bmask_sizes_w), device=device, dtype=torch.bool)
        img_mask_row_num = n_im_tokens // block_size[0]
        img_mask_col_num = n_im_tokens // block_size[1]
        img_tokens_mutiplier = n_im_tokens // block_size[1]

    for i in range(n_heads):
        if (windows_size[i, 0] == -1 and windows_size[i, 1] == -1):
            img_mask = torch.ones((img_mask_row_num, img_mask_col_num), device=device, dtype=torch.bool)
        else:
            img_mask = torch.zeros((img_mask_row_num, img_mask_col_num), device=device, dtype=torch.bool)

        for j in range(img_mask_row_num):
            start = int(max(0, (j * block_size[0] - windows_size[i, 0]) / block_size[1]))
            end = int(min(img_mask_col_num, ((j+1) * block_size[0] + windows_size[i, 1] + block_size[1] - 1) / block_size[1]))
            # if i==(n_heads - 1):
            #     print(f"row-tile {j} start: {start} end: {end}")
            img_mask[j, start:end] = True
        block_mask[i, :img_mask_row_num, : img_mask_col_num] = img_mask

        # if i == 0:
        #     torch.set_printoptions(threshold=float("inf"))
            # print(block_mask[i])
    full_mask = block_mask.repeat_interleave(block_size[0], dim=1).repeat_interleave(block_size[1], dim=2)[:, :tot_tokens, :tot_tokens]
    # print(full_mask[0])

    return full_mask, block_mask


def repeat_kv(kv: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    [1,2,3,4] => [1,1,2,2,3,3,4,4]
    """
    bs, nhkv, slen, hdim = kv.shape
    if n_rep == 1:
        return kv
    kv = kv[:, :, None, :, :].expand(bs, nhkv, n_rep, slen, hdim)
    return kv.reshape(bs, nhkv * n_rep, slen, hdim)

def draw_mask(window_mask, i, save_dir):
    window_img = window_mask
    plt.imshow(window_img, cmap='gray', vmin=0, vmax=1)
    plt.axis('off')  # 不显示坐标轴
    plt.savefig(os.path.join(save_dir, f'window_img_{i}.png'), bbox_inches='tight', pad_inches=0)  # 保存图像
    # plt.show() 

flex_attention = torch.compile(flex_attention)

FLEX_MASK_CACHE = {}
def flex_headwin_attn(q: Tensor, k: Tensor, v: Tensor, n_im_tokens: int, n_text_tokens: int, spatial_mask, block_size=(128, 128)):

    cache_key = (f"{n_im_tokens}_{n_text_tokens}_{block_size}", spatial_mask)

    if cache_key in FLEX_MASK_CACHE:
        block_mask = FLEX_MASK_CACHE[cache_key]
    else:
        def mask_function(b, h, q_idx, kv_idx):
            cond1 = spatial_mask[h, q_idx, kv_idx]
            return cond1

        total_tokens = n_im_tokens + n_text_tokens


        block_mask = create_block_mask(
            mask_function,
            B=None,
            H=q.shape[1],
            Q_LEN=total_tokens,
            KV_LEN=total_tokens,
            BLOCK_SIZE=block_size,
        )
        FLEX_MASK_CACHE[cache_key] = block_mask
    output = flex_attention(q, k, v, block_mask=block_mask)
    return output


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
    parser.add_argument("-seqt", "--seqlen-text", type=int, default=0)
    parser.add_argument("-nh", "--num-head", type=int, default=8)
    parser.add_argument("-d", "--head-dim", type=int, default=64)
    parser.add_argument("-ws1", "--window-size1", type=int, default=128)
    parser.add_argument("-ws2", "--window-size2", type=int, default=64)
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

    window_sizes_np = window_sizes.cpu().numpy()
    print(window_sizes)
    
    if args.using_bench:
        from flash_attn import flash_attn_func as flash_attn_func_ori
    else:
        if args.headwin:
            from flash_attn_ours import flash_attn_func_headwin  as flash_attn_func_ori
        else:
            from flash_attn_ours import flash_attn_func  as flash_attn_func_ori
        
    
    o_ours = torch.zeros_like(q, dtype=torch.float16, device=device)

    if args.using_bench:
        for i in range(H):
            o_ori_tmp = flash_attn_func_ori(q, k, v, window_size=tuple(window_sizes[i].tolist()))
            o_ours[:, :, i, :] = o_ori_tmp[:, :, i, :]
    else:
        if args.headwin:
            o_ours = flash_attn_func_ori(q, k, v, window_sizes=window_sizes, seqlen_q_vision = args.seqlen_vision, seqlen_k_vision = args.seqlen_vision)
        else:
            o_ours = flash_attn_func_ori(q, k, v, window_sizes=window_sizes)
    torch.cuda.synchronize()
    
    window_masks, block_mask = create_mask_new(H, args.seqlen_vision, args.seqlen_text, window_sizes, (128, 128))
    
    # window_masks_cpu, block_mask_cpu = create_mask_new(H, args.seqlen_vision, args.seqlen_text, window_sizes, (128, 128), device=torch.device("cpu"))
    save_dir = "/root/surd/flash_attn_mod/flash-attention-2.7.2/mytest/mask_img"
    # draw_mask_fast(H, args.seqlen_vision, args.seqlen_text, window_sizes_np, (128, 128), save_dir=save_dir)
    # for i in range(H):
    #     print(f"saving mask image {i}... ")
    #     draw_mask(window_masks_cpu[i].int(), i, save_dir)

    wh, wtr, wtc = window_masks.shape

    flex_masks = torch.zeros((wh, (wtr + block_sizes[0] - 1) // block_sizes[0] * block_sizes[0], (wtc + block_sizes[1] - 1) // block_sizes[1] * block_sizes[1]), device=window_masks.device, dtype=window_masks.dtype)
    flex_masks[:, :wtr, :wtc] = window_masks
    
    o_flex = flex_headwin_attn(q.permute(0, 2, 1, 3).contiguous(), 
                               k.permute(0, 2, 1, 3).contiguous(), 
                               v.permute(0, 2, 1, 3).contiguous(), 
                               args.seqlen_vision, args.seqlen_text, flex_masks, (128, 128)).permute(0, 2, 1, 3).contiguous()
    
    o_sdpa = sdpa_attn(q.permute(0, 2, 1, 3).contiguous(),
                            k.permute(0, 2, 1, 3).contiguous(), 
                            v.permute(0, 2, 1, 3).contiguous(), 
                            mask=window_masks
                            ).permute(0, 2, 1, 3)

    torch.cuda.synchronize()
    print(f"o_flex == o_ours: \033[35m{torch.allclose(o_flex, o_ours, 1e-3, 1e-3)}\033[0m")
    print(f"o_sdpa == o_ours: \033[35m{torch.allclose(o_sdpa, o_ours, 1e-3, 1e-3)}\033[0m")
    print(f"o_sdpa == o_flex: \033[35m{torch.allclose(o_sdpa, o_flex, 1e-3, 1e-3)}\033[0m")
    
    print(f"{'='*64}\n{'='*64}")


    # print(f"o_flex:\n{o_flex[0, :, 0, :]}")
    # print(f"o_ours:\n{o_ours[0, :, 0, :]}")
    # torch.set_printoptions(threshold=float("inf"))
    # print(f"o_flex - o_ours:\n{o_flex[0, :, 0, :] - o_ours[0, :, 0, :]}")



    # for i in range(H):
    #     # print(f"max error in head \033[35m{i}\033[0m between o_flex and o_ours: {(o_sdpa[:, :, i, :] - o_ours[:, :, i, :]).max()}")
    #     print(f"max error in head \033[35m{i}\033[0m between o_flex and o_ours: {(o_flex[:, :, i, :] - o_ours[:, :, i, :]).max()}")
    # print(f"max error between o_sdpa and o_ours: {(o_sdpa - o_ours).max()}")
    # print(f"max error between o_sdpa and o_flex: {(o_sdpa - o_flex).max()}")

if __name__=="__main__":
    main(sys.argv[1:])