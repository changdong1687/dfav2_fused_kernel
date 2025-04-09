
import torch
import time
import sys
import argparse

import numpy as np

import torch

from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from torch import Tensor
import time
import sys
import argparse

import torch.nn.functional as F
import numpy as np

from mask import create_full_arrow_mask, create_half_arrow_mask, create_headwise_window_mask

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
    parser.add_argument("-b", "--batch", type=int, default=8)
    parser.add_argument("-seq", "--seqlen", type=int, default=4096*2)
    parser.add_argument("-seqv", "--seqlen-vision", type=int, default=4096*2)
    parser.add_argument("-seqt", "--seqlen-text", type=int, default=0)
    parser.add_argument("-nh", "--num-head", type=int, default=8)
    parser.add_argument("-d", "--head-dim", type=int, default=64)
    parser.add_argument("--min-val", type=int, default=128)
    parser.add_argument("--max-val", type=int, default=256)
    parser.add_argument("-fn", "--full-num", type=int, default=4, help="how many heads do full attention")
    parser.add_argument("--local", action="store_true")

    args = parser.parse_args(argv)

    # B, N, H, D = args.batch, args.seqlen, args.num_head, args.head_dim                    # 2, 1024, 32, 128
    B, N, H, D = args.batch, args.seqlen_vision + args.seqlen_text, args.num_head, args.head_dim                    # 2, 1024, 32, 128
 
    device = torch.device("cuda")


    q, k, v = torch.randn(B * 3, N, H, D, dtype=torch.float16, device=device).split(B, dim=0)

    flash_attn_func_ours_times, ori_full_times = [], []
    flash_attn_func_ours_mean, ori_full_mean = [], []


    win_size_half = [256, 128, 64,]
    block_size = (128, 128)

    from flash_attn_original import flash_attn_func as flash_attn_func_original
    
    for _ in range(100):
        flash_attn_func_original(q, k, v, window_size=(-1 , -1))
    torch.cuda.synchronize()

    start_ori = torch.cuda.Event(True)
    end_ori =  torch.cuda.Event(True)
    
    for _ in range(100): 
        start_ori.record()
        o_ori = flash_attn_func_original(q, k, v, window_size=(-1 , -1))
        end_ori.record()
        torch.cuda.synchronize()
        tot_original = (start_ori.elapsed_time(end_ori)) 
        ori_full_times.append(tot_original)
    ori_full_mean = np.mean(np.array(ori_full_times))
    # print(end_ori - start_ori)

    # print(f"output original length: {len(output_originals)}")


    from flash_attn_ours import  headwise_arrow_attn, headwise_half_arrow_attn, headwise_window_attn

    ######################################################
#       test headwise full arrow attention time
######################################################
    hw_win_times, hw_win_mean = [], []
    o_ori_vs_o_fa = []

    for i in range(len(win_size_half)):
        torch.cuda.empty_cache()
        window_sizes = torch.zeros((H, 2), device=device, dtype=torch.int32) - 1
        window_sizes[:, :] = win_size_half[i]   
        window_sizes[:args.full_num, :] = -1
        
        for _ in range(100):
            headwise_window_attn(q, k, v, window_sizes=window_sizes)
        torch.cuda.synchronize()

        start_hw_full_arrow = torch.cuda.Event(True)
        end_hw_full_arrow =  torch.cuda.Event(True)
        
        
        for _ in range(100): 
            start_hw_full_arrow.record()
            o_hw_win_arrow = headwise_window_attn(q, k, v, window_sizes=window_sizes)
            end_hw_full_arrow.record()
            torch.cuda.synchronize()
            tot_hw_win = (start_hw_full_arrow.elapsed_time(end_hw_full_arrow)) 
            hw_win_times.append(tot_hw_win)
        hw_win_mean.append(np.mean(np.array(hw_win_times)))
######################################################
#       test headwise full arrow attention time
######################################################
    hw_fa_times, hw_full_arrow_mean = [], []
    flex_full_arrow_times, flex_full_arrow_mean = [], []
    output_hw_full_arrows_rights = []
    o_ori_vs_o_fa = []

    for i in range(len(win_size_half)):
        torch.cuda.empty_cache()
        window_sizes = torch.zeros((H, 2), device=device, dtype=torch.int32) - 1
        window_sizes[:, :] = win_size_half[i]   
        window_sizes[:args.full_num, :] = -1
        
        for _ in range(100):
            headwise_arrow_attn(q, k, v, window_sizes=window_sizes, seqlen_q_vision = args.seqlen_vision, seqlen_k_vision = args.seqlen_vision)
        torch.cuda.synchronize()

        start_hw_full_arrow = torch.cuda.Event(True)
        end_hw_full_arrow =  torch.cuda.Event(True)
        
        
        for _ in range(100): 
            start_hw_full_arrow.record()
            o_hw_full_arrow = headwise_arrow_attn(q, k, v, window_sizes=window_sizes, seqlen_q_vision = args.seqlen_vision, seqlen_k_vision = args.seqlen_vision)
            end_hw_full_arrow.record()
            torch.cuda.synchronize()
            tot_hw_fa = (start_hw_full_arrow.elapsed_time(end_hw_full_arrow)) 
            hw_fa_times.append(tot_hw_fa)
        hw_full_arrow_mean.append(np.mean(np.array(hw_fa_times)))
    
        window_masks, _ = create_full_arrow_mask(H, args.seqlen_vision, args.seqlen_text, window_sizes, (128, 128))
        flex_masks = torch.zeros(window_masks.shape[0], 
                                (N + block_size[0] - 1) // block_size[0] * block_size[0], 
                                (N + block_size[1] - 1) // block_size[1] * block_size[1],
                                dtype=window_masks.dtype,
                                device=window_masks.device)
        flex_masks[:, :N, :N] = window_masks

        qp, kp, vp = q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)
        
        for _ in range(100):
            flex_headwin_attn(qp,
                                    kp,
                                    vp, 
                                    args.seqlen_vision, args.seqlen_text, flex_masks, (128, 128))
            torch.cuda.synchronize()
            # o_flex = flex_headwin_attn(q.permute(0, 2, 1, 3).contiguous(), 
            #                         k.permute(0, 2, 1, 3).contiguous(), 
            #                         v.permute(0, 2, 1, 3).contiguous(), 
            #                         args.seqlen_vision, args.seqlen_text, flex_masks, (128, 128)).permute(0, 2, 1, 3).contiguous()
        # print("warm up done!")
        start_flex_fa = torch.cuda.Event(True)
        end_flex_fa = torch.cuda.Event(True)
        for _ in range(100): 
            start_flex_fa.record()
            o_flex_full_arrow = flex_headwin_attn(qp,
                                    kp,
                                    vp,
                                    args.seqlen_vision, args.seqlen_text, flex_masks, (128, 128))
            # o_flex = flex_headwin_attn(q.permute(0, 2, 1, 3).contiguous(), 
            #                     k.permute(0, 2, 1, 3).contiguous(), 
            #                     v.permute(0, 2, 1, 3).contiguous(), 
            #                     args.seqlen_vision, args.seqlen_text, flex_masks, (128, 128)).permute(0, 2, 1, 3).contiguous()
            end_flex_fa.record()
            torch.cuda.synchronize()
            tot_flex_full_arrow = (start_flex_fa.elapsed_time(end_flex_fa)) 
            flex_full_arrow_times.append(tot_flex_full_arrow)
        flex_full_arrow_mean.append(np.mean(np.array(flex_full_arrow_times)))
        output_hw_full_arrows_rights.append(torch.allclose(o_hw_full_arrow, o_flex_full_arrow.permute(0, 2, 1, 3), 1e-3, 1e-3))
        o_ori_vs_o_fa.append(torch.allclose(o_hw_full_arrow, o_ori, 1e-3, 1e-3))
        # output_flex.append(o_flex)
    


######################################################
#       test headwise half arrow attention time
######################################################
    hw_half_arrow_times, hw_half_arrow_mean = [], []
    flex_half_arrow_times, flex_half_arrow_mean = [], []
    output_hw_half_arrows_rights = []
    o_ori_vs_o_ha = []

    for i in range(len(win_size_half)):
        torch.cuda.empty_cache()
        window_sizes = torch.zeros((H, 2), device=device, dtype=torch.int32) - 1
        window_sizes[:, :] = win_size_half[i]   
        window_sizes[:args.full_num, :]  = -1
        # window_sizes[:args.full_num, :].fill_(-1)
        print(window_sizes)
        window_sizes.contiguous()
        
        for _ in range(100):
            headwise_half_arrow_attn(q, k, v, window_sizes=window_sizes, seqlen_q_vision = args.seqlen_vision, seqlen_k_vision = args.seqlen_vision)
        torch.cuda.synchronize()

        start_hw_ha = torch.cuda.Event(True)
        end_hw_ha =  torch.cuda.Event(True)
        
        
        for _ in range(100): 
            start_hw_ha.record()
            o_hw_half_arrow = headwise_half_arrow_attn(q, k, v, window_sizes=window_sizes, seqlen_q_vision = args.seqlen_vision, seqlen_k_vision = args.seqlen_vision)
            end_hw_ha.record()
            torch.cuda.synchronize()
            tot_time_ha = (start_hw_ha.elapsed_time(end_hw_ha)) 
            hw_half_arrow_times.append(tot_time_ha)
        hw_half_arrow_mean.append(np.mean(np.array(hw_half_arrow_times)))
    
        window_masks_ha, _ = create_half_arrow_mask(H, args.seqlen_vision, args.seqlen_text, window_sizes, (128, 128))
        flex_masks_ha = torch.zeros(window_masks.shape[0], 
                                (N + block_size[0] - 1) // block_size[0] * block_size[0], 
                                (N + block_size[1] - 1) // block_size[1] * block_size[1],
                                dtype=window_masks_ha.dtype,
                                device=window_masks_ha.device)
        flex_masks_ha[:, :N, :N] = window_masks_ha

        qp, kp, vp = q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)
        
        for _ in range(100):
            flex_headwin_attn(qp,
                                    kp,
                                    vp, 
                                    args.seqlen_vision, args.seqlen_text, flex_masks_ha, (128, 128))
            torch.cuda.synchronize()
            # o_flex = flex_headwin_attn(q.permute(0, 2, 1, 3).contiguous(), 
            #                         k.permute(0, 2, 1, 3).contiguous(), 
            #                         v.permute(0, 2, 1, 3).contiguous(), 
            #                         args.seqlen_vision, args.seqlen_text, flex_masks, (128, 128)).permute(0, 2, 1, 3).contiguous()
        # print("warm up done!")
        start_flex_ha = torch.cuda.Event(True)
        end_flex_ha = torch.cuda.Event(True)
        for _ in range(100): 
            start_flex_ha.record()
            o_flex_half_arrow = flex_headwin_attn(qp,
                                    kp,
                                    vp,
                                    args.seqlen_vision, args.seqlen_text, flex_masks_ha, (128, 128))
            # o_flex = flex_headwin_attn(q.permute(0, 2, 1, 3).contiguous(), 
            #                     k.permute(0, 2, 1, 3).contiguous(), 
            #                     v.permute(0, 2, 1, 3).contiguous(), 
            #                     args.seqlen_vision, args.seqlen_text, flex_masks, (128, 128)).permute(0, 2, 1, 3).contiguous()
            end_flex_ha.record()
            torch.cuda.synchronize()
            tot_flex_half_arrow = (start_flex_ha.elapsed_time(end_flex_ha)) 
            flex_half_arrow_times.append(tot_flex_half_arrow)
        flex_half_arrow_mean.append(np.mean(np.array(flex_half_arrow_times)))
        output_hw_half_arrows_rights.append(torch.allclose(o_hw_half_arrow, o_flex_half_arrow.permute(0, 2, 1, 3), 1e-3, 1e-3))
        o_ori_vs_o_ha.append(torch.allclose(o_hw_half_arrow, o_ori, 1e-3, 1e-3))
        # output_flex.append(o_flex)

    # print(f"shape: batch: {B}, seqlen: {N}, num_head: {H}, head_dim: {D}")
    # for i in range(len(win_size_half)):
    #     print(f"{'-'*64}\n{f'window-size: \033[96m({win_size_half[i]}, {win_size_half[i]})\033[0m local attn':^64}")
    #     print(f"{f'{args.full_num}/{args.num_head} heads do full {args.num_head - args.full_num} heads do {win_size_half[i]*2} local attn':^64}\n{'-'*64}\n")
        

    print(f"shape: batch: {B}, seqlen: {N}, seqlen-vision: {args.seqlen_vision}, seqlen-text: {args.seqlen_text}, num_head: {H}, head_dim: {D}")
    for i in range(len(win_size_half)):
        print(f"{'-'*64}\n{f'window-size: \033[96m({win_size_half[i]}, {win_size_half[i]})\033[0m local attn':^64}")
        print(f"{f'{args.full_num}/{args.num_head} heads do full {args.num_head - args.full_num} heads do {win_size_half[i]*2} local attn':^64}\n{'-'*64}\n")
        print(f"ours_full_arrow == felx_full_arrow: \033[92m{output_hw_full_arrows_rights[i]}\033[0m")
        print(f"ours_half_arrow == felx_half_arrow: \033[92m{output_hw_half_arrows_rights[i]}\033[0m")

        print(f"ours_full_arrow == o_ori: \033[92m{o_ori_vs_o_fa[i]}\033[0m\tours_half_arrow == o_ori: \033[92m{o_ori_vs_o_ha[i]}\033[0m")
        
        print("\t{:<24}\033[35m{:>12.6f}\033[0m".format("ous full arrow time:", hw_full_arrow_mean[i]), end="")
        print("\t{:<24}\033[35m{:>12.6f}\033[0m".format("flex full arrow time:", flex_full_arrow_mean[i]))


        print("\t{:<24}\033[35m{:>12.6f}\033[0m".format("ous half arrow time:", hw_half_arrow_mean[i]), end="")
        print("\t{:<24}\033[35m{:>12.6f}\033[0m".format("flex half arrow  time:", flex_half_arrow_mean[i]))
        print("\t{:<24}\033[35m{:>12.6f}\033[0m".format("original time:", ori_full_mean))
        print("\t{:<24}\033[35m{:>12.6f}\033[0m".format("original time:", hw_win_mean[i]))

    window_sizes = torch.zeros((H, 2), device=device, dtype=torch.int32) - 1
    o_hw_full_arrow = headwise_arrow_attn(q, k, v, window_sizes=window_sizes, seqlen_q_vision = args.seqlen_vision, seqlen_k_vision = args.seqlen_vision)
    o_hw_half_arrow = headwise_half_arrow_attn(q, k, v, window_sizes=window_sizes, seqlen_q_vision = args.seqlen_vision, seqlen_k_vision = args.seqlen_vision)

    print("window-sizes all -1\n o_hw_full_arrow == o_ori:{!s:6}\no_hw_half_arrow == o_ori:{!s:6}\no_hw_full_arrow == o_hw_half_arrow:{!s:6}".format(
          torch.allclose(o_hw_full_arrow, o_ori, 1e-3, 1e-3),
          torch.allclose(o_hw_half_arrow, o_ori, 1e-3, 1e-3),
          torch.allclose(o_hw_full_arrow, o_hw_half_arrow, 1e-3, 1e-3)))

if __name__ == "__main__":
    main(sys.argv[1:])