
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

def check_window_size(window_sizes, seqlen_k, seqlen_q_vision, seqlen_k_vision):
    window_sizes[:, 0][window_sizes[:, 0] > seqlen_k] = -1
    window_sizes[:, 1][window_sizes[:, 1] > seqlen_k] = -1

    condition_left = (window_sizes[:, 0] < 0) & (window_sizes[:, 1] >= 0)
    condition_right = (window_sizes[:, 0] >= 0) & (window_sizes[:, 1] < 0)

    # 修改张量
    window_sizes[:, 0][condition_left] = seqlen_k - 0
    window_sizes[:, 1][condition_right] = seqlen_k - 0

    if (seqlen_k_vision <= 0 or seqlen_q_vision <=0):
        window_sizes[:, 0][:] = -1
        window_sizes[:, 1][:] = -1
    
    return window_sizes

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

    flash_attn_func_ours_times, ori_times = [], []
    flash_attn_func_ours_mean, ori_mean = [], []
    flash_attn_func_ours_rights = []

    
    ori_outpput = []

    win_size_half = [256, 128, 64, -1]
    block_size = (128, 128)

    from flash_attn_original import flash_attn_func as flash_attn_func_original
    for i in range(len(win_size_half)):
        torch.cuda.empty_cache()
        for _ in range(100):
            flash_attn_func_original(q, k, v, window_size=(win_size_half[i] , win_size_half[i]))
        torch.cuda.synchronize()

        start_ori = torch.cuda.Event(True)
        end_ori =  torch.cuda.Event(True)
        
        
        for _ in range(100): 
            start_ori.record()
            o_ori = flash_attn_func_original(q, k, v, window_size=(win_size_half[i] , win_size_half[i]))
            end_ori.record()
            torch.cuda.synchronize()
            tot_original = (start_ori.elapsed_time(end_ori)) 
            ori_times.append(tot_original)
        ori_mean.append(np.mean(np.array(ori_times)))
        ori_outpput.append(o_ori)
    # print(end_ori - start_ori)

    # print(f"output original length: {len(output_originals)}")

    from flash_attn_ours import flash_attn_func, headwise_arrow_attn, headwise_window_attn, headwise_half_arrow_attn
    for i in range(len(win_size_half)):
        torch.cuda.empty_cache()
        for _ in range(100):
            flash_attn_func(q, k, v, window_size=(win_size_half[i] , win_size_half[i]))
        torch.cuda.synchronize()

        start_ours_flash_attn_func = torch.cuda.Event(True)
        end_ours_flash_attn_func =  torch.cuda.Event(True)
        
        
        for _ in range(100): 
            start_ours_flash_attn_func.record()
            flash_attn_func_ours = flash_attn_func(q, k, v, window_size=(win_size_half[i] , win_size_half[i]))
            end_ours_flash_attn_func.record()
            torch.cuda.synchronize()
            tot_ours = (start_ours_flash_attn_func.elapsed_time(end_ours_flash_attn_func)) 
            flash_attn_func_ours_times.append(tot_original)
        flash_attn_func_ours_mean.append(np.mean(np.array(tot_ours)))

        flash_attn_func_ours_rights.append(torch.allclose(ori_outpput[i], flash_attn_func_ours, 1e-3, 1e-3))
    
    print(f"shape: batch: {B}, seqlen: {N}, num_head: {H}, head_dim: {D}")
    for i in range(len(win_size_half)):
        print(f"{'-'*64}\n{f'window-size: \033[96m({win_size_half[i]}, {win_size_half[i]})\033[0m local attn':^64}\n{'-'*64}")
        print(f"ours == original: \033[92m{flash_attn_func_ours_rights[i]}\033[0m")
        print("\t{:<24}\033[35m{:>12.6f}\033[0m".format("ous time:", flash_attn_func_ours_mean[i]))
        print("\t{:<24}\033[35m{:>12.6f}\033[0m".format("original time:", ori_mean[i]))


    # for i in range(len(win_size_half)):
    #     window_sizes = torch.zeros((H, 2), device=device, dtype=torch.int32) - 1
    #     window_sizes[:, :] = win_size_half[i]   
    #     window_sizes[:args.full_num, :] = -1

    #     window_sizes = check_window_size(window_sizes, seqlen_k=k.shape[1], seqlen_q_vision=args.seqlen_vision, seqlen_k_vision=args.seqlen_vision)
    #     # print(window_sizes)

    #     # from flash_attn_ours import flash_attn_func as flash_attn_func_ours
    #     for _ in range(100):
            
    #         headwise_window_attn(q, k, v, window_sizes=window_sizes)
    #         # else:
    #         #     flash_attn_func_ours(q, k, v, window_sizes=window_sizes, seqlen_q_vision = args.seqlen_vision, seqlen_k_vision = args.seqlen_vision)
    #     torch.cuda.synchronize()

    #     start_ours = torch.cuda.Event(True)
    #     end_ours = torch.cuda.Event(True)
    #     if args.local:
    #         for _ in range(100): 
    #             start_ours.record()
    #             o_ours = headwise_window_attn(q, k, v, window_sizes=window_sizes)
    #             end_ours.record()
    #             torch.cuda.synchronize()
    #             tot_ours = (start_ours.elapsed_time(end_ours)) 
    #             ours_times.append(tot_ours)
    #     else:
    #         for _ in range(100): 
    #             start_ours.record()
    #             o_ours = flash_attn_func_ours(q, k, v, window_sizes=window_sizes, seqlen_q_vision = args.seqlen_vision, seqlen_k_vision = args.seqlen_vision)
    #             end_ours.record()
    #             torch.cuda.synchronize()
    #             tot_ours = (start_ours.elapsed_time(end_ours)) 
    #             ours_times.append(tot_ours)
            
    #     our_mean.append(np.mean(np.array(ours_times)))
    #     # output_ours.append(o_ours)

    #     window_masks, block_mask = create_mask_new(H, args.seqlen_vision, args.seqlen_text, window_sizes, (128, 128))
    #     flex_masks = torch.zeros(window_masks.shape[0], 
    #                             (N + block_size[0] - 1) // block_size[0] * block_size[0], 
    #                             (N + block_size[1] - 1) // block_size[1] * block_size[1],
    #                             dtype=window_masks.dtype,
    #                             device=window_masks.device)
    #     flex_masks[:, :N, :N] = window_masks

    #     qp, kp, vp = q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)
        
    #     for _ in range(100):
    #         o_flex = flex_headwin_attn(qp,
    #                                 kp,
    #                                 vp, 
    #                                 args.seqlen_vision, args.seqlen_text, flex_masks, (128, 128))
    #         torch.cuda.synchronize()
    #         # o_flex = flex_headwin_attn(q.permute(0, 2, 1, 3).contiguous(), 
    #         #                         k.permute(0, 2, 1, 3).contiguous(), 
    #         #                         v.permute(0, 2, 1, 3).contiguous(), 
    #         #                         args.seqlen_vision, args.seqlen_text, flex_masks, (128, 128)).permute(0, 2, 1, 3).contiguous()
    #     # print("warm up done!")
    #     start_flex = torch.cuda.Event(True)
    #     end_flex = torch.cuda.Event(True)
    #     for _ in range(100): 
    #         start_flex.record()
    #         o_flex = flex_headwin_attn(qp,
    #                                 kp,
    #                                 vp,
    #                                 args.seqlen_vision, args.seqlen_text, flex_masks, (128, 128))
    #         # o_flex = flex_headwin_attn(q.permute(0, 2, 1, 3).contiguous(), 
    #         #                     k.permute(0, 2, 1, 3).contiguous(), 
    #         #                     v.permute(0, 2, 1, 3).contiguous(), 
    #         #                     args.seqlen_vision, args.seqlen_text, flex_masks, (128, 128)).permute(0, 2, 1, 3).contiguous()
    #         end_flex.record()
    #         torch.cuda.synchronize()
    #         tot_flex = (start_flex.elapsed_time(end_flex)) 
    #         flex_times.append(tot_flex)
    #     flex_mean.append(np.mean(np.array(flex_times)))
    #     # output_flex.append(o_flex)

    #     rights.append(torch.allclose(o_ours, o_flex.permute(0, 2, 1, 3), 1e-3, 1e-3))
        

    # print(f"shape: batch: {B}, seqlen: {N}, num_head: {H}, head_dim: {D}")
    # for i in range(len(win_size_half)):
    #     print(f"ours == flex: \033[92m{rights[i]}\033[0m")
    #     print(f"{args.full_num}/{args.num_head} heads do full {args.num_head - args.full_num} heads do {win_size_half[i]*2} local attn")
    #     print("\t{:<24}\033[35m{:>12.6f}\033[0m".format("ous time:", our_mean[i]))
    #     print("\t{:<24}\033[35m{:>12.6f}\033[0m".format("flex time:", flex_mean[i]))
    #     print("\t{:<24}\033[35m{:>12.6f}\033[0m".format("full attn original time:", ori_mean[i]))



if __name__ == "__main__":
    main(sys.argv[1:])