"""
in this file we test why when fn = 0, and fn = 8, window_sizes is all -1, affect the function performace

namely :
        window_sizes = torch.zeros((H, 2), device=device, dtype=torch.int32) - 1
        window_sizes[:, :] = -1
        window_sizes[:args.full_num, :] = torch.tensor(-1, device=device, dtype=torch.int32)

        why this affects the perf
"""


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

    from flash_attn_ours import  headwise_arrow_attn, headwise_half_arrow_attn

    hw_fa_times, hw_fa_times2 = [], []

    window_sizes = torch.zeros((H, 2), device=device, dtype=torch.int32) - 1
    window_sizes[:, :] = -1
  
    torch.cuda.empty_cache()
    for _ in range(1000):
        headwise_arrow_attn(q, k, v, window_sizes=window_sizes, seqlen_q_vision = args.seqlen_vision, seqlen_k_vision = args.seqlen_vision)
    torch.cuda.synchronize()

    start_hw_full_arrow = torch.cuda.Event(True)
    end_hw_full_arrow =  torch.cuda.Event(True)
        
        
    for _ in range(100): 
        start_hw_full_arrow.record()
        o_hw_full_arrow_no_fn = headwise_arrow_attn(q, k, v, window_sizes=window_sizes, seqlen_q_vision = args.seqlen_vision, seqlen_k_vision = args.seqlen_vision)
        end_hw_full_arrow.record()
        torch.cuda.synchronize()
        tot_hw_fa = (start_hw_full_arrow.elapsed_time(end_hw_full_arrow)) 
        hw_fa_times.append(tot_hw_fa)
    hw_fa_time_no_fn = np.mean(np.array(hw_fa_times))

    window_sizes = torch.zeros((H, 2), device=device, dtype=torch.int32) - 1
    window_sizes[:, :] = -1
    window_sizes[:args.full_num, :] = -1
  
    torch.cuda.empty_cache()
    for _ in range(1000):
        headwise_arrow_attn(q, k, v, window_sizes=window_sizes, seqlen_q_vision = args.seqlen_vision, seqlen_k_vision = args.seqlen_vision)
    torch.cuda.synchronize()

    start_hw_full_arrow2 = torch.cuda.Event(True)
    end_hw_full_arrow2 =  torch.cuda.Event(True)
        
        
    for _ in range(100): 
        start_hw_full_arrow2.record()
        o_hw_full_arrow_with_fn = headwise_arrow_attn(q, k, v, window_sizes=window_sizes, seqlen_q_vision = args.seqlen_vision, seqlen_k_vision = args.seqlen_vision)
        end_hw_full_arrow2.record()
        torch.cuda.synchronize()
        tot_hw_fa2 = (start_hw_full_arrow2.elapsed_time(end_hw_full_arrow2)) 
        hw_fa_times2.append(tot_hw_fa2)
    hw_fa_time_with_fn = np.mean(np.array(hw_fa_times2))

#################################################################


    hw_half_arrow_times, hw_half_arrow_times2 = [], []

    window_sizes = torch.zeros((H, 2), device=device, dtype=torch.int32) - 1
    window_sizes[:, :] = -1

    for _ in range(100):
        headwise_half_arrow_attn(q, k, v, window_sizes=window_sizes, seqlen_q_vision = args.seqlen_vision, seqlen_k_vision = args.seqlen_vision)
    torch.cuda.synchronize()

    start_hw_ha = torch.cuda.Event(True)
    end_hw_ha =  torch.cuda.Event(True)
    
    
    for _ in range(100): 
        start_hw_ha.record()
        o_hw_half_arrow_no_fn = headwise_half_arrow_attn(q, k, v, window_sizes=window_sizes, seqlen_q_vision = args.seqlen_vision, seqlen_k_vision = args.seqlen_vision)
        end_hw_ha.record()
        torch.cuda.synchronize()
        tot_time_ha = (start_hw_ha.elapsed_time(end_hw_ha)) 
        hw_half_arrow_times.append(tot_time_ha)
    hw_ha_time_no_fn = np.mean(np.array(hw_half_arrow_times))

#-----------------------------------------------------------------#
    window_sizes = torch.zeros((H, 2), device=device, dtype=torch.int32) - 1
    window_sizes[:, :] = -1
    window_sizes[:args.full_num, :] = -1

    for _ in range(100):
        headwise_half_arrow_attn(q, k, v, window_sizes=window_sizes, seqlen_q_vision = args.seqlen_vision, seqlen_k_vision = args.seqlen_vision)
    torch.cuda.synchronize()

    start_hw_ha2 = torch.cuda.Event(True)
    end_hw_ha2 =  torch.cuda.Event(True)
    
    
    for _ in range(100): 
        start_hw_ha2.record()
        o_hw_half_arrow_with_fn = headwise_half_arrow_attn(q, k, v, window_sizes=window_sizes, seqlen_q_vision = args.seqlen_vision, seqlen_k_vision = args.seqlen_vision)
        end_hw_ha2.record()
        torch.cuda.synchronize()
        tot_time_ha2 = (start_hw_ha2.elapsed_time(end_hw_ha2)) 
        hw_half_arrow_times2.append(tot_time_ha2)
    hw_ha_time_with_fn = np.mean(np.array(hw_half_arrow_times2))


    print(f"o_hw_full_arrow_no_fn == o_hw_full_arrow_with_fn: \033[92m{torch.allclose(o_hw_full_arrow_no_fn, o_hw_full_arrow_with_fn)}\033[0m")
    print("\t{:<24}\033[35m{:>12.6f}\033[0m".format("full arrow time no fn:", hw_fa_time_no_fn), end="")
    print("\t{:<24}\033[35m{:>12.6f}\033[0m".format("full arrow time with fn:", hw_fa_time_with_fn))

    print(f"o_hw_half_arrow_no_fn == o_hw_half_arrow_with_fn: \033[92m{torch.allclose(o_hw_half_arrow_no_fn, o_hw_half_arrow_with_fn)}\033[0m")
    print("\t{:<24}\033[35m{:>12.6f}\033[0m".format("half arrow time no fn:", hw_ha_time_no_fn), end="")
    print("\t{:<24}\033[35m{:>12.6f}\033[0m".format("half arrow time with fn:", hw_ha_time_with_fn))


if __name__ == "__main__":
    main(sys.argv[1:])