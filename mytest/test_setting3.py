"""
batch:   8  seqlen:  4096   num head: 24    head dim: 64

window_left = 256/128/64    window_right = 256/128/64

每个head采用相同的 window_size
测试计算时间
"""
import torch
import time
import sys
import argparse

import numpy as np
# from flash_attn import flash_attn_func as flash_attn_func_ours

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

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch", type=int, default=2)
    parser.add_argument("-seq", "--seqlen", type=int, default=2048)
    parser.add_argument("-seqv", "--seqlen-vision", type=int, default=1024)
    parser.add_argument("-seqt", "--seqlen-text", type=int, default=0)
    parser.add_argument("-nh", "--num-head", type=int, default=8)
    parser.add_argument("-d", "--head-dim", type=int, default=64)
    parser.add_argument("--min-val", type=int, default=128)
    parser.add_argument("--max-val", type=int, default=256)
    parser.add_argument("-fn", "--full-num", type=int, default=4, help="how many heads do full attention")
    parser.add_argument("--headwin", action="store_true")

    args = parser.parse_args(argv)

    # B, N, H, D = args.batch, args.seqlen, args.num_head, args.head_dim                    # 2, 1024, 32, 128
    B, N, H, D = args.batch, args.seqlen_vision + args.seqlen_text, args.num_head, args.head_dim                    # 2, 1024, 32, 128
    device = torch.device("cuda")


    q, k, v = torch.randn(B * 3, N, H, D, dtype=torch.float16, device=device).split(B, dim=0)

    ours_times, ori_times = [], []
    our_mean, ori_mean = [], []
    rights = []
    output_ours, output_originals = [], []

    win_size_half = [256, 128, 64]

    from flash_attn_original import flash_attn_func as flash_attn_func_original
    for i in range(len(win_size_half)):
        window_size = (win_size_half[i], win_size_half[i])
        window_sizes = torch.zeros((H, 2), device=device, dtype=torch.int32) - 1
        window_sizes[:12, :] = win_size_half[i]
        # print(window_sizes)

        torch.cuda.empty_cache()
        for _ in range(100):
            flash_attn_func_original(q, k, v, window_size=(-1, -1))
        torch.cuda.synchronize()

        start_ori = torch.cuda.Event(True)
        end_ori =  torch.cuda.Event(True)
        
        
        for _ in range(100): 
            start_ori.record()
            o_ori = flash_attn_func_original(q, k, v, window_size=(-1, -1))
            end_ori.record()
            torch.cuda.synchronize()
            tot_original = (start_ori.elapsed_time(end_ori)) 
            ori_times.append(tot_original)
        ori_mean.append(np.mean(np.array(ori_times)))
        # output_originals.append(o_ori)
    # print(end_ori - start_ori)

    # print(f"output original length: {len(output_originals)}")

    if args.headwin:
        from flash_attn_ours import flash_attn_func_headwin as flash_attn_func_ours
    else:
        from flash_attn_ours import flash_attn_func as flash_attn_func_ours
    for i in range(len(win_size_half)):
        window_size = (win_size_half[i], win_size_half[i])
        window_sizes = torch.zeros((H, 2), device=device, dtype=torch.int32) - 1
        window_sizes[:, :] = win_size_half[i]   
        window_sizes[:args.full_num, :] = -1

        window_sizes = check_window_size(window_sizes, seqlen_k=k.shape[1], seqlen_q_vision=N, seqlen_k_vision=N)

        # from flash_attn_ours import flash_attn_func as flash_attn_func_ours
        # test ours
        for _ in range(100):
            if args.headwin:
                flash_attn_func_ours(q, k, v, window_sizes=window_sizes, seqlen_q_vision = N, seqlen_k_vision = N)
            else:
                flash_attn_func_ours(q, k, v, window_sizes=window_sizes)
            # flash_attn_func_headwin(q, k, v, window_sizes=window_sizes, seqlen_q_vision = N, seqlen_k_vision = N)
        torch.cuda.synchronize()

        start_ours = torch.cuda.Event(True)
        end_ours = torch.cuda.Event(True)
        if args.headwin:
            for _ in range(100): 
                start_ours.record()
                # print("start record...")
                # o_ours = flash_attn_func_ours(q, k, v, window_sizes=window_sizes)
                o_ours = flash_attn_func_ours(q, k, v, window_sizes=window_sizes, seqlen_q_vision = N, seqlen_k_vision = N)
                end_ours.record()
                torch.cuda.synchronize()
                tot_ours = (start_ours.elapsed_time(end_ours)) 
                ours_times.append(tot_ours)
        else:
            for _ in range(100): 
                start_ours.record()
                o_ours = flash_attn_func_ours(q, k, v, window_sizes=window_sizes)
                # o_ours = flash_attn_func_ours(q, k, v, window_sizes=window_sizes, seqlen_q_vision = N, seqlen_k_vision = N)
                end_ours.record()
                torch.cuda.synchronize()
                tot_ours = (start_ours.elapsed_time(end_ours)) 
                ours_times.append(tot_ours)
        our_mean.append(np.mean(np.array(ours_times)))
        # output_ours.append(o_ours)
        # rights.append(torch.allclose(o_ours, output_originals[i]))

    print(f"shape: batch: {B}, seqlen: {N}, num_head: {H}, head_dim: {D}")
    for i in range(len(win_size_half)):
        # print(f"ours == original: \033[35m{rights[i]}\033[0m")
        print(f"{args.full_num}/{args.num_head} heads do full {args.num_head - args.full_num} heads do {win_size_half[i]*2} local attn \tous time: \033[35m{our_mean[i]:>12.6f}\033[0m\tfull attn original time: \033[35m{ori_mean[i]:>12.6f}\033[0m")


if __name__ == "__main__":
    main(sys.argv[1:])