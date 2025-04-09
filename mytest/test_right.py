import torch
import time
import sys
import argparse
# import flash_attn_original
# print(flash_attn_original.__file__)

# import flash_attn_ours
# print(flash_attn_ours.__file__)


from flash_attn import flash_attn_func as flash_attn_func_ori
# from flash_attn_original import flash_attn_func as flash_attn_func_ori

def gen_win_size_uniform(n_head, min_val, max_val, device, dtype=torch.int32):
    win_size_lr = torch.randint(min_val, max_val+1, (n_head,), dtype=dtype, device=device) # * 10
    window_sizes = torch.zeros((n_head, 2), dtype=dtype, device=device)
    for i in range(n_head):
        window_sizes[i, 0] = torch.randint(1, win_size_lr[i], (1,))
        window_sizes[i, 1] = win_size_lr[i] - window_sizes[i, 0]
    
    return window_sizes, win_size_lr


def local_attn_flops(q_shape, v_shape, window_size):
    win_size = window_size[:, 0] + window_size[:, 1]
    b, seqlen, num_head, h_dim = q_shape
    v_dim = v_shape[-1]
    flops = b * seqlen * win_size * 2 * h_dim + b * seqlen * 2 * win_size * h_dim
    return flops.sum().item()

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch", type=int, default=2)
    parser.add_argument("-n", "--seqlen", type=int, default=1024)
    parser.add_argument("-nh", "--num-head", type=int, default=32)
    parser.add_argument("-d", "--head-dim", type=int, default=35)
    parser.add_argument("--min-val", type=int, default=128)
    parser.add_argument("--max-val", type=int, default=256)
    
    args = parser.parse_args(argv)

    B, N, H, D = args.batch, args.seqlen, args.num_head, args.head_dim                    # 2, 1024, 32, 128
    device = torch.device("cuda")


    q, k, v = torch.randn(B * 3, N, H, D, dtype=torch.float16, device=device).split(B, dim=0)
    print(f"q's shape batch: {q.shape[0]} seqlen: {q.shape[1]} num-head: {q.shape[2]} head-dim: {q.shape[3]}")

    # v *= torch.arange(N, dtype=torch.float16, device=device).reshape(1, -1, 1, 1)
    window_sizes, _ = gen_win_size_uniform(H, 128, 512, device=device)
    # print(window_sizes)
    o_ori = torch.zeros_like(q, dtype=torch.float16, device=device)
    window_sizes[:H//2, :] = -1
    # window_sizes[:, :] = 0

    # print(window_sizes)
    o_ori = torch.zeros_like(q, dtype=torch.float16, device=device)
    for i in range(H):
        print(window_sizes[i])
        o_ori_tmp = flash_attn_func_ori(q, k, v, window_size=tuple(window_sizes[i].tolist()))
        torch.cuda.synchronize()
        o_ori[:, :, i, :] = o_ori_tmp[:, :, i, :]
    # o_ori = flash_attn_func_ori(q, k, v, window_size=(-1, -1))

    # o_ori = flash_attn_func_ori(q, k, v, window_size=(-1, -1))

    # for i in range(H):
    #     o_ori_tmp = flash_attn_func_ori(q, k, v, window_size=tuple(window_sizes[i].tolist()))
    #     o_ori[:, :, i, :] = o_ori_tmp[:, :, i, :]
    
    
    # o_ori = flash_attn_func_ori(q, k, v, window_size=window_size)
    torch.cuda.synchronize()
    print(f"original done!")

    # from flash_attn_ours import flash_attn_func_headwin
    from flash_attn_ours import flash_attn_func
    # print(f"{'*'*8} original done! {'*'*8}")
    # # nvtx.range_push("headwin_region")
    for _ in range(10):
        o_our = flash_attn_func(q, k, v, window_sizes=window_sizes)
    # # o_our = flash_attn_func_headwin(q, k, v, window_sizes=window_sizes, seqlen_k_vision=N, seqlen_q_vision= N)
    # torch.cuda.synchronize()
    # # nvtx.range_pop()
    # print(f"o_our == o_ori: {torch.allclose(o_ori, o_our)}")
    # print(f"max error between o_our and o_ori: {(o_ori - o_our).max()}")


if __name__=="__main__":
    main(sys.argv[1:])