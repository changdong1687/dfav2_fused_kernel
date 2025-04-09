import torch
import math
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from torch import Tensor
import time
import sys
import argparse
from typing import List

import torch.nn.functional as F


from flash_attn_original import flash_attn_func as flash_attn_func_ori

def native_attn(q, k, v, mask):
    B, N, H, D = q.shape

    P = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(D)

    P += mask

    S = torch.nn.Softmax(dim=-1)(P)

    O = torch.matmul(S, v)

    return O

def sdpa_attn(q, k, v, mask):
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)



print("hello world!")

def transform_dense_block_mask(mask: torch.Tensor, block_size=(64, 64)):
    mask = mask.clone()
    nhead, H, W = mask.shape
    mask = mask.view(nhead, H // block_size[0], block_size[0], W // block_size[1], block_size[1])
    mask_max = mask.amax(dim=[2, 4], keepdim=True)
    # print(f"mask_max.shape: {mask_max.shape}")
    torch.set_printoptions(threshold=float("inf"))
    mask |= mask_max
    mask = mask.reshape(nhead, H, W)
    return mask

"""
define a mask generate function, return a 3dim mask, it receive 5 params:n_heads, n_im_tokens, n_text_tokens, windows_size, block_size
and it return shape is (n_heads, n_im_tokens+n_text_tokens, n_im_tokens+n_text_tokens)

the window_sizes is the shape of (n_heads, 2), the first column is the left window size, the second column is the right window size
the block_size is the shape of (2,), the first element is the block size of row, the second element is the block size of column

the mask can be caulated as this:
    suppose that n_tot = n_im_tokens + n_text_tokens
    we first tiled each [n_tot, n_tot] mask with size block_size to [(n_tot + block_szie[0] - 1)//block_size[0], (n_tot + block_szie[1] - 1)//block_size[1]] block
    and now we get a block_mask
    so for each row in block_mask, if its row_idx > n_im_tokens // block_size[0], we set all elements in this row to 1
    if its row_idx <= n_im_tokens // block_size[0], we set all elements in this row to 0

    for each col in block_mask, if its col_idx > n_im_tokens // block_size[1], we set all elements in this col to 1
    if its col_idx <= n_im_tokens // block_size[1], we set all elements in this col to 0

    for top left corner of the block_mask, we consider each row,
    if the row_idx > col_idx + window_size[0], we set all elements in this row to 0
    if the row_idx < col_idx - window_size[1], we set all elements in this row to 0
    otherwise, we set all elements in this row to 1

    if the col_idx >  n_im_tokens // block_size[1], we use for each col in block_mask, if its col_idx > n_im_tokens // block_size[1], we set all elements in this col to 1
    now please define it your self:
    def create_window_mask(n_heads: int, n_im_tokens: int, n_text_tokens: int, windows_size, block_size: tuple[int, int]=None):
        pass
    # n_heads: int, n_im_tokens: int, n_text_tokens: int, windows_size, block_size: tuple[int, int]=None
    # -> torch.Tensor
    # def create_window_mask(n_heads: int, n_im_tokens: int, n_text_tokens: int, windows_size, block_size: tuple[int, int]=None):
    #     tot_tokens = n_im_tokens + n_text_tokens

    #     mask = torch.zeros((n_heads, tot_tokens, tot_tokens), dtype=torch.bool, device="cuda")
    #     i = torch.arange(tot_tokens, device="cuda")
    #     j = torch.arange(tot_tokens, device="cuda")
    #     diff = (i[:, None] - j[None, :])
    #     for idx in range(n_heads):
    #         window_size_left, window_size_right = windows_size[idx, 0], windows_size[idx, 1]
    #         img_mask = (diff <= window_size_left) & (diff >= -window_size_right)
    #         mask[idx, :n_im_tokens, :n_im_tokens] = img_mask
    #         if block_size is not None:
    #             BM, BN = block_size
    #             assert n_im_tokens % BM == 0 and n_im_tokens % BN == 0, f"n_im_tokens={n_im_tokens} must be divisible by BM={BM} and BN={BN}"
    #             img_mask=transform_dense_block_mask(img_mask.unsqueeze(0), block_size)
    #         mask[idx, :n_im_tokens, :n_im_tokens] = img_mask

    #     return mask   

"""

def create_mask_new(n_heads: int, n_im_tokens: int, n_text_tokens: int, windows_size, block_size: tuple[int, int]=None):
    tot_tokens = n_im_tokens + n_text_tokens
    img_tokens_mutiplier = n_im_tokens // block_size[0]
    bmask_sizes_h, bmask_sizes_w = (tot_tokens + block_size[0] - 1) // block_size[0], (tot_tokens + block_size[1] - 1) // block_size[1]
    block_mask = torch.ones((n_heads, bmask_sizes_h, bmask_sizes_w), device="cuda", dtype=torch.bool)
    img_mask = torch.zeros((n_im_tokens//block_size[0], n_im_tokens//block_size[1]), device="cuda", dtype=torch.bool)
    full_mask = torch.zeros((n_heads, tot_tokens, tot_tokens), device="cuda", dtype=torch.bool)

    for i in range(n_heads):
        for j in range(n_im_tokens//block_size[0]):
            start = int(max(0, (j * block_size[0] - windows_size[i][0]) / block_size[1]))
            end = int(min(img_tokens_mutiplier, ((j+1) * block_size[0] + windows_size[i][1]) / block_size[1]))
            img_mask[j, start:end] = True
        block_mask[i, :img_tokens_mutiplier, : n_im_tokens // block_size[1]] = img_mask
    full_mask = block_mask.repeat_interleave(block_size[0], dim=1).repeat_interleave(block_size[1], dim=2)[:, :tot_tokens, :tot_tokens]

    return full_mask, block_mask



def create_window_mask(n_heads: int, n_im_tokens: int, n_text_tokens: int, windows_size, block_size: tuple[int, int]=None):
    tot_tokens = n_im_tokens + n_text_tokens
    # assert n_im_tokens % BM == 0 and n_im_tokens % BN == 0, f"n_im_tokens={n_im_tokens} must be divisible by BM={BM} and BN={BN}"

    # mask_h = math.ceil((n_im_tokens + n_other_tokens) / block_size[0]) * block_size[0]
    # mask_w = math.ceil((n_im_tokens + n_other_tokens) / block_size[1]) * block_size[1]
    # spatial_mask_full = torch.ones(
    #     mask_h,
    #     mask_w,
    #     dtype=torch.bool,
    #     device="cuda",
    # )
    if block_size is not None:
        img_mask_sizes = (n_im_tokens // block_size[0]) * block_size[0]
    else:
        img_mask_sizes = n_im_tokens
    final_mask = torch.ones((n_heads, tot_tokens, tot_tokens), dtype=torch.bool, device="cuda")
    # mask = torch.zeros((n_heads, tot_tokens, tot_tokens), dtype=torch.bool, device="cuda")

    i = torch.arange(img_mask_sizes, device="cuda")
    j = torch.arange(img_mask_sizes, device="cuda")
    
    diff = (i[:, None] - j[None, :])  # 使用广播计算差值
    for idx in range(n_heads):
        window_size_left, window_size_right = windows_size[idx, 0], windows_size[idx, 1]
        sys.stdout = sys.__stdout__
        # print(f"making mask: w_left {window_size_left} w_right {window_size_right}")
        img_mask = (diff <= window_size_left) & (diff >= -window_size_right)
    
    # 更新掩码
        # mask[idx, :img_mask_sizes, :img_mask_sizes] = img_mask

        if block_size is not None:
            BM, BN = block_size
            assert img_mask_sizes % BM == 0 and img_mask_sizes % BN == 0, f"n_im_tokens={n_im_tokens} must be divisible by BM={BM} and BN={BN}"
            img_mask=transform_dense_block_mask(img_mask.unsqueeze(0), block_size)
            
        img_mask=img_mask.cuda()
        # print(img_mask)

        final_mask[idx, :img_mask_sizes, :img_mask_sizes] = img_mask
    # final_mask.masked_fill_(mask, -torch.inf)
    return final_mask

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



flex_attention = torch.compile(flex_attention)

FLEX_MASK_CACHE = {}
def flex_headwin_attn(q: Tensor, k: Tensor, v: Tensor, n_im_tokens: int, n_text_tokens: int, spatial_mask, block_size=(128, 128)):
    # cache_key = (f"{n_im_tokens}_{n_text_tokens}_{block_size}", spatial_mask)
    # print(f"n_im_tokens={n_im_tokens}, n_other_tokens={n_other_tokens}")
    # if cache_key in FLEX_MASK_CACHE:
    #     block_mask = FLEX_MASK_CACHE[cache_key]
    # else:
        

    def mask_function(b, h, q_idx, kv_idx):
        cond1 = spatial_mask[h, q_idx, kv_idx]
        return cond1

    total_tokens = n_im_tokens + n_text_tokens
    print(f"tot_tokens: {total_tokens}")
    block_mask = create_block_mask(
        mask_function,
        B=None,
        H=q.shape[1],
        Q_LEN=total_tokens,
        KV_LEN=total_tokens,
        BLOCK_SIZE=block_size,
    )
    print(f"block_mask shape in flex: {block_mask.shape}")
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
    parser.add_argument("-seqv", "--seqlen-vision", type=int, default=4096)
    parser.add_argument("-seqt", "--seqlen-text", type=int, default=0)
    parser.add_argument("-nh", "--num-head", type=int, default=16)
    parser.add_argument("-d", "--head-dim", type=int, default=64)
    parser.add_argument("--min-val", type=int, default=128)
    parser.add_argument("--max-val", type=int, default=256)
    
    args = parser.parse_args(argv)

    # B, N, H, D = args.batch, args.seqlen, args.num_head, args.head_dim                    # 2, 1024, 32, 128
    B, N, H, D = args.batch, args.seqlen_vision + args.seqlen_text, args.num_head, args.head_dim                    # 2, 1024, 32, 128
    device = torch.device("cuda")


    dtype = torch.bfloat16
    q, k, v = torch.randn(B * 3, N, H, D, dtype=dtype, device=device).split(B, dim=0)
    
    # k *= torch.arange(0, N,device=k.device).view(1, -1, 1, 1)

    # window_sizes, _ = gen_win_size_uniform(H, 128, 512, device=device)


    
    window_sizes = torch.zeros((H, 2), device="cuda", dtype=torch.int32)
    # window_sizes[:, :] = -1
    window_sizes[:12, :] = 128
    window_sizes[12:, :] = 256

    o_ori = torch.zeros_like(q, dtype=dtype, device=device)
    for i in range(H):
        o_ori_tmp = flash_attn_func_ori(q, k, v, window_size=tuple(window_sizes[i].tolist()))
        o_ori[:, :, i, :] = o_ori_tmp[:, :, i, :]
    
    
    

    from flash_attn_ours import flash_attn_func as flash_attn_func_ours
    o_ours = flash_attn_func_ours(q, k, v, window_sizes=window_sizes)
    torch.cuda.synchronize()
    from flash_attn_ours import flash_attn_func_headwin

    o_headWin = flash_attn_func_headwin(q, k, v, window_sizes=window_sizes, seqlen_q_vision = args.seqlen_vision, seqlen_k_vision = args.seqlen_vision)
    torch.cuda.synchronize()
    

    # window_masks = create_window_mask(H, args.seqlen_vision, args.seqlen_text, window_sizes, (128, 128))
    window_masks, block_mask = create_mask_new(H, args.seqlen_vision, args.seqlen_text, window_sizes, (128, 128))

    native_mask = torch.zeros(B, H, N, N, device="cuda", dtype=dtype)
    native_mask[window_masks.unsqueeze(0).repeat(2, 1, 1, 1)] = 0
    native_mask[~window_masks.unsqueeze(0).repeat(2, 1, 1, 1)] = -torch.inf

    o_native = native_attn(q.permute(0, 2, 1, 3).contiguous().to(torch.float32), 
                            k.permute(0, 2, 1, 3).contiguous().to(torch.float32), 
                            v.permute(0, 2, 1, 3).contiguous().to(torch.float32), 
                            mask=native_mask).permute(0, 2, 1, 3)
    o_sdpa = sdpa_attn(q.permute(0, 2, 1, 3).contiguous().to(torch.float32), 
                            k.permute(0, 2, 1, 3).contiguous().to(torch.float32), 
                            v.permute(0, 2, 1, 3).contiguous().to(torch.float32), 
                            mask=window_masks.to(torch.bool)).permute(0, 2, 1, 3)
    
    o_flex = flex_headwin_attn(q.permute(0, 2, 1, 3).contiguous().to(torch.float32), 
                               k.permute(0, 2, 1, 3).contiguous().to(torch.float32), 
                               v.permute(0, 2, 1, 3).contiguous().to(torch.float32), 
                               args.seqlen_vision, args.seqlen_text, window_masks, (128, 128)).permute(0, 2, 1, 3).contiguous()

    torch.cuda.synchronize()
    print(f"o_flex == o_native: {torch.allclose(o_flex, o_native, 1e-3, 1e-3)}")
    print(f"o_flex == o_ori: {torch.allclose(o_flex, o_ori.to(torch.float32), 1e-3, 1e-3)}")
    print(f"o_native == o_ori: {torch.allclose(o_ori.to(torch.float32), o_native, 1e-3, 1e-3)}")
    print(f"o_sdpa == o_ori: {torch.allclose(o_ori.to(torch.float32), o_sdpa, 1e-3, 1e-3)}")
    print(f"o_sdpa == o_native: {torch.allclose(o_native, o_sdpa, 1e-3, 1e-3)}")
    

    print(f"o_sdpa == o_headWin: {torch.allclose(o_headWin.to(torch.float32), o_sdpa, 1e-3, 1e-3)}")

    print(f"full vision o_headWin == o_ori: {torch.allclose(o_headWin, o_ori)}")
    print(f"full vision o_ours == o_ori: {torch.allclose(o_ours, o_ori)}")

    print(f"max error between o_sdpa and o_native: {abs((o_sdpa - o_native)).max()} mean: {abs((o_sdpa - o_native)).mean()}")
    print(f"max error between o_flex and o_native: {abs((o_flex - o_native)).max()} mean: {abs((o_flex - o_native)).mean()}")
    print(f"max error between o_headWin and o_native: {abs((o_headWin - o_native)).max()} mean: {abs((o_headWin - o_native)).mean()}")
    
    # torch.set_printoptions(threshold=float("inf"))
    # print(window_masks.shape)
    # print(block_mask[0].to(torch.int8))
    # print(o_flex - o_ours)
    print(f"{'='*36}\n{'=='}{' '*32}{'=='}\n{'=='}{' '*32}{'=='}\n{'='*36}")



    # window_sizes[:, :] = -1
    # o_ours = flash_attn_func(q, k, v, window_sizes=window_sizes)
    # torch.cuda.synchronize()
    # window_sizes[:12, :] = 128
    # window_sizes[12:, :] = 256
    # o_headWin = flash_attn_func_headwin(q, k, v, window_sizes=window_sizes, seqlen_q_vision = 0, seqlen_k_vision = 0)
    # torch.cuda.synchronize()
    # window_masks = create_window_mask(H, 0, args.seqlen_vision, window_sizes, (128, 128))
    # o_flex = flex_headwin_attn(q.permute(0, 2, 1, 3).contiguous(), 
    #                            k.permute(0, 2, 1, 3).contiguous(), 
    #                            v.permute(0, 2, 1, 3).contiguous(), 
    #                            0, args.seqlen_vision, window_masks, (128, 128)).permute(0, 2, 1, 3).contiguous()

    # torch.cuda.synchronize()
    # print(f"o_flex == o_ours: {torch.allclose(o_flex, o_ours, 1e-3, 1e-3)}")
    # print(f"full text o_headWin == o_ours: {torch.allclose(o_headWin, o_ours)}")
    # print(f"{'*'*8} headWin done! {'*'*8}")

# (n_heads: int, n_im_tokens: int, n_text_tokens: int, windows_size, block_size: tuple[int, int]=None
    
    print(f"q shape:{q.shape}, k shape: {k.shape}, v shape: {v.shape}, window_masks shape:{window_masks.shape}")
    # print(torch.all(window_masks))
    
    # print(window_masks.to(torch.int8))


    
    
    # print(o_flex.shape)
    

    # print(f"o_flex == o_ours: {torch.allclose(o_flex, o_ours, 1e-3, 1e-3)}")
    
    # print(f"max error between o_headwin and o_ours: {(o_headWin - o_ours).max()}")
    print(f"max error between o_flex and o_ours: {(o_flex - o_ours).max()}")
    # torch.set_printoptions(threshold=float("inf"))
    # print(o_flex - o_ours)

if __name__=="__main__":
    main(sys.argv[1:])