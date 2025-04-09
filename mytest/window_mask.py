import torch
import sys

def transform_dense_block_mask(mask: torch.Tensor, block_size=(64, 64)):
    mask = mask.clone()
    nhead, H, W = mask.shape
    mask = mask.view(nhead, H // block_size[0], block_size[0], W // block_size[1], block_size[1])
    mask_max = mask.amax(dim=[2, 4], keepdim=True)
    # print(f"mask_max.shape: {mask_max.shape}")
    torch.set_printoptions(threshold=float("inf"))
    print(mask_max)
    mask |= mask_max
    mask = mask.reshape(nhead, H, W)
    return mask

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
        print(f"making mask: w_left {window_size_left} w_right {window_size_right}")
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

# H = 2
# windows_size = torch.zeros((H, 2), device=torch.device("cuda"), dtype=torch.float16)
# windows_size[:, :] = 128
# torch.set_printoptions(threshold=float("inf"))
# sys.stdout = open("test_mask.log", "w")
# mask = create_window_mask(H, 1024, 128, windows_size, (128, 128)).unsqueeze(0)
# print(mask.shape)
# B, H, Q, KV = mask.shape
# mask = mask.view(
#     B, H, Q // 128, 128, KV // 128, 128
# )  # [B, H, Q//Q_BLOCK_SIZE, Q_BLOCK_SIZE, KV//KV_BLOCK_SIZE, KV_BLOCK_SIZE]
# mask = mask.permute(
#     0, 1, 2, 4, 3, 5
# )  # [B, H, Q//Q_BLOCK_SIZE, KV//KV_BLOCK_SIZE, Q_BLOCK_SIZE, KV_BLOCK_SIZE]
# mask_block_sum = mask.sum(
#     dim=[-2, -1]
# )