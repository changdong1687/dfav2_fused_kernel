import torch

def create_full_arrow_mask_trans(n_heads: int, n_im_tokens: int, n_text_tokens: int, windows_size, block_size: tuple[int, int]=None, dtype=torch.int32, device=torch.device("cuda")):
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
        txt_mask_row_num = (n_text_tokens + block_size[0] - 1) // block_size[0]
        txt_mask_col_num = (n_text_tokens + block_size[1] - 1) // block_size[1]
        img_mask_row_num = (tot_tokens - n_text_tokens) // block_size[0]
        img_mask_col_num = (tot_tokens - n_text_tokens) // block_size[1]

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
        block_mask[i, txt_mask_row_num :, txt_mask_col_num : ] = img_mask

        # if i == 0:
        #     torch.set_printoptions(threshold=float("inf"))
            # print(block_mask[i])
    full_mask = block_mask.repeat_interleave(block_size[0], dim=1).repeat_interleave(block_size[1], dim=2)[:, :tot_tokens, :tot_tokens]
    # print(full_mask[0])

    return full_mask, block_mask

def create_full_arrow_mask(n_heads: int, n_im_tokens: int, n_text_tokens: int, windows_size, block_size: tuple[int, int]=None, dtype=torch.int32, device=torch.device("cuda")):
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


def create_half_arrow_mask(n_heads: int, n_im_tokens: int, n_text_tokens: int, windows_size, block_size: tuple[int, int]=None, dtype=torch.int32, device=torch.device("cuda")):
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
            block_mask[i, :img_mask_row_num, : img_mask_col_num] = img_mask
        else:
            img_mask = torch.zeros((img_mask_row_num, img_mask_col_num), device=device, dtype=torch.bool)
            for j in range(img_mask_row_num):
                start = int(max(0, (j * block_size[0] - windows_size[i, 0]) / block_size[1]))
                end = int(min(img_mask_col_num, ((j+1) * block_size[0] + windows_size[i, 1] + block_size[1] - 1) / block_size[1]))
               
                img_mask[j, start:end] = True
            block_mask[i, :img_mask_row_num, : img_mask_col_num] = img_mask
            block_mask[i, img_mask_row_num : , : img_mask_col_num] = False

        
    full_mask = block_mask.repeat_interleave(block_size[0], dim=1).repeat_interleave(block_size[1], dim=2)[:, :tot_tokens, :tot_tokens]
    # print(full_mask[0])

    return full_mask, block_mask

def create_headwise_window_mask(n_heads: int, tot_tokens: int, windows_size, block_size: tuple[int, int]=None, dtype=torch.int32, device=torch.device("cuda")):
    bmask_sizes_h, bmask_sizes_w = (tot_tokens + block_size[0] - 1) // block_size[0], (tot_tokens + block_size[1] - 1) // block_size[1]
    

    full_mask = torch.zeros((n_heads, tot_tokens, tot_tokens), device=device, dtype=torch.bool)
    block_mask = torch.zeros((n_heads, bmask_sizes_h, bmask_sizes_w), device=device, dtype=torch.bool)


    for i in range(n_heads):
        if (windows_size[i, 0] == -1 and windows_size[i, 1] == -1):
            img_mask = torch.ones((bmask_sizes_h, bmask_sizes_w), device=device, dtype=torch.bool)
            block_mask[i, :, : ] = img_mask
        else:
            
            img_mask = torch.zeros((bmask_sizes_h, bmask_sizes_w), device=device, dtype=torch.bool)
            for j in range(bmask_sizes_h):
                start = int(max(0, (j * block_size[0] - windows_size[i, 0]) / block_size[1]))
                end = int(min(bmask_sizes_w, ((j+1) * block_size[0] + windows_size[i, 1] + block_size[1] - 1) / block_size[1]))
               
                img_mask[j, start:end] = True
            block_mask[i] = img_mask
        
    full_mask = block_mask.repeat_interleave(block_size[0], dim=1).repeat_interleave(block_size[1], dim=2)[:, :tot_tokens, :tot_tokens]

    return full_mask, block_mask