from time import time
import torch
# from flash_attn import flash_attn_func
# from flash_attn import flash_attn_func
from flash_attn_ours import flash_attn_func
# from flash_attn import flash_attn_func as flash_attn_func_ori

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

B = 4
S = 4096
H = 24
D = 97
query = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16).transpose(1,2)
key = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16).transpose(1,2)
value = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16).transpose(1,2)

window_sizes = torch.ones((H, 2), device="cuda", dtype=torch.int32)
window_sizes = window_sizes * 4096
window_sizes = check_window_size(window_sizes, S, S, S)
print(window_sizes.shape)
# window_sizes[:,:] = 4096

# print(headwise_window_attn(query, key, value, window_sizes=window_sizes))
# flash_attn_func(query, key, value, window_sizes=window_sizes)
print(f"single compute \033[35mdone\033[0m")

for i in range(10):
    output = flash_attn_func(query.clone(), key.clone(), value.clone(), window_size=window_sizes[0])
    # output = flash_attn_func(query.clone(), key.clone(), value.clone(), window_sizes=window_sizes)
    torch.cuda.synchronize()

print(f"warmup done!")
st = time()
for i in range(20):
    torch.cuda.synchronize()
    for j in range(100):
        output = flash_attn_func(query, key, value, window_size=window_sizes[0])
        # output = flash_attn_func(query, key, value, window_sizes=window_sizes)
        # output = flex_attn_compiled(query, key, value)
        torch.cuda.synchronize()

et = time()

print(f"100 iteration time: {(et - st) / 20}s")