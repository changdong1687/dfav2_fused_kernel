from time import time
import torch
# from flash_attn import flash_attn_func
from dfav2 import headwise_arrow_attn
# from flash_attn_ours import flash_attn_func
# import matplotlib.pyplot as plt
# from flash_attn import flash_attn_func as flash_attn_func_ori
B = 4
S = 4096 + 333
H = 24
D = 72
query = torch.ones(B, S, H, D, device="cuda", dtype=torch.float16).transpose(1,2)
key = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16).transpose(1,2)
value = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16).transpose(1,2)

window_sizes = torch.ones((H, 2), device="cuda", dtype=torch.int32) * 4096
# window_sizes = window_sizes * 4096
# window_sizes[:2,:] = 0

window_sizes = torch.tensor([[256,256],
        [128, 128],
        [-1, -1],
        [-1, -1],
        [-1, -1],
        [-1, -1],
        [-1, -1],
        [-1, -1],
        [-1, -1],
        [-1, -1],
        [-1, -1],
        [-1, -1],
        [-1, -1],
        [-1, -1],
        [-1, -1],
        [-1, -1],
        [-1, -1],
        [-1, -1],
        [-1, -1],
        [-1, -1],
        [-1, -1],
        [-1, -1],
        [-1, -1],
        [-1, -1]], dtype=torch.int32, device='cuda')

# window_sizes = torch.tensor([[-1, -1],
#         [-1, -1],
#         [-1, -1],
#         [-1, -1],
#         [-1, -1],
#         [-1, -1],
#         [-1, -1],
#         [-1, -1],
#         [-1, -1],
#         [-1, -1],
#         [-1, -1],
#         [-1, -1],
#         [-1, -1],
#         [-1, -1],
#         [-1, -1],
#         [-1, -1],
#         [-1, -1],
#         [-1, -1],
#         [-1, -1],
#         [-1, -1],
#         [-1, -1],
#         [-1, -1],
#         [-1, -1],
#         [-1, -1]], dtype=torch.int32, device='cuda')

# print(headwise_window_attn(query, key, value, window_sizes=window_sizes))
# output = headwise_arrow_attn(query, key, value, window_sizes=window_sizes, seqlen_q_vision = S - 333, seqlen_k_vision = S - 333)

# for i in range(100):
#     output = headwise_arrow_attn(query, key, value, window_sizes=window_sizes, seqlen_q_vision = S - 333, seqlen_k_vision = S - 333)
#     # output = flash_attn_func(query, key, value)
#     torch.cuda.synchronize()
# st = time()
# for i in range(20):
#     torch.cuda.synchronize()
#     for j in range(100):
#         output = headwise_arrow_attn(query, key, value, window_sizes=window_sizes, seqlen_q_vision = S - 333, seqlen_k_vision = S - 333)
#         # output = flash_attn_func(query, key, value)
#         # output = flex_attn_compiled(query, key, value)
#         torch.cuda.synchronize()

# # for i in range(100):
# #     output = flash_attn_func(query, key, value, window_sizes=window_sizes)
# #     torch.cuda.synchronize()
# # st = time()
# # for i in range(20):
# #     torch.cuda.synchronize()
# #     for j in range(100):
# #         output = flash_attn_func(query, key, value, window_sizes=window_sizes)
# #         # output = flex_attn_compiled(query, key, value)
# #         torch.cuda.synchronize()

# et = time()

# output1 = flash_attn_func(query, key, value, window_sizes=window_sizes)
# window_sizes[:,:] = 4096
# output2 = flash_attn_func(query, key, value, window_sizes=window_sizes)

# print(f"100 iteration time: {(et - st) / 20}s")
# output_img = output[0,:,0,:].cpu() * 100
# # breakpoint()
# plt.imshow(output_img, vmax=output_img.max(), vmin=output_img.min())
# plt.axis('off')  # 不显示坐标轴
# plt.savefig('output_img.png', bbox_inches='tight', pad_inches=0)  # 保存图像
# plt.show() 
# print(output)
# print(f"{(output1 == output2).all()}")