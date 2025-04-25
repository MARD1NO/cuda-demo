import torch 
import numpy as np 
from deep_bf16_gemm_extension import deep_gemm_bf16

M = 1024
N = 1024 
K = 1024 


np.random.seed(0)
x = np.random.uniform(-0.1, 0.1, (M, K)).astype(np.float32)
weight = np.random.uniform(-0.1, 0.1, (N, K)).astype(np.float32)

torch_x = torch.tensor(x, device="cuda", dtype=torch.bfloat16)
# torch_x = torch.ones_like(torch_x)
torch_weight = torch.tensor(weight, device="cuda", dtype=torch.bfloat16)
# torch_weight = torch.ones_like(torch_weight)

deep_gemm_bf16_out = torch.zeros((M, N), device="cuda", dtype=torch.bfloat16)
deep_gemm_bf16(torch_x, torch_weight, deep_gemm_bf16_out)

torch_out = torch.matmul(torch_x, torch_weight.T)
# torch_out = torch.matmul(torch_x, torch_weight)

print("deep_gemm_bf16_out", deep_gemm_bf16_out)

print("torch_out", torch_out)

