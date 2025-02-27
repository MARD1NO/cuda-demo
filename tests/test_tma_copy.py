import torch 
import tma_copy_extension
import numpy as np 


np.random.seed(0)
np_x = np.random.uniform(-1, 1, (32, 64))

x = torch.tensor(np_x, dtype=torch.float32, device='cuda')
y = torch.zeros(np_x.shape, dtype=torch.float32, device='cuda')


tma_copy_extension.tma_copy(x, y)



print(x)

print(y)
