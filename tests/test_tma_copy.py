import torch 
import tma_copy_extension
import numpy as np 
from loguru import logger 

np.random.seed(0)
# np_x = np.random.uniform(-1, 1, (1024, 256))
# np_x = np.random.uniform(-1, 1, (1024, 256 + 64))
np_x = np.random.uniform(-1, 1, (1024, 256 + 32))



x = torch.tensor(np_x, dtype=torch.float32, device='cuda')
y = torch.zeros(np_x.shape, dtype=torch.float32, device='cuda')


tma_copy_extension.tma_copy(x, y)

torch.testing.assert_allclose(x, y)

logger.info(f"x is: {x=}")
logger.info(f"y is: {y=}")
