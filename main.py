import torch
import numpy as np

T = 100
attention = torch.randn((32, 8, 100, 100))
mask = torch.tril(torch.ones((T + 1, T + 1))).view(1, 1, T + 1, T + 1)


