import time
import numpy as np
import torch
import torchvision
from mkl_sparse.mkl_sparse_linear import sparse_linear

def measure_time(model, data, runtimes=1000):
    model.eval()
    times = []
    for runtime in range(runtimes):
        start = time.time()
        model(data)
        end = time.time()
        times.append(end-start)
    _drop = int(runtimes * 0.1)
    mean = np.mean(times[_drop:-1*_drop])
    std = np.std(times[_drop:-1*_drop])
    return mean, std



linear = torch.nn.Linear(1024,1024)
linear.weight.data[512:] = 0
s_linear = sparse_linear(linear)
dummy_input = torch.ones(1024, 1024)
print(s_linear.row_index.type())

with torch.no_grad():
    print('Original Linear')
    ori_mean, ori_std = measure_time(linear, dummy_input)
    print("Time Mean:{}, Time Std:{}".format(ori_mean, ori_std))
    t_out = linear(dummy_input)
    print('Sum of the original out Tensor', torch.sum(t_out).item())
    print('Sparse Linear')
    sparse_mean, sparse_std =  measure_time(s_linear, dummy_input)
    print("Time Mean:{}, Time Std:{}".format(sparse_mean, sparse_std))
    print('Sparsity ratio', s_linear.sparsity_ratio)
    t_out = s_linear(dummy_input)
    print('Sum of the sparse out Tensor', torch.sum(t_out).item())

    # s_linear(dummy_input)