#!/bin/python
import time
import numpy as np
import torch
import torchvision
from omp_sparse.omp_sparse_conv import dense_conv
from mkl_sparse.mkl_sparse_linear import sparse_linear

def measure_time(model, data, runtimes=10):
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


conv = torch.nn.Conv2d(3, 3, 3)
conv.weight.data[:] = 1
conv.bias.data[:] = 0
d_conv = dense_conv(conv)
dummy_input = torch.rand(1, 3, 224, 224)

with torch.no_grad():
    print('Original conv')
    # ori_mean, ori_std = measure_time(conv, dummy_input)
    # print("Time Mean:{}, Time Std:{}".format(ori_mean, ori_std))
    t_out = conv(dummy_input)
    # print(t_out)
    print(t_out.size())
    print('Sum of the original out Tensor', torch.sum(t_out).item())
    print('My Dense Conv')
    # _mean, _std =  measure_time(d_conv, dummy_input)
    # print("Time Mean:{}, Time Std:{}".format(_mean, _std))
    t_out = d_conv(dummy_input)
    # print(t_out)
    print('Sum of the sparse out Tensor', torch.sum(t_out).item())

    # s_linear(dummy_input)