#include <vector>
#include <fstream>
#include <iostream>
#include <torch/extension.h>
#include "mkl.h"
#include "mkl_spblas.h"
#include "mkl_types.h"
#include "omp.h"


template <typename scalar_t>
int mkl_spmm(
    int M,
    int K,
    int N,
    MKL_INT *row_index,
    MKL_INT *col_index,
    scalar_t *values,
    scalar_t *MB,
    scalar_t *MC,
    scalar_t alpha,
    scalar_t beta);

std::tuple<int, int> calculate_resolution(int ori_h, int ori_w, int kernel, int padding, int stride, int dilation)
{
    // Calculate the output resolution of the output tensor
    // printf("calculating output size: h:%d kernel:%d padding:%d stride:%d\n",ori_h, kernel, padding, stride);
    int h, w;
    h = int((ori_h + 2 * padding - kernel) / stride) + 1;
    w = int((ori_w + 2 * padding - kernel) / stride) + 1;
    return std::make_tuple(h, w);
}

inline int calculate_index(int id1, int shift1, int id2, int shift2, int id3, int shift3, int id4, int shift4 = 1)
{
    return id1 * shift1 + id2 * shift2 + id3 * shift3 + id4 * shift4;
}