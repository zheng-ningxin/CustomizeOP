#include <vector>
#include <fstream>
#include <iostream>
#include <torch/extension.h>
#include "mkl.h"
#include "mkl_spblas.h"
#include "mkl_types.h"

template <typename scalar_t>
sparse_status_t mkl_sparse_mm(
    scalar_t alpha,
    sparse_matrix_t &SA,
    matrix_descr descr,
    scalar_t *B,
    int N,
    int K,
    float beta,
    scalar_t *C,
    int M);

template <>
sparse_status_t mkl_sparse_mm<double>(
    double alpha,
    sparse_matrix_t &SA,
    matrix_descr descr,
    double *B,
    int N,
    int K,
    float beta,
    double *C,
    int M)
{
    sparse_status_t status;
    status = mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, alpha, SA, descr, SPARSE_LAYOUT_ROW_MAJOR, B, N, K, beta, C, M);
    return status;
}

template <>
sparse_status_t mkl_sparse_mm<float>(
    float alpha,
    sparse_matrix_t &SA,
    matrix_descr descr,
    float *B,
    int N,
    int K,
    float beta,
    float *C,
    int M)
{
    sparse_status_t status;
    status = mkl_sparse_s_mm(SPARSE_OPERATION_NON_TRANSPOSE, alpha, SA, descr, SPARSE_LAYOUT_ROW_MAJOR, B, N, K, beta, C, M);
    return status;
}

template <typename scalar_t>
sparse_status_t mkl_create_csr(
    sparse_matrix_t *,
    int, int, int *, int *, int *, scalar_t *);

template <>
sparse_status_t mkl_create_csr<float>(
    sparse_matrix_t *pSA,
    int M, int K, int *row_index,
    int *row_index_1, int *col_index,
    float *values)
{
    sparse_status_t status;
    status = mkl_sparse_s_create_csr(pSA, SPARSE_INDEX_BASE_ZERO, M, K, row_index, row_index_1, col_index, values);
    return status;
}

template <>
sparse_status_t mkl_create_csr<double>(
    sparse_matrix_t *pSA,
    int M, int K, int *row_index,
    int *row_index_1, int *col_index,
    double *values)
{
    sparse_status_t status;
    status = mkl_sparse_d_create_csr(pSA, SPARSE_INDEX_BASE_ZERO, M, K, row_index, row_index_1, col_index, values);
    return status;
}

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
    scalar_t beta)
{
    sparse_matrix_t SA;
    sparse_status_t status;
    status = mkl_create_csr<scalar_t>(&SA, M, K, row_index, &(row_index[1]), col_index, values);
    if (status != SPARSE_STATUS_SUCCESS)
    {
        printf("CSR Sparse matrix created failed.\n");
        return -2;
    }
    matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    descr.mode = SPARSE_FILL_MODE_LOWER;
    descr.diag = SPARSE_DIAG_NON_UNIT;
    status = mkl_sparse_set_mm_hint(SA, SPARSE_OPERATION_NON_TRANSPOSE, descr, SPARSE_LAYOUT_ROW_MAJOR, K, 1);

    if (status != SPARSE_STATUS_SUCCESS)
    {
        printf("CSR Sparse matrix created failed.\n");
        return -2;
    }

    status = mkl_sparse_mm<scalar_t>(alpha, SA, descr, MB, N, K, beta, MC, M);
    if (status != SPARSE_STATUS_SUCCESS)
    {
        printf("Sparse MM failed!!!!\n");
        return -4;
    }
    return 0;
}

at::Tensor mkl_sparse_linear_forward(
    torch::Tensor input,
    int out_features,
    torch::Tensor row_index,
    torch::Tensor col_index,
    torch::Tensor values,
    torch::Tensor bias)
{
    int in_features = input.size(1);
    int batchsize = input.size(0);
    // MKL sparse only support: sparse matrix * dense matrix
    torch::Tensor t_input = input.t();
    torch::Tensor t_output = torch::zeros({out_features, batchsize});
    AT_DISPATCH_FLOATING_TYPES(input.type(), "mkl_sparse_linear_forward", ([&] {
                                   mkl_spmm(
                                       out_features,
                                       in_features,
                                       batchsize,
                                       row_index.data_ptr<int>(),
                                       col_index.data_ptr<int>(),
                                       values.data_ptr<scalar_t>(),
                                       t_input.data_ptr<scalar_t>(),
                                       t_output.data_ptr<scalar_t>(),
                                       static_cast<scalar_t>(1),
                                       static_cast<scalar_t>(1));
                               }));
    // need transpose again
    return t_output.t();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &mkl_sparse_linear_forward, "MKL sparse forward");
}