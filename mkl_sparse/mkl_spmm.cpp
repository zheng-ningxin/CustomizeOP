#include "common.h"

template<typename scalar_t>
void debug_show(int * row_index, int n_row, int * col_index, scalar_t * value){
    std::cout<<"Debug Info"<<std::endl;
    int length = row_index[n_row];
    for(int i=0;i<length;i++){
        std::cout<< i <<" "<<value[i]<<std::endl;
    }

}

template <typename scalar_t>
sparse_status_t mkl_sparse_mm(
    int M, int K, int N,
    scalar_t alpha,
    sparse_matrix_t &SA,
    matrix_descr descr,
    scalar_t *B,
    float beta,
    scalar_t *C);

template <>
sparse_status_t mkl_sparse_mm<double>(
    int M, int K, int N,
    double alpha,
    sparse_matrix_t &SA,
    matrix_descr descr,
    double *B,
    float beta,
    double *C)
{
    sparse_status_t status;
    status = mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, alpha, SA, descr, SPARSE_LAYOUT_ROW_MAJOR, B, N, N, beta, C, N);
    return status;
}

template <>
sparse_status_t mkl_sparse_mm<float>(
    int M, int K, int N,
    float alpha,
    sparse_matrix_t &SA,
    matrix_descr descr,
    float *B,
    float beta,
    float *C)
{
    sparse_status_t status;
    status = mkl_sparse_s_mm(SPARSE_OPERATION_NON_TRANSPOSE, alpha, SA, descr, SPARSE_LAYOUT_ROW_MAJOR, B, N, N, beta, C, N);
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

    // debug_show<scalar_t>(row_index, M, col_index, values);
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
    // descr.mode = SPARSE_FILL_MODE_LOWER;
    // descr.diag = SPARSE_DIAG_NON_UNIT;
    status = mkl_sparse_set_mm_hint(SA, SPARSE_OPERATION_NON_TRANSPOSE, descr, SPARSE_LAYOUT_ROW_MAJOR, K, 1);

    if (status != SPARSE_STATUS_SUCCESS)
    {
        printf("CSR Sparse matrix created failed.\n");
        return -2;
    }

    status = mkl_sparse_mm<scalar_t>(M, K, N, alpha, SA, descr, MB, beta, MC);

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
    torch::Tensor t_input = input.t().contiguous() ;
    /// Note transpose just change the view of the tensor, the data
    /// that in the data_ptr are still not transposed! So we need call
    /// contiguous to make the data layout correct

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
                                       static_cast<scalar_t>(0));
                               }));
    // need transpose again
    return t_output.t() + bias;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &mkl_sparse_linear_forward, "MKL sparse forward");
}