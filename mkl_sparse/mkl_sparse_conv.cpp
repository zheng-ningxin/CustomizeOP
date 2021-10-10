#include "common.h"

template <typename scalar_t>
static void omp_im2col(
    const scalar_t *data_im,
    const int channels,
    const int height,
    const int width,
    const int output_height,
    const int output_width,
    const int kernel_h,
    const int kernel_w,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w,
    const int dilation_h,
    const int dilation_w,
    scalar_t *data_col)
{
  const int height_col = output_height;
  const int width_col = output_width;
  const int channels_col = channels * kernel_h * kernel_w;

#pragma omp parallel for collapse(1)
  for (int c_col = 0; c_col < channels_col; ++c_col)
  {
    int w_offset = c_col % kernel_w;
    int h_offset = (c_col / kernel_w) % kernel_h;
    int c_im = c_col / kernel_h / kernel_w;

    for (int h_col = 0; h_col < height_col; ++h_col)
    {
      int h_im = h_col * stride_h - pad_h + h_offset * dilation_h;

      for (int w_col = 0; w_col < width_col; ++w_col)
      {
        int w_im = w_col * stride_w - pad_w + w_offset * dilation_w;
        data_col[(c_col * height_col + h_col) * width_col + w_col] =
            (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width)
                ? data_im[(c_im * height + h_im) * width + w_im]
                : static_cast<scalar_t>(0);
      }
    }
  }
}

template <typename scalar_t>
static void omp_col2im(
    const scalar_t *data_col,
    const int channels,
    const int height,
    const int width,
    const int output_height,
    const int output_width,
    const int kernel_h,
    const int kernel_w,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w,
    const int dilation_h,
    const int dilation_w,
    scalar_t *data_im)
{
  std::fill_n(data_im, height * width * channels, scalar_t(0));

  const int64_t height_col = output_height;
  const int64_t width_col = output_width;
  const int64_t channels_col = channels * kernel_h * kernel_w;

#pragma omp parallel for collapse(1)
  for (int64_t c_col = 0; c_col < channels_col; ++c_col)
  {
    int64_t w_offset = c_col % kernel_w;
    int64_t h_offset = (c_col / kernel_w) % kernel_h;
    int64_t c_im = c_col / kernel_h / kernel_w;

    for (int64_t h_col = 0; h_col < height_col; ++h_col)
    {
      int64_t h_im = h_col * stride_h - pad_h + h_offset * dilation_h;

      for (int64_t w_col = 0; w_col < width_col; ++w_col)
      {
        int64_t w_im = w_col * stride_w - pad_w + w_offset * dilation_w;

        if (h_im >= 0 && h_im < height && w_im >= 0 && w_im < width)
          data_im[(c_im * height + h_im) * width + w_im] +=
              data_col[(c_col * height_col + h_col) * width_col + w_col];
      }
    }
  }
}

at::Tensor mkl_sparse_conv_forward(
    torch::Tensor input,
    torch::Tensor row_index,
    torch::Tensor col_index,
    torch::Tensor values,
    torch::Tensor bias,
    int out_channels,
    int pad_height,
    int pad_width,
    int dilation_height,
    int dilation_width,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width)
{
  int batch_size = input.size(0);
  int in_channels = input.size(1);
  int input_height = input.size(2);
  int input_width = input.size(3);
  int output_height = (input_height + 2 * pad_height -
                       (dilation_height * (kernel_height - 1) + 1)) /
                          stride_height +
                      1;
  int output_width = (input_width + 2 * pad_width -
                      (dilation_width * (kernel_width - 1) + 1)) /
                         stride_width +
                     1;
  int n_output_plane = in_channels * kernel_width * kernel_height;
  int output_length = output_height * output_width;
  torch::Tensor col = torch::zeros({batch_size, n_output_plane, output_length});
  torch::Tensor output = torch::zeros({batch_size, out_channels, output_height, output_width});
  torch::Tensor spmm_out = torch::zeros({batch_size, out_channels, output_height, output_width});
  int height_col = (output_height + 2 * pad_height -
                    (dilation_height * (kernel_height - 1) + 1)) /
                       stride_height + 1;
  int width_col = (output_width + 2 * pad_width -
                   (dilation_width * (kernel_width - 1) + 1)) /
                      stride_width + 1;

  AT_DISPATCH_FLOATING_TYPES(input.type(), "mkl_sparse_conv_forward", ([&] {
                               for (int bid = 0; bid < batch_size; bid++)
                               {
                                 torch::Tensor input_n = input.select(0, bid);
                                 torch::Tensor col_n = col.select(0, bid);
                                 omp_im2col(
                                     input_n.data_ptr<scalar_t>(),
                                     in_channels,
                                     input_height,
                                     input_width,
                                     output_height,
                                     output_width,
                                     kernel_height,
                                     kernel_width,
                                     pad_height,
                                     pad_width,
                                     stride_height,
                                     stride_width,
                                     dilation_height,
                                     dilation_width,
                                     col_n.data_ptr<scalar_t>());
                               }
                               mkl_spmm(
                                   out_channels,
                                   n_output_plane,
                                   output_length,
                                   row_index.data_ptr<int>(),
                                   col_index.data_ptr<int>(),
                                   values.data_ptr<scalar_t>(),
                                   col.data_ptr<scalar_t>(),
                                   spmm_out.data_ptr<scalar_t>(),
                                   static_cast<scalar_t>(1),
                                   static_cast<scalar_t>(0));
                               //

                               for (int bid = 0; bid < batch_size; bid++)
                               {
                                 // convert the output back to image format
                                 torch::Tensor spmm_out_n = spmm_out.select(0, bid);
                                 torch::Tensor output_n = output.select(0, bid);
                                 omp_col2im(
                                     spmm_out_n.data_ptr<scalar_t>(),
                                     out_channels,
                                     output_height,
                                     output_width,
                                     height_col,
                                     width_col,
                                     kernel_height,
                                     kernel_width,
                                     stride_height,
                                     stride_width,
                                     dilation_height,
                                     dilation_width,
                                     output_n.data_ptr<scalar_t>());
                               }
                             }));
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("conv_forward", &mkl_sparse_conv_forward, "MKL sparse conv forward");
}