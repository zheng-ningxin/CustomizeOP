#include <vector>
#include <fstream>
#include <iostream>
#include <torch/extension.h>
#include "mkl.h"
#include "mkl_spblas.h"
#include "mkl_types.h"
#include "omp.h"

template <typename scalar_t>
static void omp_im2col(
    const scalar_t *data_im,
    const int64_t channels,
    const int64_t height,
    const int64_t width,
    const int64_t output_height,
    const int64_t output_width,
    const int64_t kernel_h,
    const int64_t kernel_w,
    const int64_t pad_h,
    const int64_t pad_w,
    const int64_t stride_h,
    const int64_t stride_w,
    const int64_t dilation_h,
    const int64_t dilation_w,
    scalar_t *data_col)
{
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
    const scalar_t* data_col,
    const int64_t channels,
    const int64_t height,
    const int64_t width,
    const int64_t output_height,
    const int64_t output_width,
    const int64_t kernel_h,
    const int64_t kernel_w,
    const int64_t pad_h,
    const int64_t pad_w,
    const int64_t stride_h,
    const int64_t stride_w,
    const int64_t dilation_h,
    const int64_t dilation_w,
    scalar_t* data_im) {
  std::fill_n(data_im, height * width * channels, scalar_t(0));

  const int64_t height_col = output_height;
  const int64_t width_col = output_width;
  const int64_t channels_col = channels * kernel_h * kernel_w;

#pragma omp parallel for collapse(1)
  for (int64_t c_col = 0; c_col < channels_col; ++c_col) {
    int64_t w_offset = c_col % kernel_w;
    int64_t h_offset = (c_col / kernel_w) % kernel_h;
    int64_t c_im = c_col / kernel_h / kernel_w;

    for (int64_t h_col = 0; h_col < height_col; ++h_col) {
      int64_t h_im = h_col * stride_h - pad_h + h_offset * dilation_h;

      for (int64_t w_col = 0; w_col < width_col; ++w_col) {
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
    torch::Tensor row_idx,
    torch::Tensor col_idx,
    torch::Tensor values,
    torch::Tensor bias,
    int pad_height,
    int pad_width,
    int dilation_height,
    int dilation_width,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width
    )
{
    int64_t batch_size = input.size(0);
    int64_t in_channel = input.size(1);
    int64_t input_height = input.size(2);
    int64_t input_width = input.size(3);
    int64_t output_height = (input_height + 2 * pad_height -
                             (dilation_height * (kernel_height - 1) + 1)) /
                                stride_height +
                            1;
    int64_t output_width = (input_width + 2 * pad_width -
                            (dilation_width * (kernel_width - 1) + 1)) /
                               stride_width +
                           1;
    int64_t n_output_plane = n_input_plane * kernel_width * kernel_height;
    int64_t output_length = output_height * output_width;
    torch::Tensor col = torch::zeros({batch_size, n_output_plane, output_length});
    
    AT_DISPATCH_FLOATING_TYPES(input.type(), "mkl_sparse_conv_forward", ([&] {
        for(int bid=0; bid<batch_size; bid++){
          omp_im2col
        }
        mkl_spmm();
        for(int bid=0; bid<batch_size; bid++){
          omp_col2im
        }
    }));
    
}