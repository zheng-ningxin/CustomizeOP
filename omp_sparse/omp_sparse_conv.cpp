#include <vector>
#include <iostream>
#include <torch/extension.h>
#include <omp.h>
std::tuple<int, int> calculate_resolution(int ori_h, int ori_w, int kernel, int padding, int stride, int dilation)
{
    // Calculate the output resolution of the output tensor
    int h, w;
    h = int((ori_h + 2 * padding - kernel) / stride) + 1;
    w = int((ori_w + 2 * padding - kernel) / stride) + 1;
    return std::make_tuple(h, w);
}
inline int calculate_index(int id1, int shift1, int id2, int shift2, int id3, int shift3, int id4, int shift4 = 1)
{
    return id1 * shift1 + id2 * shift2 + id3 * shift3 + id4 * shift4;
}

template <typename scalar_t>
void _omp_dense_conv_forward(
    scalar_t *input,
    scalar_t *weight,
    scalar_t *output,
    int out_channels,
    int in_channels,
    int kernel,
    int stride,
    int groups,
    int dilation,
    int padding,
    int batch_size, int in_h, int in_w,
    int out_h, int out_w)
{
    /// Donnot support dilation temp
    int in_channel_step = in_channels / groups;
    int out_channel_step = out_channels / groups;
    int h_start = 0, w_start = 0;
    int h_end = in_h - kernel + 1, w_end = in_w - kernel + 1;
    /// weight size: [out_channel, in_channel, kernel, kernel]
    const int w_s_4 = 1;
    const int w_s_3 = w_s_4 * kernel;
    const int w_s_2 = w_s_3 * kernel;
    const int w_s_1 = w_s_2 * in_channels;
    /// input size: [batch, in_channel, in_h, in_w]
    const int i_s_4 = 1;
    const int i_s_3 = in_w * i_s_4;
    const int i_s_2 = in_h * i_s_3;
    const int i_s_1 = in_channels * i_s_2;
    /// output size [batch, out_channels, out_h, out_w]
    const int o_s_4 = 1;
    const int o_s_3 = o_s_4 * out_w;
    const int o_s_2 = o_s_3 * out_h;
    const int o_s_1 = o_s_2 * out_channels;
    if (__builtin_expect(padding > 0, true))
    {
        h_start -= padding;
        h_end += padding;
        w_start -= padding;
        w_end -= padding;
    }

// compute the tiling here
#pragma omp parallel for collapse(2)
    for (int b = 0; b < batch_size; b++)
    {
        for (int oc = 0; oc < out_channels; oc++)
        {
            int groupid = oc / out_channel_step;
            int start_inc = groupid * in_channel_step;
            int end_inc = start_inc + in_channel_step;
            for (int ic = start_inc; ic < end_inc; ic++)
            {
                for (int h = h_start; h < h_end; h += stride)
                {
                    for (int w = w_start; w < w_end; w += stride)
                    {
                        scalar_t tmp = 0;
                        for (int i = 0; i < kernel; i++)
                        {
                            for (int j = 0; j < kernel; j++)
                            {
                                int cur_h = h + i;
                                int cur_w = w + j;
                                if (__builtin_expect(cur_h >= 0 && cur_w >= 0 && cur_h < in_h && cur_w < in_w, true))
                                {
                                    tmp += weight[calculate_index(oc, w_s_1, ic, w_s_2, i, w_s_3, j, w_s_3)] *
                                           input[calculate_index(b, i_s_1, ic, i_s_2, cur_h, i_s_3, cur_w, i_s_4)];
                                    // tmp += weight[oc][ic][i][j] * input[b][ic][cur_h][cur_w];
                                }
                            }
                        }
                        // output[b][oc][h + kernel / 2][w + kernel / 2] += tmp
                        output[calculate_index(b, o_s_1, oc, o_s_2, h + kernel / 2, o_s_3, w + kernel / 2, o_s_4)] += tmp;
                    }
                }
            }
        }
    }
}

at::Tensor omp_dense_conv2d_forward(
    const torch::Tensor &input,
    const torch::Tensor &weight,
    const torch::Tensor &bias,
    // const std::optional<torch::Tensor> &bias = {},
    const int padding,
    const int dilation,
    const int groups,
    const int stride)
{

    int out_channels = weight.size(0);
    int in_channels = input.size(1);
    int batch_size = input.size(0);
    int ori_h = input.size(2);
    int ori_w = input.size(3);
    // [out_channels, in_channels, kernel_size, kernel_size]
    int kernel = weight.size(2);
    auto out_size = calculate_resolution(ori_h, ori_w, kernel, padding, stride, dilation);
    int out_h = std::get<0>(out_size);
    int out_w = std::get<1>(out_size);
    torch::Tensor output = torch::zeros({
        batch_size,
        out_channels,
    });
    AT_DISPATCH_FLOATING_TYPES(input.type(), "omp_dense_conv", ([&] {
                                   _omp_dense_conv_forward(
                                       input.data_ptr<scalar_t>(),
                                       weight.data_ptr<scalar_t>(),
                                       output.data_ptr<scalar_t>(),
                                       out_channels,
                                       in_channels,
                                       kernel,
                                       stride,
                                       groups,
                                       dilation,
                                       padding,
                                       batch_size, ori_h, ori_w,
                                       out_h, out_w);
                               }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("dense_forward", &omp_dense_conv2d_forward, "OMP conv forward");
}