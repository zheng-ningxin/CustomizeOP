import torch
import omp_sparse_conv_cpp


class sparse_conv(torch.nn.Module):
    def __init__(self, ori_conv):
        super(sparse_conv, self).__init__()




class dense_conv(torch.nn.Module):
    def __init__(self, ori_conv):
        super(dense_conv, self).__init__()
        # initialize according to the original conv layer
        assert isinstance(ori_conv, torch.nn.Conv2d), 'Input module should be the instance of the nn.Conv2d'
        self.out_channels = ori_conv.out_channels
        self.in_channels = ori_conv.in_channels
        self.weight = ori_conv.weight
        if hasattr(ori_conv, 'bias') and ori_conv.bias is not None:
            self.bias = ori_conv.bias
        else:
            self.bias = None
        self.padding = ori_conv.padding
        self.groups = ori_conv.groups
        self.stride = ori_conv.stride
        self.dilation = ori_conv.dilation
        
    def forward(self, data):
        return omp_sparse_conv_cpp.dense_forward(data, self.weight, self.bias, self.padding, self.dilation, self.stride, self.groups)

    def backward(self, *args):
        raise Exception('Current the sparse linear only support inference')
