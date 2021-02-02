import torch
import mkl_sparse_linear_cpp


class sparse_linear(torch.nn.Module):
    def __init__(self, ori_linear):
        super(sparse_linear, self).__init__()
        # initialize according to the original linear layer
        assert isinstance(
            ori_linear, torch.nn.Linear), 'Input should be the instance of the nn.Linear'
        self.out_features = ori_linear.out_features
        self.in_features = ori_linear.in_features
        self.row_index, self.col_index, self.values = self.convert_csr(
            ori_linear.weight.data)
        self.bias = ori_linear.bias

    def forward(self, data):
        return mkl_sparse_linear_cpp.forward(data, self.out_features, self.row_index, self.col_index, self.values, self.bias)

    def backward(self, *args):
        raise Exception('Current the sparse linear only support inference')

    def convert_csr(self, data, threshold=1e-8):
        """
        Convert the sparse tensor into the CSR format.
        Note: the values that lower than given threshold in the data will be taken as the sparsity.
        """
        assert len(data.size()) == 2, 'Only support the two-dimension data'
        with torch.no_grad():
            sparsity_pos = torch.abs(data) < threshold
            row_idx = []
            col_idx = []
            values = []
            H, W = sparsity_pos.size()
            for i in range(H):
                row_idx.append(len(values))
                for j in range(W):
                    if sparsity_pos[i][j] == True:
                        continue
                    col_idx.append(j)
                    values.append(data.data[i][j])
        row_idx, col_idx, values = torch.tensor(
            row_idx).to(torch.int32), torch.tensor(col_idx).to(torch.int32), torch.tensor(values)
        return row_idx, col_idx, values

    @property
    def sparsity_ratio(self):
        return self.out_features * self.in_features / self.values.size(0)
