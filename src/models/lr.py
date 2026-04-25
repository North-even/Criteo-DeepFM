import torch
import torch.nn as nn

class LRModel(nn.Module):
    def __init__(self, hash_size: int, num_dense: int = 13):
        super().__init__()

        # sparse: 每个 hashed categorical feature 查一个权重
        self.sparse_linear = nn.Embedding(hash_size, 1)

        # dense: 普通线性层
        self.dense_linear = nn.Linear(num_dense, 1)

        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, dense, sparse):
        # dense: [B, 13]
        # sparse: [B, 26]

        dense_logit = self.dense_linear(dense)  # [B, 1]
        sparse_logit = self.sparse_linear(sparse).sum(dim=1)  # [B, 1]

        logit = dense_logit + sparse_logit + self.bias
        return logit.squeeze(1)