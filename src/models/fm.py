import torch
import torch.nn as nn


class FMModel(nn.Module):
    def __init__(self, hash_size: int, num_dense: int = 13, embed_dim: int = 16):
        super().__init__()

        self.sparse_linear = nn.Embedding(hash_size, 1)
        self.dense_linear = nn.Linear(num_dense, 1)

        self.sparse_embedding = nn.Embedding(hash_size, embed_dim)

        self.bias = nn.Parameter(torch.zeros(1))

        # 关键：小初始化，避免二阶项一开始爆炸
        nn.init.normal_(self.sparse_embedding.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.sparse_linear.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.dense_linear.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.dense_linear.bias)

    def forward(self, dense, sparse):
        dense_logit = self.dense_linear(dense)
        sparse_logit = self.sparse_linear(sparse).sum(dim=1)

        embed = self.sparse_embedding(sparse)

        square_of_sum = embed.sum(dim=1) ** 2
        sum_of_square = (embed ** 2).sum(dim=1)

        interaction = 0.5 * (square_of_sum - sum_of_square)
        interaction_logit = interaction.sum(dim=1, keepdim=True)

        logit = dense_logit + sparse_logit + interaction_logit + self.bias

        return logit.squeeze(1)