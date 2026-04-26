import torch
import torch.nn as nn

class DeepFMModel(nn.Module):
    def __init__(
            self,
            hash_size: int,
            num_dense: int = 13,
            num_sparse: int = 26,
            embed_dim: int = 16,
            mlp_dims: list[int] = [256, 128, 64],
            dropout: float = 0.2,
    ):
        super().__init__()

        self.hash_size = hash_size
        self.num_dense = num_dense
        self.num_sparse = num_sparse
        self.embed_dim = embed_dim

        # 一阶 LR
        self.dense_linear = nn.Linear(num_dense, 1)
        self.sparse_linear = nn.Embedding(hash_size, 1)

        self.sparse_embedding = nn.Embedding(hash_size, embed_dim)

        # DNN
        dnn_input_dim = num_sparse * embed_dim + num_dense

        layers = []
        input_dim = dnn_input_dim

        for hidden_dim in mlp_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 1))

        self.dnn = nn.Sequential(*layers)

        self.bias = nn.Parameter(torch.zeros(1))

        self._init_weights()

    def _init_weights(self):
        # FM / sparse 初始化
        nn.init.normal_(self.sparse_embedding.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.sparse_linear.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.dense_linear.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.dense_linear.bias)

        # DNN 初始化
        for module in self.dnn:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, dense, sparse):
        """
                dense:  [B, 13]
                sparse: [B, 26]
                """

        # 一阶
        dense_logit = self.dense_linear(dense)                  # [B, 1]
        sparse_logit = self.sparse_linear(sparse).sum(dim=1)    # [B, 1]
        linear_logit = dense_logit + sparse_logit               # [B, 1]

        # embedding
        embed = self.sparse_embedding(sparse)                    # [B, 26, k]

        # 二阶
        square_of_sum = embed.sum(dim=1) ** 2                   # [B, k]
        sum_of_square = (embed ** 2).sum(dim=1)                 # [B, k]

        fm_interaction = 0.5 * (square_of_sum - sum_of_square)  # [B, k]
        fm_logit = fm_interaction.sum(dim=1, keepdim=True)      # [B, 1]

        # 高阶
        dnn_input = torch.cat(
            [
                dense,
                embed.flatten(start_dim=1)
            ],
            dim=1
        )                                                       # [B, 13 + 26*k]

        dnn_logit = self.dnn(dnn_input)                         # [B, 1]

        # 总输出
        logit = linear_logit + fm_logit + dnn_logit + self.bias

        return logit.squeeze(1)