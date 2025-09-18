import math
import torch
from jaxtyping import Float, Int64


class Embedding(torch.nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        initialization_std = math.sqrt(1.0)
        self.embeddings = torch.nn.Parameter(
            torch.nn.init.trunc_normal_(
                torch.empty(num_embeddings, embedding_dim, dtype=dtype),
                mean=0,
                std=initialization_std,
                a=-3,
                b=3,
            )
        ).to(device)

    def forward(
        self, token_ids: Int64[torch.Tensor, " ... embedding_index"]
    ) -> Float[torch.Tensor, " ... embedding"]:
        return torch.index_select(
            self.embeddings,
            0,
            token_ids.flatten(),
        ).unflatten(dim=0, sizes=token_ids.shape)
