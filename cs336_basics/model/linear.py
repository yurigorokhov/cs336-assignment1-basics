import math
import torch
from jaxtyping import Float


class Linear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        initialization_std = math.sqrt(2.0 / (in_features + out_features))
        self.w = torch.nn.Parameter(
            torch.nn.init.trunc_normal_(
                torch.empty(out_features, in_features, dtype=dtype),
                mean=0,
                std=initialization_std,
                a=-3 * initialization_std,
                b=3 * initialization_std,
            )
        ).to(device)

    def forward(
        self, x: Float[torch.Tensor, " ... in_features"]
    ) -> Float[torch.Tensor, " ... features_out"]:
        return x @ self.w.T
