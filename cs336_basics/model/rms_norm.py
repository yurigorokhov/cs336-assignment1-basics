import torch
from jaxtyping import Float


class RmsNorm(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.eps = eps
        self.d_model = d_model
        self.gain = torch.nn.Parameter(torch.ones(d_model, dtype=dtype)).to(device)

    def forward(
        self, x: Float[torch.Tensor, " batch seq d_model"]
    ) -> Float[torch.Tensor, " ... batch seq d_model"]:
        original_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt((x.square() + self.eps).sum(-1) / self.d_model)
        x = x * self.gain / rms.unsqueeze(-1)
        return x.to(original_dtype)
