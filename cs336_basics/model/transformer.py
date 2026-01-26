import torch
from jaxtyping import Int, Float

from cs336_basics.model.embedding import Embedding
from cs336_basics.model.linear import Linear
from cs336_basics.model.multi_head_self_attn import RopeConfig
from cs336_basics.model.rms_norm import RmsNorm
from cs336_basics.model.transformer_block import TransformerBlock


class Transformer(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float | None,
        use_rms_norm: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.context_length = context_length
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        if rope_theta is None:
            rope_config = None
        else:
            rope_config = RopeConfig(context_length, rope_theta)
        self.layers = torch.nn.ModuleList(
            TransformerBlock(
                d_model, num_heads, d_ff, rope_config, use_rms_norm=use_rms_norm, device=device, dtype=dtype
            )
            for _ in range(num_layers)
        )
        if use_rms_norm:
            self.ln_final = RmsNorm(d_model, device=device, dtype=dtype)
        else:
            self.ln_final = None
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(
        self,
        x: Int[torch.Tensor, "batch_size sequence_length"],
    ) -> Float[torch.Tensor, "batch_size sequence_length vocab_size"]:
        """

        Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
        """
        y = self.token_embeddings(x)
        for layer in self.layers:
            y = layer(y)
        if self.ln_final is not None:
            y = self.ln_final(y)
        y = self.lm_head(y)
        return y
