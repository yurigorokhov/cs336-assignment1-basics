import math
from collections.abc import Iterable
import torch
from torch import Tensor
from jaxtyping import Float


def softmax(
    x: Float[Tensor, " ..."], dim: int, temperature: float | None = None
) -> Float[Tensor, " ..."]:
    # subtract max for numerical stability
    x = torch.subtract(x, torch.max(x, dim=dim, keepdim=True).values)

    # softmax
    if temperature is not None:
        exp = torch.exp(x / temperature + 1e-8)
        return torch.div(exp, torch.sum(exp, dim=dim, keepdim=True))
    else:
        exp = torch.exp(x)
        return torch.div(exp, torch.sum(exp, dim=dim, keepdim=True))


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... d_model d_k"],
    K: Float[Tensor, " ... d_model d_k"],
    V: Float[Tensor, " ... d_model d_v"],
    mask: Float[Tensor, " ... d_model d_model"] | None = None,
) -> Float[Tensor, " ... d_model d_v"]:
    d_k = Q.shape[-1]
    qk = torch.div(torch.einsum("...Qk,...Kk->...QK", Q, K), math.sqrt(d_k))
    if mask is not None:
        qk = torch.masked_fill(
            qk,
            mask=~mask,
            value=-torch.inf,
        )
    return torch.einsum("...QV,...Vv->...Qv", softmax(qk, dim=-1), V)


def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter], max_l2_norm: float | None, eps: float = 1e-6
):
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    total_norm_squared = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        total_norm_squared += torch.pow(param.grad.data, 2).sum()
    gradient_norm = torch.sqrt(total_norm_squared)
    if max_l2_norm is None:
        return gradient_norm.item()
    clip_coef = max_l2_norm / (gradient_norm + eps)
    if clip_coef < 1:
        for param in parameters:
            if param.grad is not None:
                param.grad.data *= clip_coef
    return min(gradient_norm.item(), max_l2_norm)
