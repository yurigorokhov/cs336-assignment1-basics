from dataclasses import dataclass
import math

import torch


def cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    if it < warmup_iters:
        return (it / warmup_iters) * max_learning_rate
    elif it <= cosine_cycle_iters:
        return min_learning_rate + 0.5 * (
            1 + math.cos(math.pi * (it - warmup_iters) / (cosine_cycle_iters - warmup_iters))
        ) * (max_learning_rate - min_learning_rate)
    else:
        return min_learning_rate


@dataclass
class CosineLearningRate:
    max_learning_rate: float
    warmup_iters: int
    cosine_cycle_iters: int
    min_learning_rate: float | None = None
    min_learning_rate_ratio: float | None = None

    def __post_init__(self):
        if self.min_learning_rate is None:
            if self.min_learning_rate_ratio is None:
                raise ValueError("Either min_learning_rate or min_learning_rate_ratio must be set")
            else:
                self.min_learning_rate = self.max_learning_rate * self.min_learning_rate_ratio

    def update_lr(self, iteration: int, optimizer: torch.optim.Optimizer) -> float:
        lr = cosine_schedule(
            iteration,
            self.max_learning_rate,
            self.min_learning_rate,
            self.warmup_iters,
            self.cosine_cycle_iters,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        return lr
