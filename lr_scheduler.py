"""
lr_scheduler.py — Noam Learning Rate Scheduler
DA6401 Assignment 3

Reference:
"Attention Is All You Need"
https://arxiv.org/abs/1706.03762

Formula:
    lrate = d_model^(-0.5) *
            min(step^(-0.5),
                step * warmup_steps^(-1.5))
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler


# ════════════════════════════════════════════════════════════════
# Noam Scheduler
# ════════════════════════════════════════════════════════════════

class NoamScheduler(LRScheduler):
    """
    Noam Learning Rate Scheduler used in the Transformer paper.

    Learning rate schedule:

        lr = d_model^(-0.5) *
             min(step^(-0.5),
                 step * warmup_steps^(-1.5))

    Characteristics:
        • Warm-up phase:
            LR increases linearly.

        • Decay phase:
            LR decreases proportionally to inverse sqrt(step).

    Args:
        optimizer      : PyTorch optimizer
        d_model        : Transformer embedding dimension
        warmup_steps   : Number of warmup steps
        last_epoch     : Last epoch index
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        d_model: int,
        warmup_steps: int,
        last_epoch: int = -1,
    ) -> None:

        # --------------------------------------------------------
        # Store parameters
        # --------------------------------------------------------

        self.d_model = d_model
        self.warmup_steps = warmup_steps

        # --------------------------------------------------------
        # Initialize parent scheduler
        # --------------------------------------------------------

        super().__init__(optimizer, last_epoch)

    # ════════════════════════════════════════════════════════════

    def _get_lr_scale(self) -> float:
        """
        Compute Noam LR scaling factor.

        Formula:
            d_model^(-0.5) *
            min(step^(-0.5),
                step * warmup_steps^(-1.5))

        Returns:
            Scaling factor (float)
        """

        # Avoid step=0
        step = self.last_epoch + 1

        scale = (
            (self.d_model ** (-0.5))
            *
            min(
                step ** (-0.5),
                step * (self.warmup_steps ** (-1.5))
            )
        )

        return scale

    # ════════════════════════════════════════════════════════════

    def get_lr(self):
        """
        Compute learning rate for each parameter group.

        Returns:
            List of updated learning rates.
        """

        scale = self._get_lr_scale()

        return [
            base_lr * scale
            for base_lr in self.base_lrs
        ]


# ════════════════════════════════════════════════════════════════
# Helper Function
# ════════════════════════════════════════════════════════════════

def get_lr_history(
    d_model: int,
    warmup_steps: int,
    total_steps: int,
):
    """
    Simulate LR values across training steps.

    Useful for plotting LR curve.

    Args:
        d_model       : Transformer embedding size
        warmup_steps  : Warmup duration
        total_steps   : Total steps to simulate

    Returns:
        List of LR values
    """

    dummy_model = torch.nn.Linear(1, 1)

    optimizer = optim.Adam(
        dummy_model.parameters(),
        lr=1.0
    )

    scheduler = NoamScheduler(
        optimizer=optimizer,
        d_model=d_model,
        warmup_steps=warmup_steps
    )

    history = []

    for _ in range(total_steps):

        history.append(
            optimizer.param_groups[0]["lr"]
        )

        optimizer.step()
        scheduler.step()

    return history


# ════════════════════════════════════════════════════════════════
# Visualization Test
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    D_MODEL = 512
    WARMUP_STEPS = 4000
    TOTAL_STEPS = 20000

    lrs = get_lr_history(
        d_model=D_MODEL,
        warmup_steps=WARMUP_STEPS,
        total_steps=TOTAL_STEPS
    )

    plt.figure(figsize=(10, 5))

    plt.plot(lrs)

    plt.axvline(
        WARMUP_STEPS,
        color='red',
        linestyle='--',
        label=f'Warmup = {WARMUP_STEPS}'
    )

    plt.xlabel("Training Step")
    plt.ylabel("Learning Rate")

    plt.title(
        f"Noam Learning Rate Schedule\n"
        f"d_model={D_MODEL}"
    )

    plt.legend()

    plt.tight_layout()

    plt.show()