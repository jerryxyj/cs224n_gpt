"""
LoRA (Low-Rank Adaptation) module for efficient fine-tuning.

Reference: https://arxiv.org/abs/2106.09685

LoRA freezes the pretrained model weights and injects trainable low-rank
decomposition matrices into each target layer, greatly reducing the number
of trainable parameters for downstream tasks.
"""

import math
import torch
from torch import nn


class LoRALinear(nn.Module):
    """
    A wrapper around nn.Linear that adds a low-rank adapter.

    The output is:  y = W_pretrained @ x + (alpha / r) * B @ A @ x

    Where:
      - W_pretrained is the frozen original weight matrix
      - A is a (r x in_features) matrix initialized with Kaiming uniform
      - B is a (out_features x r) matrix initialized with zeros
      - alpha is a scaling factor
      - r is the rank of the adaptation
    """

    def __init__(self, original_linear: nn.Linear, r: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        self.original_linear = original_linear
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        in_features = original_linear.in_features
        out_features = original_linear.out_features

        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        # Initialize A with Kaiming uniform (same as the paper)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # Optional dropout on input before LoRA path
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Freeze the original linear layer
        self.original_linear.weight.requires_grad = False
        if self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original frozen forward pass
        result = self.original_linear(x)
        # LoRA adapter path
        lora_out = self.lora_dropout(x)
        lora_out = torch.nn.functional.linear(lora_out, self.lora_A)  # x @ A^T
        lora_out = torch.nn.functional.linear(lora_out, self.lora_B)  # (x @ A^T) @ B^T
        result = result + lora_out * self.scaling
        return result


def apply_lora_to_model(model, r=8, alpha=16.0, dropout=0.0, target_modules=None):
    """
    Apply LoRA adapters to the specified target modules in the GPT-2 model.

    Args:
        model: The GPT2Model (model.gpt)
        r: Rank of the low-rank matrices
        alpha: Scaling factor
        dropout: Dropout rate for LoRA
        target_modules: List of module names to apply LoRA to.
                        Default targets the Q and V projections in attention,
                        which is the standard LoRA configuration.
                        Options: 'query', 'key', 'value', 'attention_dense',
                                 'interm_dense', 'out_dense'
    """
    if target_modules is None:
        target_modules = ['query', 'value']  # Standard LoRA: Q and V

    lora_count = 0
    for layer in model.gpt_layers:
        for target_name in target_modules:
            # Attention projections
            if target_name in ('query', 'key', 'value'):
                original_linear = getattr(layer.self_attention, target_name)
                lora_linear = LoRALinear(original_linear, r=r, alpha=alpha, dropout=dropout)
                setattr(layer.self_attention, target_name, lora_linear)
                lora_count += 1
            # Attention output dense
            elif target_name == 'attention_dense':
                original_linear = layer.attention_dense
                lora_linear = LoRALinear(original_linear, r=r, alpha=alpha, dropout=dropout)
                layer.attention_dense = lora_linear
                lora_count += 1
            # FFN intermediate dense
            elif target_name == 'interm_dense':
                original_linear = layer.interm_dense
                lora_linear = LoRALinear(original_linear, r=r, alpha=alpha, dropout=dropout)
                layer.interm_dense = lora_linear
                lora_count += 1
            # FFN output dense
            elif target_name == 'out_dense':
                original_linear = layer.out_dense
                lora_linear = LoRALinear(original_linear, r=r, alpha=alpha, dropout=dropout)
                layer.out_dense = lora_linear
                lora_count += 1

    print(f"Applied LoRA (r={r}, alpha={alpha}) to {lora_count} linear layers "
          f"(targets: {target_modules})")
    return model

