import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha, dropout):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

        device = base.weight.device
        self.A = nn.Parameter(torch.zeros(rank, base.in_features, device=device))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, device=device))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = alpha / rank
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        out = self.base(x)
        h = self.drop(x).to(self.A.dtype)
        h = h @ self.A.T
        h = h @ self.B.T
        return out + (h * self.scale).to(out.dtype)


def add_lora(model, rank, alpha, dropout, targets):
    for p in model.parameters():
        p.requires_grad = False

    found = []
    for name, module in model.named_modules():
        last = name.split(".")[-1]
        if last in targets and isinstance(module, nn.Linear):
            found.append(name)

    for name in found:
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name)
        base = getattr(parent, child_name)
        setattr(parent, child_name, LoRALinear(base, rank, alpha, dropout))
    return len(found)


def lora_state(model):
    state = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            state[name] = p.detach().cpu()
    return state


def count_params(model):
    total = 0
    trainable = 0
    for p in model.parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    return trainable, total