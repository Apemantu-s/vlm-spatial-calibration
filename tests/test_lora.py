import torch
import torch.nn as nn
from lora import LoRALinear, add_lora, lora_state, load_lora, count_params


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(8, 8)
        self.v_proj = nn.Linear(8, 8)
        self.mlp = nn.Linear(8, 8)

    def forward(self, x):
        return self.mlp(self.q_proj(x) + self.v_proj(x))


def make():
    torch.manual_seed(0)
    return nn.Sequential(Block(), Block())


def test_starts_equal_to_base():
    model = make()
    x = torch.randn(3, 8)
    before = model(x)
    n = add_lora(model, 4, 8, 0.0, ["q_proj", "v_proj"])
    assert n == 4
    assert torch.allclose(model(x), before)


def test_only_lora_trains():
    model = make()
    add_lora(model, 4, 8, 0.0, ["q_proj", "v_proj"])
    trainable, total = count_params(model)
    assert trainable == 4 * (4 * 8 + 8 * 4)
    for name, p in model.named_parameters():
        assert p.requires_grad == (name.endswith(".A") or name.endswith(".B"))


def test_gradient_reaches_lora():
    model = make()
    add_lora(model, 4, 8, 0.0, ["q_proj"])
    model(torch.randn(3, 8)).sum().backward()
    layer = model[0].q_proj
    assert layer.B.grad.abs().sum() > 0
    assert layer.base.weight.grad is None


def test_save_and_load():
    a = make()
    add_lora(a, 4, 8, 0.0, ["q_proj", "v_proj"])
    for name, p in a.named_parameters():
        if p.requires_grad:
            p.data.normal_()
    b = make()
    add_lora(b, 4, 8, 0.0, ["q_proj", "v_proj"])
    load_lora(b, lora_state(a))
    x = torch.randn(3, 8)
    assert torch.allclose(a(x), b(x))


def test_base_in_bf16():
    base = nn.Linear(8, 8).to(torch.bfloat16)
    layer = LoRALinear(base, 4, 8, 0.0)
    assert layer.A.dtype == torch.float32
    out = layer(torch.randn(2, 8, dtype=torch.bfloat16))
    assert out.dtype == torch.bfloat16