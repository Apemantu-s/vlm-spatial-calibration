from utils import lr_at


def test_warmup_goes_up():
    lrs = [lr_at(s, 1000, 100, 1e-4) for s in range(100)]
    assert lrs[0] > 0
    assert lrs == sorted(lrs)
    assert abs(lrs[99] - 1e-4) < 1e-12


def test_decay_goes_down_to_zero():
    lrs = [lr_at(s, 1000, 100, 1e-4) for s in range(100, 1000)]
    assert lrs == sorted(lrs, reverse=True)
    assert lrs[-1] < 1e-8


def test_halfway_is_half():
    assert abs(lr_at(550, 1000, 100, 1e-4) - 0.5e-4) < 1e-9