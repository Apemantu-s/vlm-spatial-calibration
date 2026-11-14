import numpy as np


def sigmoid(x):
    x = np.clip(x, -50, 50)
    return 1 / (1 + np.exp(-x))


def p_yes(logit_diff, T=1.0):
    return sigmoid(np.asarray(logit_diff, dtype=float) / T)


def accuracy(p, y):
    pred = (p > 0.5).astype(int)
    return float((pred == y).mean())


def nll(p, y):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def brier(p, y):
    return float(((p - y) ** 2).mean())


def bin_stats(p, y, bins):
    pred = (p > 0.5).astype(int)
    conf = np.where(pred == 1, p, 1 - p)
    correct = (pred == y).astype(float)
    edges = np.linspace(0.5, 1.0, bins + 1)
    out = []
    for i in range(bins):
        lo = edges[i]
        hi = edges[i + 1]
        if i == bins - 1:
            inside = (conf >= lo) & (conf <= hi)
        else:
            inside = (conf >= lo) & (conf < hi)
        n = int(inside.sum())
        if n == 0:
            out.append({"lo": float(lo), "hi": float(hi), "n": 0, "conf": None, "acc": None})
        else:
            out.append({"lo": float(lo), "hi": float(hi), "n": n,
                        "conf": float(conf[inside].mean()), "acc": float(correct[inside].mean())})
    return out


def ece(p, y, bins=15):
    total = len(p)
    e = 0.0
    for b in bin_stats(p, y, bins):
        if b["n"] > 0:
            e += b["n"] / total * abs(b["acc"] - b["conf"])
    return float(e)


def confident_wrong(p, y, threshold=0.99):
    pred = (p > 0.5).astype(int)
    conf = np.where(pred == 1, p, 1 - p)
    wrong = pred != y
    return float((wrong & (conf > threshold)).mean())


def all_metrics(logit_diff, y, T=1.0, bins=15):
    p = p_yes(logit_diff, T)
    return {
        "n": int(len(y)),
        "accuracy": round(accuracy(p, y), 4),
        "ece": round(ece(p, y, bins), 4),
        "nll": round(nll(p, y), 4),
        "brier": round(brier(p, y), 4),
        "confident_wrong": round(confident_wrong(p, y), 4),
        "mean_confidence": round(float(np.maximum(p, 1 - p).mean()), 4),
    }