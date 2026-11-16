import argparse, time
import numpy as np
import torch
from torch.utils.data import DataLoader
from config import *
from data import VSRDataset, Collate
from utils import *

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--four_bit", action="store_true")
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()

    model, processor = load_model(EXPORT, args.four_bit)
    model.eval()
    weights_gb = torch.cuda.memory_allocated() / 1e9
    torch.cuda.reset_peak_memory_stats()

    loader = DataLoader(VSRDataset("test", args.n + 10), batch_size=1, shuffle=False, collate_fn=Collate(processor))
    times = []
    with torch.no_grad():
        for batch in loader:
            torch.cuda.synchronize()
            t = time.time()
            answer_logits(model, batch)
            torch.cuda.synchronize()
            times.append((time.time() - t) * 1000)

    times = np.array(times[10:])
    name = "4bit" if args.four_bit else "bf16"
    out = {"mode": name, "n": len(times), "weights_gb": round(weights_gb, 2),
           "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
           "latency_ms_median": round(float(np.median(times)), 1),
           "latency_ms_p95": round(float(np.percentile(times, 95)), 1)}
    print(out)
    save_json(out, RESULTS / ("benchmark_" + name + ".json"))