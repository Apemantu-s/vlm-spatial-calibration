import argparse, json
import torch
from config import *
from lora import add_lora, load_lora, merge_lora
from utils import *

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()

    model, processor = load_model()
    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt["config"]
    add_lora(model, cfg["rank"], cfg["alpha"], 0.0, cfg["targets"])
    load_lora(model, ckpt["lora"])
    n = merge_lora(model)
    print("merged", n, "layers")

    model.save_pretrained(EXPORT)
    processor.save_pretrained(EXPORT)

    with open(RESULTS / "report.json") as f:
        report = json.load(f)
    T = report["models"][args.name]["temperature"]
    save_json({"name": args.name, "temperature": T, "config": cfg}, EXPORT / "meta.json")