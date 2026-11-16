import argparse, json, os
import torch
from torch.utils.data import DataLoader
from config import *
from data import VSRDataset, Collate
from lora import add_lora, load_lora
from utils import *


def predict(model, loader, yes_id, no_id):
    rows = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            logits = answer_logits(model, batch)
            probs = torch.softmax(logits, dim=-1)
            for i in range(len(batch["ids"])):
                rows.append({
                    "id": batch["ids"][i],
                    "label": batch["labels"][i].item(),
                    "logit_diff": (logits[i, yes_id] - logits[i, no_id]).item(),
                    "other_mass": 1 - probs[i, yes_id].item() - probs[i, no_id].item(),
                })
            if len(rows) % 400 < len(batch["ids"]):
                print(len(rows), "/", len(loader.dataset))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--path", default=MODEL_NAME)
    ap.add_argument("--four_bit", action="store_true")
    ap.add_argument("--splits", nargs="+", default=["val", "test", "test_unseen"])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    model, processor = load_model(args.path, args.four_bit)
    yes_id, no_id = answer_ids(processor)

    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt, map_location="cpu")
        cfg = ckpt["config"]
        add_lora(model, cfg["rank"], cfg["alpha"], 0.0, cfg["targets"])
        load_lora(model, ckpt["lora"])
        print("loaded", args.ckpt, "from step", ckpt["val"]["step"])

    os.makedirs(PREDS, exist_ok=True)
    collate = Collate(processor)
    for split in args.splits:
        ds = VSRDataset(split, args.limit)
        loader = DataLoader(ds, batch_size=BATCH * 2, shuffle=False, collate_fn=collate, num_workers=WORKERS)
        rows = predict(model, loader, yes_id, no_id)
        by_id = {}
        for r in ds.rows:
            by_id[r["id"]] = r
        path = PREDS / (args.name + "_" + split + ".jsonl")
        with open(path, "w") as f:
            for r in rows:
                r["image"] = by_id[r["id"]]["image"]
                r["relation"] = by_id[r["id"]]["relation"]
                f.write(json.dumps(r) + "\n")
        print("wrote", path, len(rows))