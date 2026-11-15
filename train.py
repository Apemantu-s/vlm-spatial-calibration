import argparse, os, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from config import *
from calibration import all_metrics
from data import VSRDataset, Collate
from lora import add_lora, lora_state, count_params
from utils import *


def run_val(model, loader, yes_id, no_id):
    model.eval()
    diffs = []
    labels = []
    loss_sum = 0.0
    with torch.no_grad():
        for batch in loader:
            logits = answer_logits(model, batch)
            y = batch["labels"].to(DEVICE)
            target = torch.where(y == 1, yes_id, no_id)
            loss_sum += F.cross_entropy(logits, target, reduction="sum").item()
            diffs.extend((logits[:, yes_id] - logits[:, no_id]).tolist())
            labels.extend(batch["labels"].tolist())
    model.train()
    m = all_metrics(np.array(diffs), np.array(labels), bins=BINS)
    m["loss"] = round(loss_sum / len(labels), 4)
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--rank", type=int, default=RANK)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    set_seed(args.seed)
    out_dir = RUNS / args.name
    os.makedirs(out_dir, exist_ok=True)

    model, processor = load_model()
    yes_id, no_id = answer_ids(processor)

    n_layers = add_lora(model, args.rank, args.rank * ALPHA_PER_RANK, LORA_DROPOUT, TARGETS)
    trainable, total = count_params(model)
    print("lora on", n_layers, "layers. trainable", trainable, "of", total,
          "=", round(100 * trainable / total, 3), "%")

    model.activation_checkpointing_enable(activation_checkpointing_kwargs={"use_reentrant": False})
    model.config.text_config.use_cache = False

    collate = Collate(processor)
    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = DataLoader(VSRDataset("train", args.limit), batch_size=BATCH, shuffle=True, generator=g,
                              collate_fn=collate, num_workers=WORKERS, drop_last=True)
    val_loader = DataLoader(VSRDataset("val", args.limit), batch_size=BATCH * 2, shuffle=False,
                            collate_fn=collate, num_workers=WORKERS)

    params = []
    for p in model.parameters():
        if p.requires_grad:
            params.append(p)
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)

    total_steps = len(train_loader) * args.epochs // ACCUM
    warmup_steps = max(1, int(total_steps * WARMUP))
    print("batches per epoch", len(train_loader), "optimizer steps", total_steps, "warmup", warmup_steps)

    cfg = {"rank": args.rank, "alpha": args.rank * ALPHA_PER_RANK, "seed": args.seed, "epochs": args.epochs,
           "batch": BATCH, "accum": ACCUM, "lr": LR, "warmup_steps": warmup_steps, "total_steps": total_steps,
           "max_side": MAX_SIDE, "targets": TARGETS, "trainable_params": trainable, "model": MODEL_NAME}
    writer = SummaryWriter(out_dir)
    writer.add_text("config", str(cfg))

    history = []
    best = None
    step = 0
    seen = 0
    running = 0.0
    t_start = time.time()
    t_log = time.time()

    model.train()
    opt.zero_grad()
    for epoch in range(args.epochs):
        for batch in train_loader:
            logits = answer_logits(model, batch)
            y = batch["labels"].to(DEVICE)
            target = torch.where(y == 1, yes_id, no_id)
            loss = F.cross_entropy(logits, target)

            (loss / ACCUM).backward()
            running += loss.item()
            seen += 1
            if seen % ACCUM != 0:
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(params, CLIP)
            lr = lr_at(step, total_steps, warmup_steps, LR)
            for group in opt.param_groups:
                group["lr"] = lr
            opt.step()
            opt.zero_grad()
            step += 1

            if step % LOG_EVERY == 0:
                dt = time.time() - t_log
                log = {"train/loss": running / (LOG_EVERY * ACCUM), "train/lr": lr,
                       "train/grad_norm": grad_norm.item(), "train/epoch": seen / len(train_loader),
                       "train/samples_per_sec": LOG_EVERY * ACCUM * BATCH / dt}
                if DEVICE == "cuda":
                    log["train/peak_vram_gb"] = torch.cuda.max_memory_allocated() / 1e9
                for k in log:
                    writer.add_scalar(k, log[k], step)
                print(step, "/", total_steps, "loss", round(log["train/loss"], 4), "lr", round(lr, 7),
                      "samples/s", round(log["train/samples_per_sec"], 2),
                      "vram", round(log.get("train/peak_vram_gb", 0), 2))
                running = 0.0
                t_log = time.time()

            if step % EVAL_EVERY == 0 or step == total_steps:
                m = run_val(model, val_loader, yes_id, no_id)
                m["step"] = step
                history.append(m)
                for k in ["loss", "accuracy", "ece"]:
                    writer.add_scalar("val/" + k, m[k], step)
                print("val", m)
                if best is None or m["loss"] < best["loss"]:
                    best = m
                    torch.save({"lora": lora_state(model), "config": cfg, "val": m}, out_dir / "best.pt")
                    print("saved best at step", step)
                t_log = time.time()

    summary = {"name": args.name, "config": cfg, "best": best, "history": history,
               "train_minutes": round((time.time() - t_start) / 60, 1)}
    if DEVICE == "cuda":
        summary["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    save_json(summary, RESULTS / "runs" / (args.name + ".json"))
    writer.close()