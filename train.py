import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from config import *
from data import VSRDataset, Collate
from lora import add_lora, count_params
from utils import *


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--rank", type=int, default=RANK)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    set_seed(args.seed)

    model, processor = load_model()
    yes_id, no_id = answer_ids(processor)

    n_layers = add_lora(model, args.rank, args.rank * ALPHA_PER_RANK, LORA_DROPOUT, TARGETS)
    trainable, total = count_params(model)
    print("lora on", n_layers, "layers. trainable", trainable, "of", total,
          "=", round(100 * trainable / total, 3), "%")

    collate = Collate(processor)
    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = DataLoader(VSRDataset("train", args.limit), batch_size=BATCH, shuffle=True, generator=g,
                              collate_fn=collate, num_workers=WORKERS, drop_last=True)

    params = []
    for p in model.parameters():
        if p.requires_grad:
            params.append(p)
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)

    step = 0
    running = 0.0
    model.train()
    for epoch in range(args.epochs):
        for batch in train_loader:
            logits = answer_logits(model, batch)
            y = batch["labels"].to(DEVICE)
            target = torch.where(y == 1, yes_id, no_id)
            loss = F.cross_entropy(logits, target)

            loss.backward()
            opt.step()
            opt.zero_grad()
            step += 1

            running += loss.item()
            if step % LOG_EVERY == 0:
                print("epoch", epoch, "step", step, "loss", round(running / LOG_EVERY, 4))
                running = 0.0