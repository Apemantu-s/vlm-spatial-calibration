import torch
from torch.utils.data import DataLoader
from config import *
from data import VSRDataset, Collate
from utils import *


model, processor = load_model()
yes_id, no_id = answer_ids(processor)
print("yes id", yes_id, "no id", no_id)

ds = VSRDataset("val", 8)
loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=Collate(processor))
batch = next(iter(loader))

print("keys", list(batch.keys()))
print("input_ids", batch["input_ids"].shape)
print(processor.tokenizer.decode(batch["input_ids"][0]).replace("<|image_pad|>", ""))
n_image = (batch["input_ids"][0] == processor.image_token_id).sum().item()
print("image tokens in row 0:", n_image)

model.eval()
with torch.no_grad():
    logits = answer_logits(model, batch)
probs = torch.softmax(logits, dim=-1)
for i in range(len(batch["ids"])):
    top = torch.topk(probs[i], 5)
    words = []
    for j in range(5):
        words.append(repr(processor.tokenizer.decode(top.indices[j])) + " " + str(round(top.values[j].item(), 3)))
    print("label", batch["labels"][i].item(), "| p(yes)", round(probs[i, yes_id].item(), 3),
          "p(no)", round(probs[i, no_id].item(), 3), "| top 5:", ", ".join(words))

if DEVICE == "cuda":
    print("peak vram gb", round(torch.cuda.max_memory_allocated() / 1e9, 2))