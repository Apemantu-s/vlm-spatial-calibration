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