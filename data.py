import torch
from PIL import Image
from torch.utils.data import Dataset
from config import *
from utils import read_jsonl


class VSRDataset(Dataset):
    def __init__(self, split, limit=None):
        self.rows = read_jsonl(SPLITS / (split + ".jsonl"))
        if limit is not None:
            self.rows = self.rows[:limit]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(IMAGES / r["image"]).convert("RGB")
        img.thumbnail((MAX_SIDE, MAX_SIDE))
        return {"image": img, "text": PROMPT.format(caption=r["caption"]),
                "label": r["label"], "id": r["id"]}


class Collate:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, items):
        texts = []
        images = []
        labels = []
        ids = []
        for it in items:
            msg = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": it["text"]}]}]
            texts.append(self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True))
            images.append(it["image"])
            labels.append(it["label"])
            ids.append(it["id"])
        batch = self.processor(text=texts, images=images, padding=True, return_tensors="pt")
        batch = dict(batch)
        batch["labels"] = torch.tensor(labels)
        batch["ids"] = ids
        return batch