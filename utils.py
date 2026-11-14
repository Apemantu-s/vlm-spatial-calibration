import json
import torch
from config import *

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(path=MODEL_NAME):
    from transformers import AutoModelForImageTextToText, AutoProcessor
    processor = AutoProcessor.from_pretrained(path)
    processor.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16)
    model.to(DEVICE)
    return model, processor


def answer_ids(processor):
    yes = processor.tokenizer.encode(YES, add_special_tokens=False)
    no = processor.tokenizer.encode(NO, add_special_tokens=False)
    assert len(yes) == 1 and len(no) == 1, (yes, no)
    return yes[0], no[0]


def answer_logits(model, batch):
    inputs = {}
    for k in batch:
        if k not in ["labels", "ids"]:
            inputs[k] = batch[k].to(DEVICE)
    with torch.autocast(DEVICE, dtype=torch.bfloat16):
        out = model(**inputs)
    return out.logits[:, -1, :].float()


def read_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows