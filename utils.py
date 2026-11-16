import json, math, os, random
import numpy as np
import torch
from config import *

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def lr_at(step, total_steps, warmup_steps, base_lr):
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


def load_model(path=MODEL_NAME, four_bit=False):
    from transformers import AutoModelForImageTextToText, AutoProcessor
    processor = AutoProcessor.from_pretrained(path)
    processor.tokenizer.padding_side = "left"
    if four_bit:
        from transformers import BitsAndBytesConfig
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForImageTextToText.from_pretrained(path, quantization_config=quant, device_map="cuda")
    else:
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


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print("wrote", path)


def read_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows