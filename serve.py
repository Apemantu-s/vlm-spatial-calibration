import io, json, os, time
import torch
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from PIL import Image
from config import *
from calibration import p_yes

app = FastAPI()
state = {}


def load():
    from data import Collate
    from utils import load_model, answer_ids
    path = os.environ.get("MODEL_PATH", str(EXPORT))
    model, processor = load_model(path)
    model.eval()
    state["model"] = model
    state["collate"] = Collate(processor)
    state["yes_id"], state["no_id"] = answer_ids(processor)
    with open(os.path.join(path, "meta.json")) as f:
        state["temperature"] = json.load(f)["temperature"]


def logit_diff(img, statement):
    from utils import answer_logits
    if "model" not in state:
        load()
    item = {"image": img, "text": PROMPT.format(caption=statement), "label": 0, "id": 0}
    batch = state["collate"]([item])
    with torch.no_grad():
        logits = answer_logits(state["model"], batch)
    return (logits[0, state["yes_id"]] - logits[0, state["no_id"]]).item()


@app.get("/health")
def health():
    return {"ok": True, "model_loaded": "model" in state}


@app.post("/judge")
async def judge(statement: str = Form(...), image: UploadFile = File(...)):
    data = await image.read()
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="could not read the image")
    img.thumbnail((MAX_SIDE, MAX_SIDE))

    t = time.time()
    d = logit_diff(img, statement)
    ms = (time.time() - t) * 1000

    raw = float(p_yes(d))
    p = float(p_yes(d, state["temperature"]))
    return {"answer": "yes" if p > 0.5 else "no",
            "p_yes": round(p, 4),
            "confidence": round(max(p, 1 - p), 4),
            "p_yes_raw": round(raw, 4),
            "latency_ms": round(ms, 1)}