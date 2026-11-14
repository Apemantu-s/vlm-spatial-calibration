import hashlib, json, os
from config import *


def bucket(image):
    h = hashlib.md5(image.encode()).hexdigest()
    return int(h[:8], 16) / 16 ** 8


def make_splits(rows):
    unseen_images = set()
    for r in rows:
        if r["relation"] in HELD_OUT:
            unseen_images.add(r["image"])

    splits = {"train": [], "val": [], "test": [], "test_unseen": []}
    for r in rows:
        b = bucket(r["image"])
        if r["relation"] in HELD_OUT:
            splits["test_unseen"].append(r)
        elif r["image"] in unseen_images or b < TEST_FRAC:
            splits["test"].append(r)
        elif b < TEST_FRAC + VAL_FRAC:
            splits["val"].append(r)
        else:
            splits["train"].append(r)
    return splits


def load_split(name):
    rows = []
    with open(SPLITS / (name + ".jsonl")) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


if __name__ == "__main__":
    rows = []
    with open(DATA / "vsr.jsonl") as f:
        for line in f:
            r = json.loads(line)
            rows.append({"id": len(rows), "image": r["image"], "caption": r["caption"],
                         "label": r["label"], "relation": r["relation"]})

    splits = make_splits(rows)
    os.makedirs(SPLITS, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)

    info = {}
    for name in splits:
        part = splits[name]
        with open(SPLITS / (name + ".jsonl"), "w") as f:
            for r in part:
                f.write(json.dumps(r) + "\n")
        images = set()
        yes = 0
        for r in part:
            images.add(r["image"])
            yes += r["label"]
        info[name] = {"rows": len(part), "images": len(images), "frac_yes": round(yes / len(part), 4)}
        print(name, info[name])

    train_images = set()
    for r in splits["train"]:
        train_images.add(r["image"])
    for name in ["val", "test", "test_unseen"]:
        shared = 0
        for r in splits[name]:
            if r["image"] in train_images:
                shared += 1
        info[name]["rows_with_train_image"] = shared
        print(name, "rows whose image is in train:", shared)

    with open(RESULTS / "split.json", "w") as f:
        json.dump(info, f, indent=2)