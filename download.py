import json, os, time
import requests
from config import *


def get(url):
    for i in range(5):
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            return r.content
        except requests.RequestException as e:
            if i == 4:
                raise
            print("retry", url, e)
            time.sleep(5 * 2 ** i)


def get_annotations():
    path = DATA / "vsr.jsonl"
    if not path.exists():
        with open(path, "wb") as f:
            f.write(get(VSR_URL))
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


if __name__ == "__main__":
    os.makedirs(IMAGES, exist_ok=True)
    rows = get_annotations()
    print(len(rows), "rows")

    links = {}
    for r in rows:
        links[r["image"]] = r["image_link"]
    print(len(links), "images")

    new = 0
    skipped = 0
    for name in sorted(links):
        path = IMAGES / name
        if path.exists():
            skipped += 1
            continue
        data = get(links[name])
        tmp = str(path) + ".part"
        with open(tmp, "wb") as f:
            f.write(data)
        os.rename(tmp, path)
        new += 1
        time.sleep(0.2)
        if new % 200 == 0:
            print(new + skipped, "/", len(links), "new", new, "skipped", skipped)
    print("done. new", new, "skipped", skipped)