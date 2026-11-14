from config import HELD_OUT
from split import bucket, make_splits


def fake_rows():
    rels = ["on", "under", "left of", HELD_OUT[0]]
    rows = []
    for i in range(2000):
        rows.append({"id": i, "image": "img" + str(i // 3) + ".jpg", "caption": "x",
                     "label": i % 2, "relation": rels[i % 4]})
    return rows


def test_bucket_is_stable():
    assert bucket("000000296471.jpg") == bucket("000000296471.jpg")
    assert 0 <= bucket("000000296471.jpg") < 1


def test_every_row_used_once():
    rows = fake_rows()
    splits = make_splits(rows)
    ids = []
    for name in splits:
        for r in splits[name]:
            ids.append(r["id"])
    assert sorted(ids) == list(range(2000))


def test_no_image_shared_with_train():
    splits = make_splits(fake_rows())
    train_images = set(r["image"] for r in splits["train"])
    for name in ["val", "test", "test_unseen"]:
        for r in splits[name]:
            assert r["image"] not in train_images


def test_held_out_relations_only_in_unseen():
    splits = make_splits(fake_rows())
    for name in ["train", "val", "test"]:
        for r in splits[name]:
            assert r["relation"] not in HELD_OUT
    for r in splits["test_unseen"]:
        assert r["relation"] in HELD_OUT