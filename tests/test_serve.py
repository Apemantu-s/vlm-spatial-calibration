import io
from fastapi.testclient import TestClient
from PIL import Image
import serve


def png():
    buf = io.BytesIO()
    Image.new("RGB", (64, 48), (200, 30, 30)).save(buf, format="PNG")
    return buf.getvalue()


def client(monkeypatch, diff):
    monkeypatch.setattr(serve, "logit_diff", lambda img, statement: diff)
    monkeypatch.setitem(serve.state, "temperature", 2.0)
    return TestClient(serve.app)


def post(c, image):
    return c.post("/judge", data={"statement": "The cat is on the mat."}, files={"image": ("a.png", image, "image/png")})


def test_yes(monkeypatch):
    r = post(client(monkeypatch, 4.0), png())
    assert r.status_code == 200
    out = r.json()
    assert out["answer"] == "yes"
    assert abs(out["p_yes_raw"] - 0.982) < 1e-3
    assert abs(out["p_yes"] - 0.8808) < 1e-3
    assert out["confidence"] == out["p_yes"]


def test_no(monkeypatch):
    out = post(client(monkeypatch, -4.0), png()).json()
    assert out["answer"] == "no"
    assert abs(out["confidence"] - 0.8808) < 1e-3


def test_bad_image(monkeypatch):
    r = post(client(monkeypatch, 1.0), b"not an image")
    assert r.status_code == 400


def test_health(monkeypatch):
    assert client(monkeypatch, 1.0).get("/health").json()["ok"] == True