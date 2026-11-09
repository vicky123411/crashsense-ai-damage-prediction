import json
import threading
import urllib.error
import urllib.request

import pytest

from crashdamage.export import Predictor
from crashdamage.serve import create_server
from crashdamage.vehicle import N_PARTS


@pytest.fixture(scope="module")
def base_url():
    server = create_server(Predictor(), port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def post(url, payload, raw=False):
    data = payload if raw else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def get(url):
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_health_parts_and_obstacles(base_url):
    code, body = get(base_url + "/health")
    assert code == 200 and body["status"] == "ok" and body["model_loaded"] is False
    code, body = get(base_url + "/parts")
    assert code == 200 and len(body["parts"]) == N_PARTS and len(body["edges"]) > 20
    code, body = get(base_url + "/obstacles")
    assert code == 200 and "steel_pole" in body["presets"]


def test_predict_returns_a_report(base_url):
    code, body = post(base_url + "/predict", {"speed_kmh": 50, "face": "front", "obstacle": {"name": "concrete_wall"}})
    assert code == 200 and len(body["parts"]) == N_PARTS and body["engine"] == "physics"


def test_bad_requests_get_400(base_url):
    assert post(base_url + "/predict", b"{not json", raw=True)[0] == 400
    assert post(base_url + "/predict", {"speed_kmh": 50, "face": "top"})[0] == 400
    assert post(base_url + "/predict?engine=model", {"speed_kmh": 50})[0] == 400
    assert get(base_url + "/nothing")[0] == 404
