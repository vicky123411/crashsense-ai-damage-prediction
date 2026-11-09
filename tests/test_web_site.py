"""Open the website in a headless browser and check what it shows against the Python code.

Skipped when Playwright or its browser is missing:  pip install playwright && playwright install chromium
"""
import base64
import functools
import http.server
import json
import os
import threading
import time
from decimal import ROUND_HALF_UP, Decimal

import numpy as np
import pytest

from crashdamage.physics import scenario_from_dict, simulate

sync_api = pytest.importorskip("playwright.sync_api")

DOCS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs")
EXAMPLES = json.load(open(os.path.join(DOCS, "data", "static.json"), encoding="utf-8"))["examples"]


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


@pytest.fixture(scope="module")
def site():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(QuietHandler, directory=DOCS))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/"
    server.shutdown()
    server.server_close()


@pytest.fixture(scope="module")
def browser():
    with sync_api.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as exc:  # the browser itself is not installed
            pytest.skip(f"Chromium is not available: {exc}")
        yield b
        b.close()


class Session:
    """One open page plus everything it complained about."""

    def __init__(self, page):
        self.page = page
        self.problems = []
        page.on("pageerror", lambda e: self.problems.append(f"page error: {e}"))
        page.on("console", lambda m: self.problems.append(f"console {m.type}: {m.text}")
                if m.type in ("error", "warning") else None)

    def request_in_address(self):
        """The crash request that the page keeps in the address (the part after #s=)."""
        raw = self.page.evaluate("location.hash")[len("#s="):]
        return json.loads(base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4)).decode("utf-8"))["r"]

    def renders(self):
        return int(self.page.get_attribute("body", "data-renders") or 0)

    def after_redraw(self, action):
        """Do something and wait until the page has drawn the result again."""
        before = self.renders()
        action()
        self.page.wait_for_function("n => Number(document.body.dataset.renders) > n", arg=before)


@pytest.fixture
def session(browser, site):
    page = browser.new_page(viewport={"width": 1280, "height": 900})
    s = Session(page)
    page.goto(site)
    page.wait_for_selector("body.ready")
    yield s
    page.close()


def shown(x, digits):
    """A number the way the page prints it: rounded half up, with thousands separators."""
    q = Decimal(x).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP)
    return f"{q:,.{digits}f}"


def readout(page, key):
    return page.evaluate("k => document.querySelector(`[data-ro=${k}] .v`).firstChild.nodeValue", key)


def test_page_loads_without_errors(session):
    assert session.page.inner_text("#v-word").strip() != ""
    assert session.problems == []


@pytest.mark.parametrize("example", EXAMPLES, ids=[e["id"] for e in EXAMPLES])
def test_example_shows_the_python_numbers(session, example):
    pg = session.page
    session.after_redraw(lambda: pg.click("#eng-physics"))   # the physics layer can be compared exactly
    res = simulate(scenario_from_dict(example["request"]))
    want = {"crush_depth": shown(res.crush_depth, 2), "intrusion": shown(res.intrusion, 2),
            "peak_force_kn": shown(res.peak_force_kn, 0), "e_diss_kj": shown(res.e_diss_kj, 0),
            "delta_v": shown(res.delta_v * 3.6, 0)}
    session.after_redraw(lambda: pg.click(f"#examples-list .chip[data-id={example['id']}]"))
    got = {k: readout(pg, k) for k in want}
    assert got == want
    assert session.problems == []


@pytest.mark.parametrize("example", EXAMPLES, ids=[e["id"] for e in EXAMPLES])
def test_request_built_by_the_page_reproduces_the_example(session, example):
    """The form turns the example into a request again; Python must get the same damage from it."""
    pg = session.page
    session.after_redraw(lambda: pg.click(f"#examples-list .chip[data-id={example['id']}]"))
    a = simulate(scenario_from_dict(example["request"]))
    b = simulate(scenario_from_dict(session.request_in_address()))
    assert np.allclose(a.damage, b.damage, atol=1e-12)
    assert a.crush_depth == pytest.approx(b.crush_depth, abs=1e-12)
    assert a.peak_force_kn == pytest.approx(b.peak_force_kn, abs=1e-9)


def test_typing_in_a_number_box_keeps_every_digit(session):
    pg = session.page
    pg.select_option("select >> nth=0", "steel_crate")
    box = pg.locator('.field.compact:has(label:text-is("Width")) input')
    box.click()
    box.press("Control+a")
    session.after_redraw(lambda: pg.keyboard.type("0.805", delay=30))
    deadline = time.time() + 5                               # the last key press is drawn a moment later
    while session.request_in_address()["obstacle"]["width"] != pytest.approx(0.805) and time.time() < deadline:
        time.sleep(0.05)
    assert box.input_value() == "0.805"
    request = session.request_in_address()
    assert request["obstacle"]["width"] == pytest.approx(0.805)
    # a crate's loaded perimeter follows its width and depth, as in the training data
    assert request["obstacle"]["perimeter"] == pytest.approx(0.805 + request["obstacle"]["depth"])


def test_compare_shows_both_engines(session):
    pg = session.page
    session.after_redraw(lambda: pg.click("#eng-compare"))
    assert pg.is_visible("#v-diff")
    assert pg.locator(".ro .o:visible").count() == 5
    assert pg.locator("#parts tbody tr:first-child .dmg").count() == 2
    assert session.problems == []


def test_a_pasted_link_changes_the_crash(session):
    pg = session.page
    req = {"speed_kmh": 40, "face": "rear", "offset": 0.3, "angle_deg": 0, "obstacle": {"name": "concrete_block"}}
    blob = base64.urlsafe_b64encode(json.dumps({"r": req, "e": "physics"}).encode()).decode().rstrip("=")
    session.after_redraw(lambda: pg.evaluate("h => { location.hash = h; }", f"#s={blob}"))
    assert "concrete block" in pg.inner_text("#summary")
    assert "rear" in pg.inner_text("#summary")
    assert pg.is_checked("#eng-physics")
    assert session.problems == []


def test_a_broken_link_falls_back_to_the_first_example(browser, site):
    page = browser.new_page()
    s = Session(page)
    page.goto(site + "#s=this-is-not-a-crash")
    page.wait_for_selector("body.ready")
    assert page.get_attribute("#examples-list .chip[data-id=side_pole_29kmh]", "aria-pressed") == "true"
    assert s.problems == []
    page.close()
