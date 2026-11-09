"""The website runs a JavaScript copy of the physics layer and of the neural network.

This test runs the Node script (tests/web/check_parity.mjs) that compares that copy with the Python code on the
55 cases in tests/web/parity_cases.json. For a bigger check make a set with scripts/make_web_parity.py --n 1500
--out big.json and pass it to the script. The test is skipped when Node.js is not installed.
"""
import os
import shutil
import subprocess

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_javascript_copy_matches_python():
    done = subprocess.run(["node", os.path.join(ROOT, "tests", "web", "check_parity.mjs")],
                          capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, (done.stdout + done.stderr)[-3000:]
