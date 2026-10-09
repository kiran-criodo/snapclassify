#!/usr/bin/env bash
# Crio grader entry point for SnapClassify (one module, SNAP). Runs every test file in tests/visible and, when the
# grader adds them, the hidden tests in tests/assessment, then prints
#   TEST SUMMARY: RAN n TESTS, x SUCCEEDED, y FAILED
#   TEST_STATUS_SUCCESS          (only when every test passed)
# Offline and deterministic: synthetic images, random weights, no downloads, no key, TZ=UTC, PYTHONHASHSEED=0.
# Extra arguments go to pytest.
set -u
cd "$(dirname "$0")"
unset GEMINI_API_KEY LLM_API_KEY
export TZ=UTC PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1
PY="${PYTHON:-python3}"
exec "$PY" - "$@" <<'PYEOF'
import os, subprocess, sys, tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

files = [str(p) for kind in ("visible", "assessment") for p in sorted(Path("tests", kind).glob("test_*.py"))]
ran = ok = failed = 0
pytest_exit = 5
if files:
    with tempfile.TemporaryDirectory() as td:
        xml = os.path.join(td, "report.xml")
        pytest_exit = subprocess.call([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                                       "--continue-on-collection-errors", f"--junitxml={xml}", *files, *sys.argv[1:]])
        if os.path.exists(xml):
            for tc in ET.parse(xml).getroot().iter("testcase"):
                if tc.find("skipped") is not None:
                    continue
                ran += 1
                if tc.find("failure") is not None or tc.find("error") is not None:
                    failed += 1
                else:
                    ok += 1
print(f"TEST SUMMARY: RAN {ran} TESTS, {ok} SUCCEEDED, {failed} FAILED")
if pytest_exit == 0 and ran and not failed:
    print("TEST_STATUS_SUCCESS")
    sys.exit(0)
sys.exit(1)
PYEOF
