"""Runs all tests without pytest:  python tests/run_tests.py
(With pytest installed you can also just run:  pytest)"""
import importlib
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

passed = failed = 0
for path in sorted(HERE.glob("test_*.py")):
    module = importlib.import_module(path.stem)
    for name in sorted(n for n in dir(module) if n.startswith("test_")):
        try:
            getattr(module, name)()
            passed += 1
            print(f"PASS  {path.stem}.{name}")
        except Exception:
            failed += 1
            print(f"FAIL  {path.stem}.{name}")
            traceback.print_exc()
print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
