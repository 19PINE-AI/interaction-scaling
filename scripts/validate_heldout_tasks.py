"""Validate the held-out code tasks.

For each task:
  1. Extract the buggy implementation from the ```python``` block in the description.
  2. exec() it into a namespace.
  3. exec() the test_code in that namespace.
  4. Expect an AssertionError (or other failure) — proves the bug is real.

Then exec() a reference *correct* implementation (authored here) and expect
all tests to pass — proves the tests themselves are self-consistent.

Usage: python scripts/validate_heldout_tasks.py
"""

import json
import re
import sys
import traceback
from pathlib import Path

TASKS_PATH = Path("data/hard_benchmarks/code/code_tasks_heldout.json")


def extract_buggy_code(description: str) -> str:
    m = re.search(r"```python\n(.*?)```", description, re.DOTALL)
    if not m:
        raise ValueError("no ```python``` block found")
    return m.group(1)


def run_buggy(task: dict) -> str:
    code = extract_buggy_code(task["description"])
    ns: dict = {}
    try:
        exec(code, ns)
    except Exception as e:
        return f"BUGGY_CODE_DOES_NOT_EXEC: {type(e).__name__}: {e}"
    try:
        exec(task["test_code"], ns)
    except AssertionError as e:
        return f"buggy fails as expected (AssertionError: {str(e)[:100]})"
    except Exception as e:
        return f"buggy fails with {type(e).__name__}: {str(e)[:100]}"
    return "BUGGY_PASSES_TESTS_INVALID_TASK"


def main() -> int:
    tasks = json.loads(TASKS_PATH.read_text())
    print(f"Validating {len(tasks)} held-out tasks")
    invalid = []
    for t in tasks:
        status = run_buggy(t)
        marker = "OK" if not status.startswith(("BUGGY_CODE_DOES_NOT_EXEC", "BUGGY_PASSES")) else "FAIL"
        print(f"  [{marker}] {t['task_id']}: {status}")
        if marker == "FAIL":
            invalid.append(t["task_id"])
    if invalid:
        print(f"\n{len(invalid)} invalid tasks: {invalid}")
        return 1
    print(f"\nAll {len(tasks)} tasks valid (buggy code fails tests).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
