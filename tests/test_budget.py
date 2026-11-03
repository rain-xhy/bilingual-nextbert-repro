import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class BudgetTests(unittest.TestCase):
    def test_timeout_is_charged_and_exhausted_ledger_cannot_restart(self):
        with tempfile.TemporaryDirectory() as folder:
            ledger = Path(folder) / "budget.json"
            command = [sys.executable, "-m", "scripts.budget_run", "--ledger", str(ledger), "--hours", ".0002", "--",
                       sys.executable, "-c", "import time; time.sleep(10)"]
            result = subprocess.run(command, capture_output=True, timeout=20)
            self.assertEqual(result.returncode, 124, result.stderr.decode(errors="replace"))
            data = json.loads(ledger.read_text(encoding="utf-8"))
            self.assertGreaterEqual(data["spent_seconds"], data["limit_seconds"])
            self.assertNotIn("active", data)
            self.assertFalse(ledger.with_suffix(".lock").exists())
            result = subprocess.run(command, capture_output=True, timeout=20)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(len(json.loads(ledger.read_text(encoding="utf-8"))["jobs"]), 1)
