"""Persistent wall-clock budget across sequential experiment commands, fail closed after crashes."""
import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from src.common import write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ledger", default="runs/budget.json")
    p.add_argument("--hours", type=float, default=24)
    p.add_argument("command", nargs=argparse.REMAINDER)
    a = p.parse_args()
    command = a.command[1:] if a.command and a.command[0] == "--" else a.command
    if not command or a.hours <= 0:
        p.error("Specify a positive budget and a command after --")
    path = Path(a.ledger)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(".lock")
    try:
        handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise SystemExit("Budget lock exists: another job is active or a previous wrapper crashed. Inspect before recovery.")
    os.close(handle)
    try:
        ledger = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"spent_seconds": 0, "limit_seconds": a.hours * 3600, "jobs": []}
        if ledger.get("active"):
            raise RuntimeError("Unsettled job in ledger; audit its elapsed time before restarting")
        ledger["limit_seconds"] = min(ledger["limit_seconds"], a.hours * 3600)
        remaining = ledger["limit_seconds"] - ledger["spent_seconds"]
        if remaining <= 0:
            raise SystemExit("Experiment time budget exhausted")
        ledger["active"] = {"command": command, "started_unix": time.time()}
        write_json(path, ledger)
        start = time.monotonic()
        status = -1
        try:
            process = subprocess.Popen(command, start_new_session=os.name != "nt")
            try:
                status = process.wait(timeout=remaining)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                # Terminate the complete experiment process tree, including train children.
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True)
                else:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                process.wait()
                status = 124
        finally:
            elapsed = time.monotonic() - start
            ledger["spent_seconds"] += elapsed
            ledger["jobs"].append({**ledger.pop("active"), "seconds": elapsed, "exit_code": status})
            write_json(path, ledger)
        raise SystemExit(status)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
