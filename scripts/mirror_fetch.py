"""Public HF mirror downloads: IPv4, bounded retries and live byte progress."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from urllib.parse import urlsplit


def endpoint_url(url):
    base = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    parsed = urlsplit(base)
    if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
        raise ValueError("HF_ENDPOINT must be an HTTPS endpoint without credentials")
    original = urlsplit(url)
    if original.scheme != "https" or original.netloc != "huggingface.co":
        raise ValueError("Expected a public https://huggingface.co source URL")
    return base + original.path + ("?" + original.query if original.query else "")


def total_bytes(headers):
    if not headers.exists():
        return None
    blocks = headers.read_text(encoding="latin-1").replace("\r\n", "\n").split("\n\n")
    for block in reversed(blocks):
        lines = block.strip().splitlines()
        if not lines or not lines[0].startswith("HTTP/"):
            continue
        parts = lines[0].split()
        if len(parts) < 2 or parts[1] != "200":
            return None
        for line in lines[1:]:
            key, _, value = line.partition(":")
            if key.lower() == "content-length":
                try:
                    return int(value.strip())
                except ValueError:
                    return None
        return None
    return None


def transfer(url, target, api=False, interval=5):
    curl = shutil.which("curl")
    if not curl:
        raise RuntimeError("curl is required: apt-get update && apt-get install -y curl")
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    remote = endpoint_url(url)
    errors = []
    for attempt in range(1, 4):
        print(f"[NET] attempt {attempt}/3 GET {remote}", flush=True)
        with target.open("wb"):
            pass
        with tempfile.TemporaryDirectory(prefix="hf_headers_") as directory:
            headers = Path(directory) / "headers"
            stderr = Path(directory) / "curl.stderr"
            command = [curl, "--ipv4", "--fail", "--location", "--silent", "--show-error",
                       "--connect-timeout", "15", "--speed-limit", "1024", "--speed-time", "60",
                       "--max-time", "60" if api else "0", "--proto", "=https",
                       "--proto-redir", "=https", "--dump-header", str(headers),
                       "--output", str(target), remote]
            start, previous, last = time.monotonic(), 0, time.monotonic()
            with stderr.open("wb") as err:
                process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=err)
                try:
                    while True:
                        try:
                            code = process.wait(timeout=interval)
                            break
                        except subprocess.TimeoutExpired:
                            now = time.monotonic()
                            size = target.stat().st_size if target.exists() else 0
                            total = total_bytes(headers)
                            fraction = f" / {total / 2**20:.2f} MiB ({100*size/total:.1f}%)" if total and size <= total else " (total unknown)"
                            speed = max(0, size - previous) / max(now - last, .001) / 2**20
                            print(f"[DOWNLOAD] {target.name}: {size/2**20:.2f} MiB{fraction}; {speed:.2f} MiB/s; elapsed={now-start:.0f}s", flush=True)
                            previous, last = size, now
                except BaseException:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise
            if code == 0:
                size = target.stat().st_size
                print(f"[NET DONE] {target.name}: {size/2**20:.2f} MiB; elapsed={time.monotonic()-start:.1f}s", flush=True)
                return
            reason = stderr.read_text(encoding="utf-8", errors="replace").strip()
            errors.append(f"curl exit={code}: {reason}")
            print(f"[NET RETRY] {errors[-1]}", flush=True)
        if attempt < 3:
            time.sleep(2)
    raise RuntimeError("Mirror download failed after 3 attempts: " + " | ".join(errors))


def fetch_json(url):
    with tempfile.TemporaryDirectory(prefix="hf_api_") as directory:
        target = Path(directory) / "response.json"
        transfer(url, target, api=True)
        return json.loads(target.read_text(encoding="utf-8"))


def download_file(url, target):
    target = Path(target)
    if target.exists():
        print(f"[CACHE] {target}: {target.stat().st_size/2**20:.2f} MiB", flush=True)
        return
    temporary = target.with_suffix(target.suffix + ".partial")
    transfer(url, temporary)
    temporary.replace(target)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true", required=True)
    p.parse_args()
    result = fetch_json("https://huggingface.co/api/datasets/Shitao/MLDR")
    if not isinstance(result.get("sha"), str) or len(result["sha"]) != 40:
        raise ValueError("Mirror API did not return a valid dataset revision")
    print(f"[MIRROR OK] Shitao/MLDR revision={result['sha']}", flush=True)
