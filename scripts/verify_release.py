"""Verify the public release payload without downloading models or data."""
from pathlib import Path
import csv
import hashlib
import json
import math

def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root/'repro/release_manifest.json').read_text(encoding='utf-8'))
    for relative, digest in manifest['files'].items():
        assert hashlib.sha256((root/relative).read_bytes()).hexdigest() == digest, relative
    with (root/'results/dev_metrics.csv').open(encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    keys = [k for k in rows[0] if k not in ('configuration','mean_gain')]
    assert len(rows) == 8 and len(keys) == 8
    for row in rows:
        gain = sum(float(row[k])-float(rows[0][k]) for k in keys)/8
        assert math.isclose(gain,float(row['mean_gain']),abs_tol=1e-12)
    selection = json.loads((root/'results/selection_summary.json').read_text())
    assert selection['selected'] is None and selection['status'] == 'no_admissible_model'
    print(f"Release verification passed: {len(manifest['files'])} files, 8 configurations, 8 task-language metrics")

if __name__ == '__main__':
    main()
