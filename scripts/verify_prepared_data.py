"""Compare regenerated JSONL files against the recorded data hashes without shipping text."""
from pathlib import Path
import hashlib
import json

def main():
    verified = 0
    for dataset in ('mldr', 'multitask', 'evidence'):
        expected = json.loads(Path(f'repro/manifests/{dataset}.json').read_text(encoding='utf-8'))
        files = expected['outputs'] if dataset == 'mldr' else expected['files']
        for name, digest in files.items():
            path = Path('data') / dataset / name
            if not path.exists():
                raise SystemExit(f'Missing prepared file: {path}')
            observed = hashlib.sha256(path.read_bytes()).hexdigest()
            if observed != digest:
                raise SystemExit(f'Hash mismatch: {path}. Preserve this dataset and inspect library versions and sampling before training.')
            verified += 1
    print(f'Prepared-data verification passed: {verified} files')

if __name__ == '__main__':
    main()
