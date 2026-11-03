#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -c "import torch; from packaging.version import Version; assert Version(torch.__version__.split('+')[0]) >= Version('2.6'); assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))"
python -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
mkdir -p artifacts
python -m pip freeze > artifacts/environment.lock.txt
echo 'Environment ready. Continue with the README reproduction commands.'
