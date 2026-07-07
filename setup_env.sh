#!/usr/bin/env bash
# Prerequisites:
#   - Python 3.8+
#   - Java 17+ on PATH
#   - Run on Linux or WSL
#
# Usage:
#   bash setup_env.sh
# Afterwards you can activate the venv:
#   source .venv/bin/activate

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

VENV=".venv"
UPGRADE_FLAG=""
[[ "${1:-}" == "--upgrade" ]] && UPGRADE_FLAG="--upgrade"

[[ -d "$VENV" ]] || python3 -m venv "$VENV"
source "$VENV/bin/activate"

pip install --quiet --upgrade pip
pip install $UPGRADE_FLAG -r requirements.txt
pip install $UPGRADE_FLAG --no-deps -r requirements-autosklearn.txt

echo "Done"
