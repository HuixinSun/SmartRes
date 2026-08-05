#!/usr/bin/env bash
# Scripted form of README "Installation". Usage: bash env/setup.sh [env_name]
set -euo pipefail

ENV_NAME="${1:-smartres}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

conda create -n "$ENV_NAME" python=3.10 -y
eval "$(conda shell.bash hook)"; conda activate "$ENV_NAME"

pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r env/requirements.txt
pip install trl==0.9.6 --no-deps
pip install -e .

git submodule update --init
pip install -e third_party/LLaMA-Factory --no-deps

python -c "import smartres, llamafactory; print('smartres', smartres.__version__)"
echo "done -- conda activate $ENV_NAME"
