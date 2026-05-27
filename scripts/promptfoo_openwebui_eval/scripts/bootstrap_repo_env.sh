#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${DEA_REPO_ROOT:-/workspace/document_embedding_analysis}"
VENV_DIR="${PROMPTFOO_VENV_DIR:-/workspace/.venv}"
BUNDLE_ROOT="/workspace/promptfoo_openwebui_dea"

if [[ ! -d "$REPO_ROOT" ]]; then
  echo "DEA_REPO_ROOT does not exist: $REPO_ROOT" >&2
  exit 1
fi

python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip "setuptools<82" wheel
python -m pip install -r "$BUNDLE_ROOT/requirements.txt"

if [[ -f "$REPO_ROOT/pyproject.toml" ]]; then
  python -m pip install -e "$REPO_ROOT[comparison]" || python -m pip install -e "$REPO_ROOT"
elif [[ -f "$REPO_ROOT/requirements.txt" ]]; then
  python -m pip install -r "$REPO_ROOT/requirements.txt"
  python -m pip install -e "$REPO_ROOT" || true
else
  echo "Could not find pyproject.toml or requirements.txt under $REPO_ROOT" >&2
  exit 1
fi

echo "Bootstrap complete"
echo "PROMPTFOO_PYTHON=$VENV_DIR/bin/python"
