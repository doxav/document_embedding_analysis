#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="${PROMPTFOO_ENV_FILE:-$BUNDLE_ROOT/.env}"
MODE="${1:---full}"

if [[ "$MODE" != "--full" && "$MODE" != "--runtime" ]]; then
  echo "Usage: $0 [--full|--runtime]" >&2
  exit 2
fi

env_value() {
  python3 - "$ENV_FILE" "$1" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
key = sys.argv[2]
if path.exists():
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#") and "=" in line:
            current, value = line.split("=", 1)
            if current.strip() == key:
                print(value.strip().strip('"').strip("'"))
                break
PY
}

default_repo_root="$(cd "$BUNDLE_ROOT/../.." && pwd)"
REPO_ROOT="${DEA_REPO_ROOT:-$(env_value DEA_REPO_ROOT)}"
if [[ -z "$REPO_ROOT" || ! -d "$REPO_ROOT" ]]; then
  configured_repo_path="$(env_value DEA_REPO_PATH)"
  REPO_ROOT="${configured_repo_path:-$default_repo_root}"
fi
if [[ ! -d "$REPO_ROOT" ]]; then
  echo "Document Embedding Analysis repository does not exist: $REPO_ROOT" >&2
  exit 1
fi
if [[ ! -f "$BUNDLE_ROOT/requirements.txt" || ! -f "$REPO_ROOT/pyproject.toml" ]]; then
  echo "The bundle requirements or repository pyproject.toml is missing." >&2
  exit 1
fi

if [[ "$BUNDLE_ROOT" == /workspace/* ]]; then
  default_venv="/workspace/.venv"
else
  default_venv="$BUNDLE_ROOT/.venv"
fi
VENV_DIR="${PROMPTFOO_VENV_DIR:-$default_venv}"

for command_name in python3 curl; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Required command is missing: $command_name" >&2
    exit 1
  fi
done

python3 -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --disable-pip-version-check --quiet --upgrade pip "setuptools<82" wheel
python -m pip install --disable-pip-version-check --quiet -r "$BUNDLE_ROOT/requirements.txt"
if [[ -f "$REPO_ROOT/requirements.txt" ]]; then
  python -m pip install --disable-pip-version-check --quiet -r "$REPO_ROOT/requirements.txt"
fi
export PYTHONPATH="$REPO_ROOT:$BUNDLE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
python - <<'PY'
import fastapi
import requests
import yaml
from common import doc_eval

print(f"Python dependencies OK (FastAPI {fastapi.__version__}, requests {requests.__version__}, PyYAML {yaml.__version__}).")
print(f"DEA import OK ({doc_eval.__file__}).")
PY

if command -v promptfoo >/dev/null 2>&1; then
  promptfoo --version
elif [[ "$MODE" == "--runtime" ]]; then
  echo "promptfoo is missing from the runtime image." >&2
  exit 1
elif ! command -v docker >/dev/null 2>&1; then
  echo "Neither promptfoo nor Docker is installed; Promptfoo cannot be run." >&2
  exit 1
else
  echo "Host promptfoo is absent; the checked Docker promptfoo service will be used."
fi

configured_base_url="${OPENWEBUI_BASE_URL:-$(env_value OPENWEBUI_BASE_URL)}"
base_url_was_missing=false
normalized_base_url="${configured_base_url,,}"
if [[ -z "$configured_base_url" || "$normalized_base_url" == "none" || "$normalized_base_url" == "null" ]]; then
  echo "OPENWEBUI_BASE_URL was not provided in $ENV_FILE; selecting the managed local endpoint http://127.0.0.1:8080."
  configured_base_url="http://127.0.0.1:8080"
  base_url_was_missing=true
else
  echo "OPENWEBUI_BASE_URL was provided in $ENV_FILE: $configured_base_url"
fi

openwebui_is_ready() {
  curl --silent --show-error --fail --max-time 5 "$configured_base_url/health" >/dev/null 2>&1
}

if [[ "$MODE" == "--runtime" ]]; then
  if ! openwebui_is_ready; then
    echo "Configured OpenWebUI is not reachable at $configured_base_url." >&2
    exit 1
  fi
  echo "Runtime bootstrap complete; OpenWebUI is reachable at $configured_base_url."
  echo "PROMPTFOO_PYTHON=$VENV_DIR/bin/python"
  exit 0
fi

if ! command -v docker >/dev/null 2>&1 || ! docker compose version >/dev/null 2>&1; then
  echo "Docker with Compose v2 is required for the full bootstrap." >&2
  exit 1
fi

if [[ "$base_url_was_missing" == true ]]; then
  python "$SCRIPT_DIR/setup_openwebui.py" \
    --env-file "$ENV_FILE" \
    --local-env-file "$BUNDLE_ROOT/openwebui_docker/.env" \
    --repo-root "$REPO_ROOT" \
    --prepare-local-env
fi

managed_setting="${OPENWEBUI_MANAGED:-$(env_value OPENWEBUI_MANAGED)}"
if ! openwebui_is_ready; then
  if [[ "$base_url_was_missing" == true || "$managed_setting" == "true" ]]; then
    if [[ "$configured_base_url" != "http://127.0.0.1:8080" && "$configured_base_url" != "http://localhost:8080" ]]; then
      echo "Managed OpenWebUI can only be started for the local :8080 endpoint, got $configured_base_url." >&2
      exit 1
    fi
    if [[ "$base_url_was_missing" != true ]]; then
      python "$SCRIPT_DIR/setup_openwebui.py" \
        --env-file "$ENV_FILE" \
        --local-env-file "$BUNDLE_ROOT/openwebui_docker/.env" \
        --repo-root "$REPO_ROOT" \
        --prepare-local-env
    fi
    docker compose \
      --project-directory "$BUNDLE_ROOT/openwebui_docker" \
      --env-file "$BUNDLE_ROOT/openwebui_docker/.env" \
      -f "$BUNDLE_ROOT/openwebui_docker/docker-compose.yml" \
      up -d --wait
  else
    echo "Configured OpenWebUI is not reachable at $configured_base_url." >&2
    echo "No container was created because an endpoint was supplied and OPENWEBUI_MANAGED is not true." >&2
    exit 1
  fi
else
  echo "Existing OpenWebUI is reachable; no OpenWebUI container was created."
fi

python "$SCRIPT_DIR/setup_openwebui.py" \
  --env-file "$ENV_FILE" \
  --local-env-file "$BUNDLE_ROOT/openwebui_docker/.env" \
  --repo-root "$REPO_ROOT" \
  --configure --test-inference --test-tool

docker compose -f "$BUNDLE_ROOT/docker-compose.yml" build owui-bridge
docker compose -f "$BUNDLE_ROOT/docker-compose.yml" up -d owui-bridge

bridge_port="$(env_value OWUI_BRIDGE_PORT)"
bridge_url="http://127.0.0.1:${bridge_port:-8001}"
for _attempt in $(seq 1 120); do
  if curl --silent --show-error --fail --max-time 5 "$bridge_url/healthz" >/dev/null 2>&1; then
    break
  fi
  sleep 2
done
if ! curl --silent --show-error --fail --max-time 5 "$bridge_url/healthz" >/dev/null; then
  echo "OpenWebUI bridge did not become healthy at $bridge_url." >&2
  exit 1
fi

bridge_model="$(env_value OWI_DEFAULT_MODEL_ID)"
bridge_marker="OWI_BRIDGE_OK_$(date +%s)"
bridge_response=""
bridge_ok=false
for _attempt in 1 2 3; do
  bridge_response="$(curl --silent --show-error --fail --max-time 600 \
    -H 'Content-Type: application/json' \
    -d "{\"openwebui_pipe_model\":\"$bridge_model\",\"openwebui_tool_ids_json\":\"[]\",\"request_prompt\":\"Reply with exactly $bridge_marker\",\"generation_temperature\":\"0\",\"generation_max_tokens\":\"32\",\"openwebui_model_params_json\":\"{\\\"model_extra_payload_json\\\":{\\\"reasoning\\\":{\\\"enabled\\\":false,\\\"effort\\\":\\\"low\\\"}}}\"}" \
    "$bridge_url/generate")"
  if python - "$bridge_marker" "$bridge_response" <<'PY'
import json
import sys

marker = sys.argv[1]
payload = json.loads(sys.argv[2])
if marker not in str(payload.get("output") or ""):
    raise SystemExit(1)
PY
  then
    echo "OpenWebUI bridge inference OK."
    bridge_ok=true
    break
  fi
done
if [[ "$bridge_ok" != true ]]; then
  echo "Bridge response did not contain the expected marker after 3 attempts." >&2
  exit 1
fi

echo "Full bootstrap complete."
echo "PROMPTFOO_PYTHON=$VENV_DIR/bin/python"
