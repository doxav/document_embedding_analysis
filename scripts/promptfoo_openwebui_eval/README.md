# Promptfoo OpenWebUI/Ollama Evaluation Bundle

This directory contains a local evaluation harness for `document_embedding_analysis`.
It turns the repository's generated dataset outputs into Promptfoo test cases, runs
candidate generation through a local bridge, and scores answers with either:

- repository-native DEA metrics and the DEA qualitative judge;
- Promptfoo model-graded assertions served by local Ollama.

The default path is fully local: Docker runs Promptfoo and the bridge, Ollama serves
`qwen-laptop:latest` on the host, and the OpenAI-compatible API points to
`http://127.0.0.1:11434/v1`. Do not use the public OpenAI API for this setup.

## Table Of Contents

- [Main Concepts](#main-concepts)
- [Directory Layout](#directory-layout)
- [Generated CSV Datasets](#generated-csv-datasets)
- [CSV Versus JSON Or YAML](#csv-versus-json-or-yaml)
- [Evaluation Steps](#evaluation-steps)
- [OpenWebUI Prompt And Valve Parameters](#openwebui-prompt-and-valve-parameters)
- [Setup](#setup)
- [Generate Local CSVs](#generate-local-csvs)
- [Run Evaluations](#run-evaluations)
- [Troubleshooting](#troubleshooting)
- [Git Policy](#git-policy)

## Main Concepts

**Promptfoo** is the evaluation runner. It reads one YAML config per dataset,
step, and scoring strategy. Each YAML points to a generated CSV file containing
test-case variables.

**DEA**, or Document Embedding Analysis, is the repository's native document
scoring path. The `*.dea.yaml` configs call `common.doc_eval.evaluate_document`
through `assertions/dea_metrics.py`. These configs keep `skip_dea=False` and
enable `useDeaJudge=true`.

**DEA qualitative judge** is the optional LLM part of DEA evaluation. In this
bundle it uses local Ollama through the OpenAI-compatible client. The configured
judge model is `qwen-laptop-dea:latest`, a local Ollama alias based on
`qwen-laptop:latest` but configured to return strict JSON.

**LLM judge** means Promptfoo's model-graded assertions such as
`llm-rubric`, `factuality`, and `context-faithfulness`. The `*.llm_judge.yaml`
configs also point to local Ollama, not the public OpenAI API.

**OpenWebUI bridge** is `api/openwebui_bridge.py`, a small FastAPI service that
receives Promptfoo rows and calls either OpenWebUI or Ollama. The default
`GENERATION_BACKEND=ollama` calls Ollama directly. `GENERATION_BACKEND=openwebui`
uses OpenWebUI upload and chat endpoints.

**MQS evaluation dataset** means the `MQS_evaluation_dataset.jsonl` files
created by the repository importers under `output/bigsurvey` and
`output/multilexsum`. The CSV builder normalizes their paths for Promptfoo.

## Directory Layout

| Path | Purpose | Commit? |
|---|---|---|
| `*.step*.dea.yaml` | Promptfoo configs for DEA scoring | yes |
| `*.step*.llm_judge.yaml` | Promptfoo configs for local model-graded scoring | yes |
| `api/openwebui_bridge.py` | FastAPI bridge for Step 2 and Step 3 generation | yes |
| `assertions/dea_metrics.py` | Promptfoo Python assertion wrapping `evaluate_document` | yes |
| `lib/*.py` | Shared CSV, path, OpenWebUI, and Ollama helpers | yes |
| `scripts/*.py` | Bootstrap, CSV generation, and answer generation scripts | yes |
| `ollama/Modelfile.qwen-laptop-dea` | Local JSON-judge Ollama model configuration | yes |
| `datasets/` | Generated Promptfoo CSV fixtures | yes |
| `results/` | Generated Promptfoo result JSONs | no |
| `.env` | Local paths and secrets | no |
| `.env.example` | Safe environment template | yes |

## Generated CSV Datasets

The files under `datasets/<dataset>/` are generated Promptfoo fixtures. They are
committed so a new checkout can run the documented evaluations immediately, but
they should be regenerated when dataset importers, row limits, or CSV schema
change.

Promptfoo uses CSV rows as test cases. Each CSV row becomes one evaluation case.
Every column is available in YAML templates as `{{column_name}}`.

The generated CSV names are:

| File | Meaning |
|---|---|
| `step1.csv` | Offline evaluation rows with a simple baseline `candidate_answer` already filled |
| `step2_input.csv` | Generation input rows with empty `candidate_answer` |
| `step2_output.csv` | Generated answers from Step 2, then evaluated offline |
| `step3.csv` | Live Promptfoo generation rows; Promptfoo calls the bridge during evaluation |

Important columns:

| Column | Meaning |
|---|---|
| `task_id`, `dataset`, `task_type` | Identifies the test case and dataset family |
| `title`, `abstract`, `instruction`, `request_prompt` | Inputs used for generation |
| `gold_summary` | Reference answer used by judge-style assertions and article metrics |
| `dea_solution_path` | Repository-relative path to the DEA JSON solution |
| `source_paths_json` | JSON-encoded list of repository-relative source files |
| `kb_ids_json` | JSON-encoded OpenWebUI knowledge-base ids/names for `<kb_list>` |
| `source_document_mode` | Whether source files are provided or only open-retrieval references exist |
| `openwebui_pipe_model` | OpenWebUI model or pipe model id used by Step 2/3 generation |
| `tool_parameters_json` | Extra JSON object copied into `<tool_parameters>` |
| `summarizer_model_id`, `algorithm`, `target_length`, `structure` | Common pipe/tool parameters copied into `<tool_parameters>` |
| `generation_temperature`, `generation_top_p`, `generation_max_tokens` | HTTP generation options sent to OpenWebUI or Ollama |
| `candidate_answer` | Answer being evaluated; present in Step 1 and Step 2 output |
| `min_chars`, `max_chars` | Dynamic length guard derived from the reference |

The usual local row counts from the current setup are:

| Dataset | Mode | Limit |
|---|---|---:|
| BigSurvey | source-provided MDS | 20 |
| MultiLexSum | source-provided MDS | 20 |
| latex | native DEA/open retrieval | 10 |
| arxiv | native DEA/open retrieval | 10 |
| patent | native DEA/open retrieval | 24 |

## CSV Versus JSON Or YAML

CSV is used because Promptfoo can load `tests: file://...csv` directly and each
column becomes a template variable in YAML without writing a custom provider.

It is not the cleanest representation for this data. Some fields contain long
text, and nested values such as source paths are stored as JSON strings inside
CSV cells. JSONL would be more natural for nested test records, and YAML would
be more readable for hand-authored cases. For this bundle, CSV is a pragmatic
Promptfoo interop format, while the canonical dataset outputs remain the
repository's JSON/JSONL files under `output/`.

## Evaluation Steps

| Step | What happens | Promptfoo provider | CSV file |
|---|---|---|---|
| Step 1 | Evaluate an existing baseline answer | `echo` | `step1.csv` |
| Step 2 | Generate answers first, then evaluate them offline | generation script, then `echo` | `step2_input.csv` -> `step2_output.csv` |
| Step 3 | Promptfoo performs live generation and scoring in one run | HTTP bridge | `step3.csv` |

Scoring strategies:

| Strategy | YAML suffix | What it checks |
|---|---|---|
| DEA | `.dea.yaml` | Native DEA plan/content/resource scores, ROUGE/entity metrics, and DEA judge |
| LLM judge | `.llm_judge.yaml` | Promptfoo's model-graded assertions using local Ollama |

## OpenWebUI Prompt And Valve Parameters

The bridge sends one user message to OpenWebUI. It does not add a separate
system message. The user message is built by `build_openwebui_user_prompt` from
CSV row values and has this shape:

```text
<request_prompt text>

<tool_parameters>
  <algorithm>kohaku</algorithm>
  <target_length>long</target_length>
  <structure>sectioned</structure>
  <emit_diagnostics>false</emit_diagnostics>
</tool_parameters>

<files_list>
["openwebui-file-id-1", "openwebui-file-id-2"]
</files_list>

<kb_list>
["openwebui-knowledge-base-id"]
</kb_list>
```

The OpenWebUI thematic-summary pipe is not required by this bundle, but the
bridge is compatible with its documented control tags:

| Tag block | Source in this bundle | Purpose |
|---|---|---|
| `<tool_parameters>` | `tool_parameters_json` plus typed columns | Per-call pipe/tool parameters |
| `<files_list>` | Uploaded ids resolved from `source_paths_json` | OpenWebUI file ids passed to the pipe/tool |
| `<kb_list>` | `kb_ids_json` | OpenWebUI knowledge-base ids/names passed as vector collection candidates |

Typed columns override same-named keys from `tool_parameters_json`:

| CSV column / env fallback | Sent parameter |
|---|---|
| `summarizer_model_id` / `OPENWEBUI_DEFAULT_SUMMARIZER_MODEL_ID` | `<summarizer_model_id>` |
| `algorithm` / `OPENWEBUI_DEFAULT_ALGORITHM` | `<algorithm>` |
| `target_length` / `OPENWEBUI_DEFAULT_TARGET_LENGTH` | `<target_length>` |
| `structure` / `OPENWEBUI_DEFAULT_STRUCTURE` | `<structure>` |

Use `tool_parameters_json` for less common parameters accepted by the selected
pipe/tool, for example:

```json
{"emit_diagnostics": false, "summary_structure": "sectioned literature review"}
```

Use `kb_ids_json` for knowledge bases:

```json
["kb-lit-review", {"id": "kb-cases", "name": "Case law KB"}]
```

When `GENERATION_BACKEND=openwebui`, local files from `source_paths_json` are
uploaded to OpenWebUI and their resulting file ids are inserted into
`<files_list>`. When `GENERATION_BACKEND=ollama`, the bridge bypasses OpenWebUI
and appends trimmed source text directly to the Ollama prompt instead.

## Setup

Run commands from this directory unless stated otherwise:

```bash
cd scripts/promptfoo_openwebui_eval
```

Create local environment settings:

```bash
cp .env.example .env
```

Required values:

```bash
DEA_REPO_PATH=/absolute/path/to/document_embedding_analysis
DEA_REPO_ROOT=/workspace/document_embedding_analysis
GENERATION_BACKEND=ollama
OLLAMA_BASE_URL=http://127.0.0.1:11434
OPENAI_BASE_URL=http://127.0.0.1:11434/v1
OPENAI_API_KEY=ollama-local
OPENAI_MODEL=qwen-laptop-dea:latest
DEA_USE_JUDGE=true
DEA_JUDGE_MODEL=qwen-laptop-dea:latest
```

This Compose file uses Linux `network_mode: host`, because local Ollama is
normally bound to `127.0.0.1:11434`. With host networking, container
`127.0.0.1` is the host loopback.

Create the local JSON-judge Ollama alias:

```bash
ollama create qwen-laptop-dea:latest -f ./ollama/Modelfile.qwen-laptop-dea
```

Build and bootstrap the container:

```bash
DOCKER_BUILDKIT=0 docker build -t promptfoo-openwebui-dea:latest .
docker compose run --rm promptfoo bash -lc "./scripts/bootstrap_repo_env.sh"
```

Start the bridge:

```bash
docker compose up -d owui-bridge
curl http://127.0.0.1:8001/healthz
```

Expected health shape:

```json
{"ok":true,"backend":"ollama","base_url":"http://127.0.0.1:8080","ollama_base_url":"http://127.0.0.1:11434"}
```

If your shell does not yet have the Docker group activated, wrap Docker commands:

```bash
newgrp docker <<'EOF'
docker compose ps
EOF
```

## Generate Local CSVs

The YAML configs expect CSVs under `datasets/<dataset>/`. Generate them from the
repository outputs before running Promptfoo.

Helper function:

```bash
build_three() {
  mode="$1"
  dataset="$2"
  limit="$3"
  python ./scripts/build_promptfoo_csvs.py --mode "$mode" --dataset "$dataset" --stage step1 --limit "$limit" --output "./datasets/$dataset/step1.csv"
  python ./scripts/build_promptfoo_csvs.py --mode "$mode" --dataset "$dataset" --stage step2-input --limit "$limit" --output "./datasets/$dataset/step2_input.csv"
  python ./scripts/build_promptfoo_csvs.py --mode "$mode" --dataset "$dataset" --stage step3 --limit "$limit" --output "./datasets/$dataset/step3.csv"
}
```

Generate the standard local set:

```bash
docker compose run --rm promptfoo bash -lc '
  source /workspace/.venv/bin/activate
  build_three() {
    mode="$1"; dataset="$2"; limit="$3"
    python ./scripts/build_promptfoo_csvs.py --mode "$mode" --dataset "$dataset" --stage step1 --limit "$limit" --output "./datasets/$dataset/step1.csv"
    python ./scripts/build_promptfoo_csvs.py --mode "$mode" --dataset "$dataset" --stage step2-input --limit "$limit" --output "./datasets/$dataset/step2_input.csv"
    python ./scripts/build_promptfoo_csvs.py --mode "$mode" --dataset "$dataset" --stage step3 --limit "$limit" --output "./datasets/$dataset/step3.csv"
  }
  build_three mds bigsurvey 20
  build_three mds multilexsum 20
  build_three native latex 10
  build_three native arxiv 10
  build_three native patent 24
'
```

Parameter flags can be added to any `build_promptfoo_csvs.py` call:

```bash
--algorithm kohaku
--target-length long
--structure "sectioned literature review"
--tool-parameters-json '{"emit_diagnostics": false}'
--kb-id kb-lit-review
```

Those values are written into the generated CSV rows and later become
`<tool_parameters>` and `<kb_list>` blocks in the OpenWebUI user message.

Create Step 2 generated answers for one deterministic row:

```bash
docker compose run --rm promptfoo bash -lc '
  source /workspace/.venv/bin/activate
  python ./scripts/generate_openwebui_csv.py \
    --input ./datasets/bigsurvey/step2_input.csv \
    --output ./datasets/bigsurvey/step2_output.csv \
    --bridge-url http://127.0.0.1:8001/generate \
    --limit 1 --overwrite
'
```

## Run Evaluations

One-row DEA check:

```bash
docker compose run --rm promptfoo bash -lc \
  "promptfoo eval -c ./arxiv.step1.dea.yaml --filter-first-n 1 --no-progress-bar"
```

One-row LLM-judge check:

```bash
docker compose run --rm promptfoo bash -lc \
  "promptfoo eval -c ./arxiv.step1.llm_judge.yaml --filter-first-n 1 --no-progress-bar"
```

Full Step 1 DEA for a dataset:

```bash
docker compose run --rm promptfoo bash -lc \
  "promptfoo eval -c ./bigsurvey.step1.dea.yaml --no-progress-bar"
```

Step 2 offline evaluation after generation:

```bash
docker compose run --rm promptfoo bash -lc \
  "promptfoo eval -c ./bigsurvey.step2.dea.yaml --no-progress-bar"
```

Step 3 live evaluation through the bridge:

```bash
docker compose run --rm promptfoo bash -lc \
  "promptfoo eval -c ./bigsurvey.step3.dea.yaml --filter-first-n 1 --no-progress-bar"
```

## Troubleshooting

| Symptom | Check |
|---|---|
| Docker permission denied | Run the command inside `newgrp docker` |
| Promptfoo says no tests loaded | Generate the required `datasets/<dataset>/<step>.csv` file |
| DEA assertion cannot find files | Check `DEA_REPO_PATH`, `DEA_REPO_ROOT`, and `dea_solution_path` in the CSV |
| OpenAI API connection error | Confirm `OPENAI_BASE_URL=http://127.0.0.1:11434/v1` inside the container |
| LLM judge returns invalid JSON | Confirm `DEA_JUDGE_MODEL=qwen-laptop-dea:latest` and recreate the Ollama alias |
| Step 3 cannot reach bridge | Use `http://127.0.0.1:8001/generate` with host networking |
| OpenWebUI path fails | Use `GENERATION_BACKEND=ollama` first; OpenWebUI also needs a valid API key and pipe model |

Run repository tests from the repo root:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate humanllm
PYTHONPATH=. pytest -q
```

## Git Policy

Add reusable source/configuration files:

- `README.md`, `.env.example`, `Dockerfile`, `docker-compose.yml`
- `*.yaml`
- `api/`, `assertions/`, `lib/`, `scripts/`, `ollama/`
- `datasets/`
- `requirements.txt`

Do not add local/generated files:

- `.env`
- `.cache/`
- `results/`
- `temp/`
- `__pycache__/` and `*.pyc`

The generated CSVs are committed as deterministic fixtures for the documented
20/20/10/10/24 local setup. Regenerate them when the dataset importers, prompt
schema, or row limits change.
