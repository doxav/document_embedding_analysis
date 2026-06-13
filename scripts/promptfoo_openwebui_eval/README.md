# Promptfoo OpenWebUI/OpenAI Endpoint Evaluation Bundle

This directory contains a local evaluation harness of summarization tools using `document_embedding_analysis` and PromptFoo (we test it on OpenWebUI model/pipe with a specific tool).
It turns the repository's generated dataset outputs into Promptfoo test cases, runs
candidate generation through a local bridge, and scores answers with either:

- repository-native DEA metrics and the DEA qualitative judge;
- Promptfoo model-graded assertions served by any OpenAI-compatible endpoint.

For the summarization task being evaluated, the bridge should call OpenWebUI so
the selected summarization pipe/tool is exercised. Direct OpenAI-compatible
endpoint calls are for judges, embeddings, or non-tool baselines.
The default generation pipe is `summarizer---kohaku-OR`.

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
enable `useDeaJudge=true`. The assertion exposes Promptfoo named scores for
plan similarity, content similarity, resource/bibliography similarity,
ROUGE-L, entity recall, citation count, length alignment, the composite DEA
score, and raw native DEA metrics under `dea_*` names.

For source-provided MDS datasets such as BigSurvey and MultiLexSum, the
imported `dea_solution.json` may contain a text-only `plan`/`resources` target
and a `target_file_path` pointing to `full_text.md`. During evaluation the
native DEA path resolves solution-local target paths and creates a temporary
embedded DEA target JSON when the saved target lacks DEA embeddings. This keeps
generated datasets lightweight while still allowing plan/content/resource and
length-ratio scoring.

**DEA qualitative judge** is the optional LLM part of DEA evaluation. It uses
the OpenAI-compatible client, so the same config shape works with local
OpenAI-compatible endpoints, vLLM, OpenRouter, or OpenAI. Set
`OPENAI_BASE_URL`, `OPENAI_API_KEY`, and
`DEA_JUDGE_MODEL`. Some OpenAI-compatible providers need extra request fields;
set `DEA_JUDGE_EXTRA_BODY_JSON` to a JSON object for those cases.

**LLM judge** means Promptfoo's model-graded assertions such as
`llm-rubric`, `factuality`, and `context-faithfulness`. The `*.llm_judge.yaml`
configs use `openai:chat:{{ env.OPENAI_MODEL }}`. Point that environment at
an OpenAI-compatible local endpoint, vLLM, OpenRouter, or OpenAI as needed. The
`similar` assertion uses an embedding provider, so set `OPENAI_EMBEDDING_MODEL`
when the endpoint requires a provider-prefixed embedding model name.

**OpenWebUI bridge** is `api/openwebui_bridge.py`, a small FastAPI service that
receives Promptfoo rows and calls either OpenWebUI or an OpenAI-compatible
endpoint. Use `GENERATION_BACKEND=openwebui` for the summarization pipe/tool
under test. Use `GENERATION_BACKEND=openai_endpoint` only for direct-model
generation baselines.

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
| `lib/*.py` | Shared CSV, path, OpenWebUI, and OpenAI endpoint helpers | yes |
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
| `openwebui_pipe_model` | OpenWebUI model or pipe model id used by Step 2/3 generation; current OpenRouter-backed pipe is `summarizer---kohaku-OR` |
| `tool_parameters_json` | Extra JSON object copied into `<tool_parameters>` |
| `summarizer_model_id`, `algorithm`, `target_length`, `structure` | Common pipe/tool parameters copied into `<tool_parameters>` |
| `generation_temperature`, `generation_top_p`, `generation_max_tokens` | HTTP generation options sent to OpenWebUI or an OpenAI-compatible endpoint |
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

Note on providers: the `echo` provider tells Promptfoo to use the CSV `candidate_answer` column
as the model output (no external model call). This is the intended behaviour for Step 1,
where `step1.csv` already contains baseline answers, and for Step 2 offline evaluation,
where generation is performed separately (producing `step2_output.csv`) and Promptfoo then
evaluates those generated answers by echoing them. For live generation (Step 3) Promptfoo
uses the HTTP bridge provider to call the configured backend.


| Strategy | YAML suffix | What it checks |
|---|---|---|
| DEA | `.dea.yaml` | Native DEA plan/content/resource scores, length alignment, ROUGE/entity metrics, and DEA judge |
| LLM judge | `.llm_judge.yaml` | Promptfoo's model-graded assertions using the configured OpenAI-compatible endpoint |

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
["patch_examples", {"id": "kb-cases", "name": "Case law KB"}]
```

When `GENERATION_BACKEND=openwebui`, local files from `source_paths_json` are
uploaded to OpenWebUI and their resulting file ids are inserted into
`<files_list>`. This is the correct mode for testing the summarization
pipe/tool. When `GENERATION_BACKEND=openai_endpoint`, the bridge bypasses
OpenWebUI and appends trimmed source text directly to the endpoint prompt
instead.

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
GENERATION_BACKEND=openwebui
OPENWEBUI_BASE_URL=http://127.0.0.1:8080
OPENWEBUI_API_KEY=<local-openwebui-api-key>
OPENWEBUI_PIPE_MODEL=summarizer---kohaku-OR
DEA_USE_JUDGE=true
```

Choose one OpenAI-compatible evaluation endpoint:

| Backend | `OPENAI_BASE_URL` | `OPENAI_API_KEY` | `OPENAI_MODEL` / `DEA_JUDGE_MODEL` |
|---|---|---|---|
| Ollama | `http://127.0.0.1:11434/v1` | any non-empty value | local model, for example `qwen-laptop-dea:latest` |
| vLLM | `http://127.0.0.1:8000/v1` | local token or dummy value | served model id |
| OpenRouter | `https://openrouter.ai/api/v1` | OpenRouter key from your shell or `.env` | for example `qwen/qwen3.6-35b-a3b` |
| OpenAI | `https://api.openai.com/v1` | OpenAI key from your shell or `.env` | OpenAI model id |

For OpenRouter Qwen reasoning models that return reasoning separately from
message content, disable reasoning for JSON judge calls:

```bash
OPENAI_MODEL=qwen/qwen3.6-35b-a3b
OPENAI_EMBEDDING_MODEL=openai/text-embedding-3-small
DEA_JUDGE_MODEL=qwen/qwen3.6-35b-a3b
DEA_JUDGE_EXTRA_BODY_JSON='{"reasoning":{"enabled":false}}'
```

For a lower-latency hosted DEA judge, use the same OpenRouter endpoint with
the flash model:

```bash
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_MODEL=qwen/qwen3.6-flash
DEA_JUDGE_MODEL=qwen/qwen3.6-flash
DEA_JUDGE_EXTRA_BODY_JSON='{"reasoning":{"enabled":false}}'
```

For Promptfoo's own `llm-rubric`, `factuality`, and
`context-faithfulness` assertions, the equivalent setting lives in the
assertion provider config:

```yaml
provider:
  id: "openai:chat:{{ env.OPENAI_MODEL }}"
  config:
    showThinking: false
    passthrough:
      reasoning:
        enabled: false
```

Native DEA embeddings are controlled separately:

```bash
DEA_EMBEDDING_BACKEND=hf        # deterministic local default
DEA_EMBEDDING_MODEL=
```

Keep native DEA embeddings aligned with the embeddings used to create the DEA
dataset. Do not switch an existing DEA dataset from `hf` to an OpenAI-compatible
embedding backend unless you intentionally regenerate the dataset embeddings;
otherwise scores are not comparable and regeneration can be costly. Set
`DEA_EMBEDDING_BACKEND=openai` only for a dataset built for that embedding
backend, when the selected `OPENAI_BASE_URL` supports `/embeddings` and
`DEA_EMBEDDING_MODEL` is an embedding model. A chat model such as
`qwen/qwen3.6-35b-a3b` is not an embedding model unless that provider explicitly
exposes it through `/embeddings`.
For OpenRouter, a valid embedding model id uses a provider prefix, for example
`openai/text-embedding-3-small`.

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
{"ok":true,"backend":"openwebui","base_url":"http://127.0.0.1:8080","openai_endpoint_base_url":"http://127.0.0.1:11434/v1"}
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
--kb-id patch_examples
```

Those values are written into the generated CSV rows and later become
`<tool_parameters>` and `<kb_list>` blocks in the OpenWebUI user message.

Create Step 2 generated answers for one deterministic row:

```bash
docker compose run --rm promptfoo bash -lc '
  source /workspace/.venv/bin/activate
  python ./scripts/generate_candidate_csv.py \
    --input ./datasets/bigsurvey/step2_input.csv \
    --output ./datasets/bigsurvey/step2_output.csv \
    --bridge-url http://127.0.0.1:8001/generate \
    --limit 1 --overwrite
'
```

For large multi-file rows, raise both the bridge OpenWebUI timeout and the
generator request timeout. The generator writes completed rows incrementally,
so partial outputs are preserved if a later row fails:

```bash
OPENWEBUI_TIMEOUT_SECONDS=2400 docker compose up owui-bridge
docker compose run --rm promptfoo bash -lc '
  BRIDGE_REQUEST_TIMEOUT_SECONDS=2400 /workspace/.venv/bin/python ./scripts/generate_candidate_csv.py \
    --input ./datasets/bigsurvey/step2_input.csv \
    --output ./datasets/bigsurvey/step2_output.csv \
    --bridge-url http://127.0.0.1:8001/generate \
    --limit 3 --overwrite
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
| DEA shows only ROUGE-L | Check the assertion reason for `dea_status`; `computed` means native plan/content/resource metrics ran, while `empty` or `error` means the DEA target could not be scored |
| OpenAI-compatible API connection error | Confirm `OPENAI_BASE_URL`, `OPENAI_API_KEY`, and model env vars inside the container |
| LLM judge returns invalid JSON | Use a judge-capable model and check `OPENAI_MODEL` / `DEA_JUDGE_MODEL`; for reasoning models, set provider-specific JSON such as `DEA_JUDGE_EXTRA_BODY_JSON='{"reasoning":{"enabled":false}}'` |
| Step 3 cannot reach bridge | Use `http://127.0.0.1:8001/generate` with host networking |
| OpenWebUI path fails | Check `GENERATION_BACKEND=openwebui`, `OPENWEBUI_API_KEY`, and the selected pipe model |

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
