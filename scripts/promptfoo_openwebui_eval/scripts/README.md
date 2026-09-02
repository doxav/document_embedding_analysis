# Scripts reference

Run these commands from `scripts/promptfoo_openwebui_eval` unless a command says otherwise.

## Inventory

| Script | Role |
|---|---|
| `bootstrap_repo_env.sh` | Checks Python/DEA/Promptfoo/Docker, selects supplied or managed OpenWebUI, provisions and tests the default model/tool, then starts and tests the bridge. |
| `setup_openwebui.py` | Creates local secrets, authenticates the admin, imports or updates a tool, applies Valves, creates a custom model, and tests inference/tool execution. |
| `build_promptfoo_csvs.py` | Builds BigSurvey, MultiLexSum, or native DEA Promptfoo rows and their parameter columns. |
| `generate_candidate_csv.py` | Runs generation outside Promptfoo and incrementally stores CSV metrics, raw bridge responses, and Markdown documents. |
| `export_promptfoo_results.py` | Exports Promptfoo outputs to Markdown and writes a compact JSON score summary. |
| `promptfoo_response.js` | Maps a live bridge response to Promptfoo output, token usage, cost, duration, and tool-call metadata. |
| `ensure_mds_outputs.py` | Checks or prepares the MDS dataset outputs used by the CSV builder. |

## Bootstrap modes

```bash
./scripts/bootstrap_repo_env.sh --full
```

`--full` is the supported first-run command. Its OpenWebUI decision is deliberate:

| `.env` state | Behaviour |
|---|---|
| `OPENWEBUI_BASE_URL=none`, `null`, or empty | Prints that no endpoint was supplied, starts `ghcr.io/open-webui/open-webui:v0.10.2`, provisions it, and rewrites the local ignored `.env`. |
| Reachable URL supplied | Uses it and does not create an OpenWebUI container. |
| Unreachable URL supplied and not marked managed | Fails without creating a container. |

`--runtime` only validates dependencies and reachability. It never provisions or starts OpenWebUI.

The full bootstrap verifies admin authentication, a bounded model inference, a real tool call with two attached documents in a persisted thread, bridge health, and bridge inference. OpenWebUI executes server tools only through its streaming persisted-chat route; the bridge hides that implementation detail behind its non-streaming `/generate` response.

## Provision another tool/model

The default is `tools/comparator/contradiction_auditor.py`. To benchmark another file from `OPENWEBUI_TOOLS_ROOT`, create a dedicated OpenWebUI model so each benchmark has an unambiguous tool and persistent Valve configuration:

```bash
.venv/bin/python scripts/setup_openwebui.py \
  --env-file .env \
  --repo-root "$DEA_REPO_PATH" \
  --tool-path summarize/thematic_summary_v5b.py \
  --model-id dea-large-thematic-dtcrs \
  --base-model-id deepseek/deepseek-v4-flash-0731 \
  --tool-valves-json '{"algorithm":"dtcrs","target_language":"en","default_target_length":"long","summary_structure":"thematic"}' \
  --preserve-defaults \
  --configure --test-inference
```

The created custom model's system prompt names the selected function and explicitly requires one tool call for matching requests. This prevents a model from answering the summarization/audit task without exercising the tool under test.

## Parameters: what changes where

There are three distinct parameter layers:

| Layer | Examples | How to set it |
|---|---|---|
| Persistent OpenWebUI tool Valves | thematic tool `algorithm`, language, internal model, context/output limits, map-reduce and concurrency | `setup_openwebui.py --tool-valves-json ...` |
| Per-call tool arguments | thematic `instruction`, `target_length`, `structure`; other tools expose their own arguments | CSV typed columns and `--tool-parameters-json` |
| Outer model request | temperature, top-p, max tokens, provider-specific reasoning controls | CSV generation flags and `openwebui_model_params_json` |

For `thematic_summary_v5b.py` version 0.7+, `algorithm` is a Valve, not a per-call function argument. The CSV `algorithm` column remains useful as an experiment label and for older pipes, but changing it alone does not change this tool's algorithm. Provision/update the Valve for a true algorithm comparison.

Model-extra JSON may not replace bridge-owned fields such as `model`, `messages`, `stream`, `files`, or `tool_ids`. OpenRouter reasoning is controlled at request time with `{"reasoning":{"enabled":false,"effort":"low"}}`; `default_enabled` and `default_effort` describe model defaults and are not request controls.

## Direct generation outside Promptfoo

```bash
DEA_REPO_ROOT="$DEA_REPO_PATH" .venv/bin/python scripts/build_promptfoo_csvs.py \
  --mode mds --dataset bigsurvey --stage step3 --limit 1 \
  --output results/my_run/input.csv \
  --pipe-model dea-large-thematic-dtcrs \
  --tool-id large_thematic_summarizer \
  --target-length long --structure thematic --temperature 0 --top-p 1

BRIDGE_REQUEST_TIMEOUT_SECONDS=2400 .venv/bin/python scripts/generate_candidate_csv.py \
  --input results/my_run/input.csv --output results/my_run/candidates.csv \
  --overwrite --responses-dir results/my_run/responses \
  --documents-dir results/my_run/documents
```

The raw response trace records the observed tool name/arguments plus OpenWebUI-reported envelope tokens, cost, and total wall time. OpenWebUI v0.10.2 does not add the tool's private internal LLM calls to that usage object, so these token/cost values are not a full internal-tool accounting.

## Live and offline Promptfoo

Live generation and scoring:

```bash
docker compose run --rm promptfoo bash -lc \
  'promptfoo eval -c ./bigsurvey.step3.dea.yaml -t ./results/my_run/input.csv \
   --output ./results/my_run/promptfoo.json --no-progress-bar --no-cache'
```

Score a candidate generated outside Promptfoo:

```bash
docker compose run --rm promptfoo bash -lc \
  'promptfoo eval -c ./bigsurvey.step2.dea.yaml -t ./results/my_run/candidates.csv \
   --output ./results/my_run/promptfoo.json --no-progress-bar --no-cache'
```

Export the generated documents and compact scores:

```bash
.venv/bin/python scripts/export_promptfoo_results.py \
  --input results/my_run/promptfoo.json \
  --documents-dir results/my_run/documents \
  --summary results/my_run/summary.json
```
