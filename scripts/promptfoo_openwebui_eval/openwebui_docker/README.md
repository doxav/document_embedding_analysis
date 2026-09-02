# Managed OpenWebUI

This compose project is used only when the parent `.env` does not supply an endpoint (`OPENWEBUI_BASE_URL=none`, `null`, or empty), or explicitly marks the local endpoint managed. A reachable supplied endpoint always wins and no OpenWebUI container is created.

The compose intentionally pins `ghcr.io/open-webui/open-webui:v0.10.2`, publishes only `127.0.0.1:8080`, enables authentication/API keys, disables signup and telemetry, and persists `/app/backend/data` in `openwebui_data`. `RESET_CONFIG_ON_START=true` makes the declared OpenAI/OpenRouter provider configuration authoritative on restart. The configured model allow-list defaults to `deepseek/deepseek-v4-flash-0731`.

`scripts/setup_openwebui.py --prepare-local-env` creates the ignored `.env` with mode `0600`. It copies `OPENROUTER_API_KEY` from the parent ignored `.env` and generates missing admin/secret values. Never copy real values into `.env.example`.

Normal lifecycle is owned by the parent bootstrap:

```bash
../scripts/bootstrap_repo_env.sh --full
```

For diagnostics only:

```bash
docker compose --env-file .env up -d --wait
docker compose ps
docker compose logs openwebui
```

Removing the named volume deletes local OpenWebUI users, models, tools, chats, and uploaded documents. Do not run `down -v` as a routine restart.
