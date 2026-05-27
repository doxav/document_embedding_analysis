FROM node:20-bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       python3 \
       python3-venv \
       python3-pip \
       build-essential \
       curl \
       ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN npm install -g promptfoo@latest --omit=optional --no-audit --no-fund

WORKDIR /workspace/promptfoo_openwebui_dea

COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --break-system-packages --no-cache-dir -r /tmp/requirements.txt

CMD ["bash"]
