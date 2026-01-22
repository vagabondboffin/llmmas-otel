# Demo: ChatDev-Ollama + llmmas-otel (Milestone 1 observability)

This demo runs **ChatDev (Ollama backend)** instrumented with **llmmas-otel** and exports traces to **Jaeger** via OTLP.

Expected trace structure (proposal-aligned):

`llmmas.session → llmmas.workflow.phase → agent_step → send/process`

## What’s in this demo folder

```
demos/chatdev-ollama/
  ChatDev-Ollama/                 # git submodule (pinned)
  data/programdev_sample3.json     # 3-task sample dataset
  docker/compose.yaml              # Jaeger (OTLP enabled)
  scripts/run_programdev_dataset_otel.py
  README.md
```

Submodule repo: `https://github.com/vagabondboffin/ChatDev-Ollama.git`  
Pinned commit (expected): `c25ae2ea2ecef80d23052450b5e77c9a67926213`

---

## Prerequisites

- Python 3.10+ recommended
- Docker (for Jaeger)
- Ollama installed + running locally
  - Check: `ollama list`
  - If you don’t have a model yet, pull one (example): `ollama pull llama3`

> If you use a different model than the example above, you may need to update the model name in ChatDev’s configuration (inside the `ChatDev-Ollama/` submodule).

---

## 1) Clone llmmas-otel with the ChatDev submodule

**Recommended (one command):**
```bash
git clone --recurse-submodules https://github.com/<YOUR_GITHUB_USERNAME>/llmmas-otel.git
```

If you already cloned it without submodules:
```bash
cd llmmas-otel
git submodule update --init --recursive
```

Optional sanity check:
```bash
cd demos/chatdev-ollama/ChatDev-Ollama
git rev-parse HEAD
# should print: c25ae2ea2ecef80d23052450b5e77c9a67926213
```

---

## 2) Start Jaeger (OTLP enabled)

From the **repo root** (`llmmas-otel/`):

```bash
docker compose -f demos/chatdev-ollama/docker/compose.yaml up -d
```

Jaeger UI: http://localhost:16686

---

## 3) Create a virtual environment for ChatDev + install dependencies

```bash
cd demos/chatdev-ollama/ChatDev-Ollama

python -m venv .venv
source .venv/bin/activate

pip install -U pip setuptools wheel
pip install -r requirements.txt
```

---

## 4) Install llmmas-otel into the ChatDev venv (editable)

From inside `demos/chatdev-ollama/ChatDev-Ollama/`:

```bash
pip install -e ../../..
python -c "import llmmas_otel; print('llmmas_otel loaded from:', llmmas_otel.__file__)"
```

---

## 5) Run the ProgramDev sample (first 3 tasks)

This demo runner lives in the `llmmas-otel` repo under `demos/chatdev-ollama/scripts/`.

### Option A (simplest): copy the runner into ChatDev and run
```bash
cp ../scripts/run_programdev_dataset_otel.py .
OPENAI_API_KEY=ollama python run_programdev_dataset_otel.py \
  --dataset ../data/programdev_sample3.json \
  --limit 3
```

### Option B: run the runner in-place (no copy)
```bash
PYTHONPATH=. OPENAI_API_KEY=ollama python ../scripts/run_programdev_dataset_otel.py \
  --dataset ../data/programdev_sample3.json \
  --limit 3
```

> Note: Some parts of ChatDev may still read `OPENAI_API_KEY` at import time for compatibility.  
> For Ollama, the value can be any string (we use `ollama`).

---

## 6) View traces in Jaeger

Open http://localhost:16686 and select service: **`chatdev-programdev`**.

You should see spans like:
- `llmmas.session`
- `llmmas.workflow.segment` / phase span
- `agent_step`
- `send X->Y`
- `process X->Y`

---

## Troubleshooting

### Jaeger exporter: `StatusCode.UNAVAILABLE`
- Confirm Jaeger is running:
  ```bash
  docker ps
  ```
- Ensure port `4317` is exposed by the compose file.

### ChatDev errors about missing log files under `WareHouse/`
The demo integration creates required log files/directories. If you still see a file-not-found error,
ensure you are running from the `ChatDev-Ollama/` directory (Option A or Option B above).

### Red imports in PyCharm / IDE
Make sure the interpreter for `ChatDev-Ollama/` is set to:
`demos/chatdev-ollama/ChatDev-Ollama/.venv/bin/python`,
and that you ran:
`pip install -e ../../..`
