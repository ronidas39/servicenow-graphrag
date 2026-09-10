#!/usr/bin/env bash
#
# Two models, one GPU, two ports. Runs ON the server.
#
# Part 8 sections 84 and 85. The article needs both halves of a RAG system running on
# hardware you control: the model that writes the answer, and the model that turns text
# into vectors. Nothing here talks to a hosted API, which is the entire point of Part 3.
#
# ⛔ TWO SERVERS ON ONE CARD, AND THE SPLIT IS EXPLICIT. vLLM reserves KV cache up front
# from a fraction of the card, and that fraction defaults to 0.9. Start the second server
# with the default and it asks for 90% of a card that already has 90% spoken for, and it
# dies with a CUDA out of memory that names a number far smaller than the card you rented.
# The two fractions below add up to less than one on purpose.
#
# ⛔ BOTH MODELS ARE UNGATED. A gated model needs a Hugging Face token and an accepted
# licence, which turns "run this script" into "go and fill in a form". Every reader can
# run these two.
#
# Author: Roni Das
# Created: 2026-09-10

set -euo pipefail

CHAT_MODEL="${CHAT_MODEL:-Qwen/Qwen2.5-7B-Instruct-AWQ}"
EMBED_MODEL="${EMBED_MODEL:-Qwen/Qwen3-Embedding-0.6B}"
VENV="$HOME/vllm-env/bin"

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

wait_for() {  # port, name, minutes
  local port="$1" name="$2" mins="${3:-15}" i
  for i in $(seq 1 $(( mins * 12 )) ); do
    if curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      echo "  $name answered on $port after $(( i * 5 ))s"; return 0
    fi
    sleep 5
  done
  echo "  ⛔ $name never answered on $port"; tail -40 "$HOME/${name}.log"; return 1
}

say "the card, before anything is loaded"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv

say "1. the model that writes the answer, on 8000"
# --gpu-memory-utilization 0.60: three fifths of the card, leaving room for the second
# server. --max-model-len 8192: enough for a retrieved context plus a question, and small
# enough that the KV cache fits in what is left.
nohup "$VENV/vllm" serve "$CHAT_MODEL" \
  --host 0.0.0.0 --port 8000 \
  --gpu-memory-utilization 0.60 \
  --max-model-len 8192 \
  --served-model-name chat \
  > "$HOME/chat.log" 2>&1 &
wait_for 8000 chat 20

say "2. the model that makes the vectors, on 8001"
# --task embed makes vLLM load this as a pooling model rather than a generator. Without
# it vLLM tries to serve /v1/completions from an encoder and fails in a way that reads
# like a broken model rather than a wrong flag.
nohup "$VENV/vllm" serve "$EMBED_MODEL" \
  --host 0.0.0.0 --port 8001 \
  --task embed \
  --gpu-memory-utilization 0.25 \
  --max-model-len 4096 \
  --served-model-name embed \
  > "$HOME/embed.log" 2>&1 &
wait_for 8001 embed 20

say "the card, with both models resident"
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv

say "3. proof that each one answers"
echo "-- chat --"
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"chat","messages":[{"role":"user","content":"In one sentence: what is a configuration item in a CMDB?"}],"max_tokens":60,"temperature":0}' \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['message']['content'].strip())"
echo "-- embed --"
curl -s http://127.0.0.1:8001/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"model":"embed","input":"payments service is down"}' \
  | python3 -c "import json,sys; e=json.load(sys.stdin)['data'][0]['embedding']; \
print(f'{len(e)} dimensions, first three {[round(x,4) for x in e[:3]]}')"

say "both serving"
