#!/usr/bin/env bash
# llama-server (o binário que vem com o Ollama) com slots paralelos para o Qwen3.5.
# O Ollama força -np 1 para a arquitetura qwen35 ("does not currently support parallel requests"),
# então para o experimento usamos o mesmo binário direto. Um modelo por vez.
#
# uso: scripts/llama_serve.sh start <modelo-ollama|arquivo.gguf> [np] [ctx_por_slot]
#      scripts/llama_serve.sh stop | status
# env: WSP_LLAMA_PORT (8090), WSP_LLAMA_DIR (estado/log), WSP_LLAMA_NGL=0 -> só CPU (testes)
set -euo pipefail
DIR=${WSP_LLAMA_DIR:-$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)/data/llama}
NGL=${WSP_LLAMA_NGL:-999}
BIN=/usr/local/lib/ollama/llama-server
PORT=${WSP_LLAMA_PORT:-8090}
mkdir -p "$DIR"

blob_of() {  # nome de modelo do Ollama -> caminho do GGUF (camada "model" do manifest)
  local m=$1
  [[ -f $m ]] && { echo "$m"; return; }
  ollama show --modelfile "$m" | awk '/^FROM \//{print $2; exit}'
}

case "${1:-status}" in
start)
  "$0" stop >/dev/null 2>&1 || true
  ollama stop "$2" >/dev/null 2>&1 || true   # libera a VRAM se o Ollama estiver com ele carregado
  GGUF=$(blob_of "$2"); NP=${3:-16}; CTX=${4:-9216}
  BACKEND=/usr/local/lib/ollama/cuda_v13/libggml-cuda.so; [[ $NGL == 0 ]] && BACKEND=/nonexistent
  LD_LIBRARY_PATH=/usr/local/lib/ollama:/usr/local/lib/ollama/cuda_v13 \
  GGML_BACKEND_PATH=$BACKEND \
    setsid nohup "$BIN" --model "$GGUF" --port "$PORT" --host 127.0.0.1 --no-webui --offline \
      -c $((NP * CTX)) -np "$NP" -ngl "$NGL" --flash-attn auto -b 2048 -ub 512 \
      --cache-ram 0 --ctx-checkpoints 1 --metrics >"$DIR/server.log" 2>&1 &  # sem cache em RAM: o padrão (8 GB) causou OOM
  echo $! >"$DIR/server.pid"; echo "$2 $NP $CTX" >"$DIR/model"
  for _ in $(seq 60); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break; sleep 1; done
  curl -s "http://127.0.0.1:$PORT/health"; echo " $2 np=$NP ctx/slot=$CTX porta=$PORT"
  ;;
stop)
  [[ -f $DIR/server.pid ]] && { kill "$(cat "$DIR/server.pid")" 2>/dev/null || true; }; rm -f "$DIR/server.pid"; echo parado
  ;;
status)
  curl -s "http://127.0.0.1:$PORT/health" && echo " $(cat "$DIR/model" 2>/dev/null)" || echo "fora do ar"
  ;;
esac
