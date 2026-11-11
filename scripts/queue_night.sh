#!/usr/bin/env bash
# Madrugada de 10→11/11: espera a extensão da f2, confirma f1x/f2x no dev48 e roda a fila 2.
set -uo pipefail
cd /mnt/dados/dev/agentic-weight-editing
OLD=/mnt/dados/dev/deep_traces/data/wsp/m2a/runs
DEADLINE=$(date -d "tomorrow 00:50" +%s); [[ $(date +%H) -lt 12 ]] && DEADLINE=$(date -d "today 00:50" +%s)
echo "esperando final.json novo da f2 (até 00:50): $(date)"
until [[ $OLD/f2/final.json -nt data/m2a/runs/f2/final_1840.json && $(stat -c %Y $OLD/f2/final.json) -gt $(date -d "2026-11-10 23:11" +%s) ]] || [[ $(date +%s) -ge $DEADLINE ]]; do sleep 30; done
sleep 60
echo "seguindo: $(date)"
args=()
for r in f1 f2; do
  rsync -a --exclude sealed $OLD/$r/ data/m2a/runs/$r/
  new=$(uv run --no-sync python -c "import json;print(json.load(open('data/m2a/runs/$r/final.json'))['ranked'][0])")
  old=$(uv run --no-sync python -c "import json;print(json.load(open('data/m2a/runs/$r/final_1840.json'))['ranked'][0])")
  echo "$r: 18h40=$old agora=$new"
  [[ $new != "$old" ]] && args+=("$r:$new")
done
[[ ${#args[@]} -gt 0 ]] && scripts/queue_confirm.sh "${args[@]}" || { systemctl --user stop wsp-m2a 2>/dev/null; sleep 5; }
exec scripts/queue_fix2.sh
