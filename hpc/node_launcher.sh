#!/usr/bin/env bash
# node_launcher.sh
# run on each cluster node (or via batch scheduler)
# Example:
# NODE_ID=$(hostname)
# ./node_launcher.sh --db-url postgresql://user:pass@dbhost:5432/sepdb --node-id ${NODE_ID} --gpus 4

set -euo pipefail

DB_URL=""
NODE_ID=""
GPUS=4
WORK_DIR="/shared/checkpoints"
TRAINER_PY="trainer.py"
PYTHON_EXE="python"   # or python3
MASTER_PORT=29500     # must be reachable; if using SLURM set MASTER_ADDR env separately

while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --db-url) DB_URL="$2"; shift; shift;;
    --node-id) NODE_ID="$2"; shift; shift;;
    --gpus) GPUS="$2"; shift; shift;;
    --work-dir) WORK_DIR="$2"; shift; shift;;
    *) echo "Unknown arg $1"; exit 1;;
  esac
done

if [[ -z "$DB_URL" || -z "$NODE_ID" ]]; then
  echo "Usage: node_launcher.sh --db-url <db> --node-id <node> [--gpus N] [--work-dir DIR]"
  exit 1
fi

# Loop: claim jobs via DB and run torchrun for one shard per node (nproc_per_node = GPUS)
# The trainer.py will receive SHARD_PATH and DB_URL and update DB on completion/failure.
while true; do
  SHARD_PATH=$($PYTHON_EXE - <<PYCODE
import psycopg2, os, sys
conn = psycopg2.connect("${DB_URL}")
cur = conn.cursor()
# atomically claim a pending shard
cur.execute(\"\"\"
UPDATE shards SET status='running', claimed_by=%s, attempts=attempts+1, last_update=now()
WHERE shard_id = (
    SELECT shard_id FROM shards WHERE status='pending' ORDER BY shard_id LIMIT 1
) RETURNING shard_path;
\"\"\", ("${NODE_ID}",))
row = cur.fetchone()
conn.commit()
if row:
    print(row[0])
else:
    # no pending shard
    print("")
cur.close()
conn.close()
PYCODE
)
  if [[ -z "$SHARD_PATH" ]]; then
    echo "No pending shards. Sleeping 60s."
    sleep 60
    continue
  fi

  echo "Node ${NODE_ID} claimed shard ${SHARD_PATH}"

  # Launch torchrun across GPUs on this node
  # Use MASTER_ADDR=localhost if node-local; when multi-node multi-node training is needed, set MASTER_ADDR & PORT accordingly.
  export MASTER_ADDR=127.0.0.1
  export MASTER_PORT=${MASTER_PORT}
  export NCCL_DEBUG=INFO
  export PYTHONUNBUFFERED=1

  torchrun --nproc_per_node=${GPUS} ${TRAINER_PY} \
    --db-url "${DB_URL}" \
    --node-id "${NODE_ID}" \
    --shard-path "${SHARD_PATH}" \
    --work-dir "${WORK_DIR}" \
    || echo "Training failed for shard ${SHARD_PATH}"

  # After torchrun returns, check trainer updated DB. If trainer crashes without updating, mark shard failed.
  $PYTHON_EXE - <<PYCODE
import psycopg2, sys
conn = psycopg2.connect("${DB_URL}")
cur = conn.cursor()
cur.execute("SELECT status FROM shards WHERE shard_path=%s", ("${SHARD_PATH}",))
st = cur.fetchone()[0]
if st == 'running':
    # trainer didn't update; mark as failed so scheduler may requeue
    cur.execute("UPDATE shards SET status='failed', last_update=now() WHERE shard_path=%s", ("${SHARD_PATH}",))
    conn.commit()
cur.close(); conn.close()
PYCODE

done
