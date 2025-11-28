# scheduler.py
# Usage: python scheduler.py --data-dir /shared/data/shards --db-url postgresql://user:pass@dbhost:5432/sepdb

import argparse, os, psycopg2, time, json
from psycopg2.extras import execute_values

def discover_shards(data_dir):
    # expects shards named *.tar or *.tar.zst etc
    exts = ('.tar', '.tar.zst', '.tar.gz')
    return sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(exts)])

def upsert_shards(conn, shards):
    with conn.cursor() as cur:
        # insert if not exists
        vals = [(s,) for s in shards]
        execute_values(cur,
            "INSERT INTO shards (shard_path) VALUES %s ON CONFLICT (shard_path) DO NOTHING",
            vals)
    conn.commit()

def monitor(conn, poll_interval=300):
    while True:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM shards WHERE status = 'pending'")
            pending = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM shards WHERE status = 'running'")
            running = cur.fetchone()[0]
            print(f"[scheduler] pending={pending} running={running}")
        time.sleep(poll_interval)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', required=True)
    p.add_argument('--db-url', required=True)
    args = p.parse_args()
    conn = psycopg2.connect(args.db_url)
    shards = discover_shards(args.data_dir)
    print(f"[scheduler] discovered {len(shards)} shards")
    upsert_shards(conn, shards)
    try:
        monitor(conn)
    except KeyboardInterrupt:
        print("Scheduler exiting.")
