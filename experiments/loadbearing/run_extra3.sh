#!/usr/bin/env bash
# Bigram-level null arms for the two recurrent families whose first nulls could not represent a bigram model.
set -u
cd "$(dirname "$0")"
while pgrep -f "run_queue.sh|run_lrcheck.sh|run_extra.sh|run_extra2.sh" > /dev/null; do sleep 60; done
echo "=== extra3 $(date -u +%FT%TZ)" >> runs/queue.log
python3 e4_kuramoto.py --budget 6000000 --seeds 0,1 --arms kuramoto_nomem >> runs/e4_kuramoto/log.txt 2>&1
python3 e6_predcode.py --budget 6000000 --seeds 0,1 --arms pc_nomem >> runs/e6_predcode/log.txt 2>&1
python3 report.py > /dev/null 2>&1
echo "=== extra3 done $(date -u +%FT%TZ)" >> runs/queue.log
