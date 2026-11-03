#!/usr/bin/env bash
# Extra arms added after seeing the first results; runs after the main queue and the LR check.
set -u
cd "$(dirname "$0")"
while pgrep -f "run_queue.sh|run_lrcheck.sh" > /dev/null; do sleep 60; done
echo "=== extra $(date -u +%FT%TZ)" >> runs/queue.log
python3 e3_genome.py --budget 8000000 --seeds 0,1 --arms genome_h192 >> runs/e3_genome/log.txt 2>&1
python3 report.py > /dev/null 2>&1
echo "=== extra done $(date -u +%FT%TZ)" >> runs/queue.log
