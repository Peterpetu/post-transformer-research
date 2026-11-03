#!/usr/bin/env bash
# Second batch of extra arms: the genome at a lower learning rate (its wider variant did not
# train at the default), one seed.  Runs after everything else.
set -u
cd "$(dirname "$0")"
while pgrep -f "run_queue.sh|run_lrcheck.sh|run_extra.sh" > /dev/null; do sleep 60; done
echo "=== extra2 $(date -u +%FT%TZ)" >> runs/queue.log
python3 e3_genome.py --budget 8000000 --seeds 0 --arms genome,genome_h192 --lr 0.0005 --suffix _lr0.0005 >> runs/e3_genome/log.txt 2>&1
python3 report.py > /dev/null 2>&1
echo "=== extra2 done $(date -u +%FT%TZ)" >> runs/queue.log
