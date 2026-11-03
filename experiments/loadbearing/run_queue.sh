#!/usr/bin/env bash
# Runs the experiments one after another (two cores; two at once would only slow each
# other down).  Each arm whose record already exists is skipped, so the queue can be
# restarted after an interruption.
set -u
cd "$(dirname "$0")"
run() { exp=$1; shift; mkdir -p "runs/$exp"; echo "=== $exp $(date -u +%FT%TZ)" >> runs/queue.log; python3 "$@" >> "runs/$exp/log.txt" 2>&1; echo "=== $exp done $(date -u +%FT%TZ) rc=$?" >> runs/queue.log; }
run e1_nca       e1_nca.py       --budget 8000000 --seeds 0,1
run e2_reservoir e2_reservoir.py --budget 6000000 --seeds 0,1
run e3_genome    e3_genome.py    --budget 8000000 --seeds 0,1
run e4_kuramoto  e4_kuramoto.py  --budget 6000000 --seeds 0,1
run e5_morphogen e5_morphogen.py --budget 8000000 --seeds 0,1
run e6_predcode  e6_predcode.py  --budget 6000000 --seeds 0,1
python3 report.py > /dev/null 2>&1
echo "=== all done $(date -u +%FT%TZ)" >> runs/queue.log
