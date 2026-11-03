#!/usr/bin/env bash
# Learning-rate check: the mechanism arm and the incumbent of each experiment at half and
# double the default learning rate, one seed.  Runs after the main queue.
set -u
cd "$(dirname "$0")"
while pgrep -f run_queue.sh > /dev/null; do sleep 60; done
run() { exp=$1; shift; echo "=== lrcheck $exp $(date -u +%FT%TZ)" >> runs/queue.log; python3 "$@" >> "runs/$exp/log.txt" 2>&1; }
for lr in 0.001 0.004; do
  run e1_nca       e1_nca.py       --budget 8000000 --seeds 0 --arms nca_shared6,transformer_s --lr $lr --suffix _lr$lr
  run e2_reservoir e2_reservoir.py --budget 6000000 --seeds 0 --arms reservoir_r0.9,gru --lr $lr --suffix _lr$lr
  run e3_genome    e3_genome.py    --budget 8000000 --seeds 0 --arms genome,direct --lr $lr --suffix _lr$lr
  run e4_kuramoto  e4_kuramoto.py  --budget 6000000 --seeds 0 --arms kuramoto,gru --lr $lr --suffix _lr$lr
  run e5_morphogen e5_morphogen.py --budget 8000000 --seeds 0 --arms morphogen,transformer_s --lr $lr --suffix _lr$lr
  run e6_predcode  e6_predcode.py  --budget 6000000 --seeds 0 --arms pc_k3,gru --lr $lr --suffix _lr$lr
done
python3 report.py > /dev/null 2>&1
echo "=== lrcheck done $(date -u +%FT%TZ)" >> runs/queue.log
