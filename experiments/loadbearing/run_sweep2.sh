#!/usr/bin/env bash
# Learning-rate sweep (continued) and budget check, one seed each.
#   1. lr 0.008 and 0.016 for each mechanism and incumbent (0.001, 0.002 and 0.004 already run)
#   2. the 2-layer transformer, E1's incumbent, over the whole sweep
#   3. the two headline mechanisms and the 2-layer transformer at three times the byte budget,
#      each at its own best learning rate from the sweep
set -u
cd "$(dirname "$0")"
while pgrep -f "run_queue.sh|run_lrcheck.sh|run_extra.sh|run_extra2.sh|run_extra3.sh|run_sweep.sh|e[1-6]_[a-z]*\.py" > /dev/null; do sleep 30; done
run() { exp=$1; shift; echo "=== sweep2 $exp $(date -u +%FT%TZ)" >> runs/queue.log; python3 "$@" >> "runs/$exp/log.txt" 2>&1; }
for lr in 0.008 0.016; do
  run e1_nca       e1_nca.py       --budget 8000000 --seeds 0 --arms nca_shared6,transformer_s --lr $lr --suffix _lr$lr
  run e2_reservoir e2_reservoir.py --budget 6000000 --seeds 0 --arms reservoir_r0.9,gru --lr $lr --suffix _lr$lr
  run e3_genome    e3_genome.py    --budget 8000000 --seeds 0 --arms genome,direct --lr $lr --suffix _lr$lr
  run e4_kuramoto  e4_kuramoto.py  --budget 6000000 --seeds 0 --arms kuramoto,gru --lr $lr --suffix _lr$lr
  run e5_morphogen e5_morphogen.py --budget 8000000 --seeds 0 --arms morphogen,transformer_s --lr $lr --suffix _lr$lr
  run e6_predcode  e6_predcode.py  --budget 6000000 --seeds 0 --arms pc_k3,gru --lr $lr --suffix _lr$lr
done
for lr in 0.001 0.004 0.008 0.016; do
  run e1_nca e1_nca.py --budget 8000000 --seeds 0 --arms transformer --lr $lr --suffix _lr$lr
done
python3 report.py > /dev/null 2>&1
best() { python3 -c "import json,sys; e=json.load(open('summary.json'))['experiments']['$1']; print(e['tuned']['$2'][0])"; }
echo "=== budget check $(date -u +%FT%TZ)" >> runs/queue.log
run e1_nca       e1_nca.py       --budget 24000000 --seeds 0 --arms transformer --lr "$(best e1_nca transformer)"   --suffix _b24m
run e1_nca       e1_nca.py       --budget 24000000 --seeds 0 --arms nca_shared6 --lr "$(best e1_nca nca_shared6)"  --suffix _b24m
run e5_morphogen e5_morphogen.py --budget 24000000 --seeds 0 --arms morphogen   --lr "$(best e5_morphogen morphogen)" --suffix _b24m
python3 report.py > /dev/null 2>&1
echo "=== sweep2 done $(date -u +%FT%TZ)" >> runs/queue.log
