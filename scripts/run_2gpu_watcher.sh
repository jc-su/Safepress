#!/bin/bash
# All-done watcher: waits for both Track A and Track B to complete,
# then emits the FISHER60_AND_G1_ALL_DONE sentinel.

set -u
cd /home/jis23009/Dev/safepress_repo

LOG=runs/emnlp_fisher60/all_done_watcher.log
mkdir -p runs/emnlp_fisher60

echo "===== All-done watcher started $(date -Iseconds) =====" > $LOG
echo "Waiting for TRACK_A_DONE and TRACK_B_DONE..." >> $LOG

A_LOG=runs/emnlp_fisher60/track_a_cuda0.log
B_LOG=runs/emnlp_fisher60/track_b_cuda1.log
A_SENTINEL="===== TRACK_A_DONE"
B_SENTINEL="===== TRACK_B_DONE"

while true; do
    a_done=0
    b_done=0
    [ -f "$A_LOG" ] && grep -F "$A_SENTINEL" "$A_LOG" > /dev/null 2>&1 && a_done=1
    [ -f "$B_LOG" ] && grep -F "$B_SENTINEL" "$B_LOG" > /dev/null 2>&1 && b_done=1
    if [ $a_done -eq 1 ] && [ $b_done -eq 1 ]; then
        break
    fi
    sleep 120
done

echo "===== FISHER60_AND_G1_ALL_DONE $(date -Iseconds) =====" >> $LOG

# Run final synthesis
echo "" >> $LOG
echo "----- Running synthesis $(date -Iseconds) -----" >> $LOG
python scripts/synthesize_g2_pilot_table.py >> $LOG 2>&1 || \
    echo "[synthesis failed]" >> $LOG
python scripts/generate_g1_figure.py >> $LOG 2>&1 || \
    echo "[g1 figure failed]" >> $LOG

echo "" >> $LOG
echo "===== ALL_TASKS_DONE $(date -Iseconds) =====" >> $LOG
