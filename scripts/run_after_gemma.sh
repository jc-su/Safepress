#!/bin/bash
# Wait for the current experiment to finish, then run new models
LOG=/home/jis23009/Dev/safepress_repo/runs/remaining_fixed.log

echo "Waiting for current experiment to complete..."
while pgrep -f "run_remaining_fixed" > /dev/null; do
    sleep 60
    echo "$(date): Still running... $(tail -c 500 $LOG | grep -oE 'Generating:.*%' | tail -1)"
done

echo "$(date): Current experiment finished. Checking results..."
ls -la /home/jis23009/Dev/safepress_repo/runs/*_ssmp_gemma2_9b/*.json 2>/dev/null || echo "No Gemma2 results yet"

echo ""
echo "$(date): Starting new models experiment..."
cd /home/jis23009/Dev/safepress_repo
nohup python scripts/run_new_models.py > runs/new_models.log 2>&1 &
echo "Started new models experiment with PID $!"
