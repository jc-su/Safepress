#!/bin/bash
echo "=== Experiment Status $(date) ==="
echo ""
echo "Process status:"
ps aux | grep "run_remaining_fixed" | grep -v grep || echo "  Not running"
echo ""
echo "GPU usage:"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null || echo "  N/A"
echo ""
echo "Last 20 lines of log:"
tail -20 /home/jis23009/Dev/safepress_repo/runs/remaining_fixed.log 2>/dev/null || echo "  No log yet"
echo ""
echo "Completed results:"
ls -la /home/jis23009/Dev/safepress_repo/runs/*ssmp_phi4_mini/ 2>/dev/null | head -5
ls -la /home/jis23009/Dev/safepress_repo/runs/*ssmp_smollm3_3b/ 2>/dev/null | head -5
ls -la /home/jis23009/Dev/safepress_repo/runs/*ssmp_gemma2_9b/*.json 2>/dev/null | head -5
