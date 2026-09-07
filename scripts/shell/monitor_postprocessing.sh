#!/bin/bash
################################################################################
# Post-Processing Pipeline Monitor
################################################################################
# 
# This script monitors the progress of the post-processing pipeline
#
# Usage:
#   ./monitor_postprocessing.sh
#
################################################################################

echo "================================================================================"
echo "POST-PROCESSING PIPELINE MONITOR"
echo "================================================================================"
echo ""

# Check if tmux session is running
if tmux has-session -t postprocessing 2>/dev/null; then
    echo "✓ TMUX session 'postprocessing' is RUNNING"
else
    echo "✗ TMUX session 'postprocessing' is NOT RUNNING"
    echo ""
    echo "To start the pipeline:"
    echo "  $ cd /root/fMRI/scripts/shell"
    echo "  $ ./run_postprocessing_tmux.sh"
    exit 1
fi

echo ""
echo "================================================================================"
echo "PROGRESS SUMMARY"
echo "================================================================================"
echo ""

# Count processed subjects
SMOOTHED=$(ls /root/fMRI/output/postprocessed/smoothed/*.nii.gz 2>/dev/null | wc -l)
FILTERED=$(ls /root/fMRI/output/postprocessed/filtered/*.nii.gz 2>/dev/null | wc -l)
CLEANED=$(ls /root/fMRI/output/postprocessed/cleaned/*.nii.gz 2>/dev/null | wc -l)
ROI=$(ls /root/fMRI/output/postprocessed/roi_timeseries/*.npy 2>/dev/null | wc -l)
QC=$(ls /root/fMRI/output/postprocessed/quality_metrics/*.json 2>/dev/null | wc -l)

echo "Subjects processed:"
echo "  - Spatial smoothing:  $SMOOTHED / 71"
echo "  - Temporal filtering: $FILTERED / 71"
echo "  - Confound cleaning:  $CLEANED / 71"
echo "  - ROI extraction:     $ROI / 71"
echo "  - QC metrics:         $QC / 71"
echo ""

# Check disk usage
echo "================================================================================"
echo "DISK USAGE"
echo "================================================================================"
echo ""

df -h /root/fMRI | grep -v Filesystem

echo ""
echo "Output directory size:"
du -sh /root/fMRI/output/postprocessed 2>/dev/null || echo "Calculating..."

echo ""
echo "================================================================================"
echo "LATEST LOG ENTRIES"
echo "================================================================================"
echo ""

# Show latest log entries
LATEST_LOG=$(ls -t /root/fMRI/logs/postprocessing/postprocessing_*.log 2>/dev/null | head -1)
if [ -n "$LATEST_LOG" ]; then
    echo "Log file: $LATEST_LOG"
    echo ""
    tail -n 20 "$LATEST_LOG"
else
    echo "No log files found"
fi

echo ""
echo "================================================================================"
echo "MONITORING OPTIONS"
echo "================================================================================"
echo ""
echo "1. Attach to tmux session (view live execution):"
echo "   $ tmux attach -t postprocessing"
echo ""
echo "2. Tail the log file:"
echo "   $ tail -f $LATEST_LOG"
echo ""
echo "3. Check processing summary:"
echo "   $ cat /root/fMRI/output/postprocessed/processing_summary.json"
echo ""
echo "4. List processed subjects:"
echo "   $ ls /root/fMRI/output/postprocessed/roi_timeseries/"
echo ""
echo "================================================================================"
