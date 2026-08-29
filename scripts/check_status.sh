#!/bin/bash
echo "=== Second-Level Analysis Status ==="
echo ""
echo "Stat Maps:"
ls -lh /root/fMRI/analysis/second_level/stat_maps/ 2>/dev/null | tail -20
echo ""
echo "Figures:"
ls -lh /root/fMRI/analysis/second_level/figures/ 2>/dev/null || echo "No figures yet"
echo ""
echo "Log file lines:"
wc -l /root/fMRI/logs/second_level_analysis.log 2>/dev/null || echo "No log file"
echo ""
echo "Running processes:"
pgrep -a python | grep second_level || echo "No Python process running"
