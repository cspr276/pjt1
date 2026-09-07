#!/bin/bash
################################################################################
# Post-Processing Pipeline Runner with TMUX
################################################################################
# 
# This script runs the post-processing pipeline in a tmux session for
# persistence across VPN disconnections.
#
# Usage:
#   ./run_postprocessing_tmux.sh
#
# To attach to the session:
#   tmux attach -t postprocessing
#
# To detach from the session (keep running):
#   Press Ctrl+B then D
#
# To kill the session:
#   tmux kill-session -t postprocessing
#
################################################################################

# Configuration
SESSION_NAME="postprocessing"
PROJECT_DIR="/root/fMRI"
SCRIPT_DIR="$PROJECT_DIR/scripts/python"
LOG_DIR="$PROJECT_DIR/logs/postprocessing"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/tmux_session_${TIMESTAMP}.log"

# Create log directory
mkdir -p "$LOG_DIR"

echo "================================================================================"
echo "POST-PROCESSING PIPELINE - TMUX SESSION LAUNCHER"
echo "================================================================================"
echo ""
echo "Session Name: $SESSION_NAME"
echo "Project Dir:  $PROJECT_DIR"
echo "Log File:     $LOG_FILE"
echo ""

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo "ERROR: tmux is not installed!"
    echo "Please install tmux: apt-get install tmux"
    exit 1
fi

# Check if session already exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "WARNING: Session '$SESSION_NAME' already exists!"
    echo ""
    echo "Options:"
    echo "  1. Attach to existing session:    tmux attach -t $SESSION_NAME"
    echo "  2. Kill and restart:             tmux kill-session -t $SESSION_NAME"
    echo "  3. Create new session:           Change SESSION_NAME in this script"
    echo ""
    exit 1
fi

# Create new tmux session
echo "Creating new tmux session..."
tmux new-session -d -s "$SESSION_NAME" -x 200 -y 50

# Set working directory
tmux send-keys -t "$SESSION_NAME" "cd $PROJECT_DIR" C-m

# Activate Python environment (if using virtualenv)
# Uncomment and modify if needed:
# tmux send-keys -t "$SESSION_NAME" "source /path/to/venv/bin/activate" C-m

# Start logging
tmux send-keys -t "$SESSION_NAME" "exec > >(tee -a '$LOG_FILE') 2>&1" C-m

# Print header
tmux send-keys -t "$SESSION_NAME" "echo '================================================================================'" C-m
tmux send-keys -t "$SESSION_NAME" "echo 'POST-PROCESSING PIPELINE - TMUX SESSION'" C-m
tmux send-keys -t "$SESSION_NAME" "echo '================================================================================'" C-m
tmux send-keys -t "$SESSION_NAME" "echo 'Started: $(date)'" C-m
tmux send-keys -t "$SESSION_NAME" "echo 'Session: $SESSION_NAME'" C-m
tmux send-keys -t "$SESSION_NAME" "echo 'Log:     $LOG_FILE'" C-m
tmux send-keys -t "$SESSION_NAME" "echo ''" C-m

# Run the pipeline
tmux send-keys -t "$SESSION_NAME" "python3 $SCRIPT_DIR/postprocess_fmri.py" C-m

# Print completion message
tmux send-keys -t "$SESSION_NAME" "echo ''" C-m
tmux send-keys -t "$SESSION_NAME" "echo '================================================================================'" C-m
tmux send-keys -t "$SESSION_NAME" "echo 'PIPELINE COMPLETE'" C-m
tmux send-keys -t "$SESSION_NAME" "echo '================================================================================'" C-m
tmux send-keys -t "$SESSION_NAME" "echo 'Finished: $(date)'" C-m

echo ""
echo "✓ TMUX session created successfully!"
echo ""
echo "================================================================================"
echo "HOW TO MANAGE THE TMUX SESSION"
echo "================================================================================"
echo ""
echo "1. ATTACH TO SESSION (view execution):"
echo "   $ tmux attach -t $SESSION_NAME"
echo ""
echo "2. DETACH FROM SESSION (keep running in background):"
echo "   Press: Ctrl+B then D"
echo ""
echo "3. LIST ALL TMUX SESSIONS:"
echo "   $ tmux ls"
echo ""
echo "4. KILL THE SESSION:"
echo "   $ tmux kill-session -t $SESSION_NAME"
echo ""
echo "5. VIEW LOG FILE:"
echo "   $ tail -f $LOG_FILE"
echo ""
echo "================================================================================"
echo "IMPORTANT NOTES"
echo "================================================================================"
echo ""
echo "✓ The pipeline is now running in the background"
echo "✓ Your VPN disconnection will NOT interrupt the process"
echo "✓ All output is being logged to: $LOG_FILE"
echo "✓ You can safely close this terminal"
echo ""
echo "To monitor progress:"
echo "  - Attach to tmux session (see above)"
echo "  - Or tail the log file: tail -f $LOG_FILE"
echo ""
echo "================================================================================"
