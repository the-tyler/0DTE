#!/bin/bash
# Cron wrapper for the SPY 1DTE IV pipeline.
# Invoked by crontab at 4:00 pm ET Mon–Fri.
# Logs every run to data/pipeline.log with a timestamped header.

SCRIPT_DIR="/Users/sarp/Desktop/Files/Python Workspace/1DTE-SPX-OPTION"
LOG_FILE="$SCRIPT_DIR/data/pipeline.log"
PYTHON="/opt/anaconda3/bin/python3"

# Rotate log when it exceeds 5 MB to avoid unbounded growth
LOG_SIZE=$(stat -f%z "$LOG_FILE" 2>/dev/null || echo 0)
if [ "$LOG_SIZE" -gt 5242880 ]; then
    mv "$LOG_FILE" "${LOG_FILE}.old"
fi

{
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  Cron trigger: $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "════════════════════════════════════════════════════════════"
} >> "$LOG_FILE"

cd "$SCRIPT_DIR"
"$PYTHON" spy_1dte_vol_pipeline.py >> "$LOG_FILE" 2>&1

EXIT_CODE=$?
echo "  Exit code: $EXIT_CODE  —  finished $(date '+%H:%M:%S')" >> "$LOG_FILE"
