#!/bin/bash
# Double-click this file in Finder to run the SPY 1DTE pipeline on macOS.
# On first use: right-click → Open (to approve the untrusted script).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "╔═══════════════════════════════════════════╗"
echo "║   SPY 1DTE Vol Pipeline — macOS runner    ║"
echo "╚═══════════════════════════════════════════╝"
echo ""

# Require Python 3
if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found."
    echo "Install it from https://www.python.org/downloads/macos/"
    echo ""
    read -r -p "Press Enter to close…"
    exit 1
fi

PYTHON_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python $PYTHON_VER detected."
echo ""

# Install / upgrade dependencies quietly
echo "→ Checking / installing dependencies…"
python3 -m pip install --upgrade -q -r requirements.txt
echo "   Dependencies OK."
echo ""

echo "→ Running pipeline…"
echo ""
python3 spy_1dte_vol_pipeline.py
STATUS=$?

echo ""
if [ $STATUS -eq 0 ]; then
    echo "Pipeline finished successfully."
    echo "Output files are in:  $(pwd)/data/"
else
    echo "Pipeline exited with error code $STATUS."
fi

echo ""
read -r -p "Press Enter to close this window…"
