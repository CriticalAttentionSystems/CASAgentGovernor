#!/bin/bash
# enable_gemini_today.sh
#
# Run this on any day you want tomorrow's (or tonight's, depending on your
# cron time) scheduled Gemini run to actually execute. Does nothing by
# itself -- run_gemini_sham_experiment.sh checks for this flag and no-ops
# if it's absent.
#
# Usage: ./enable_gemini_today.sh

set -euo pipefail

PROJECT_DIR="/Users/veronique/BASE/Code/CASAgentGovernor"
FLAG_FILE="$PROJECT_DIR/.run_gemini_today"

touch "$FLAG_FILE"
echo "Enabled: the next scheduled run will execute. (Flag file: $FLAG_FILE)"
echo "It will be consumed (deleted) automatically once that run starts --"
echo "run this script again on any future day you want to opt back in."
