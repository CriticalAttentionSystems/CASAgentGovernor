#!/bin/bash
#
# run_gemini_sham_experiment.sh
#
# Wrapper invoked daily by cron. Does NOTHING unless you've explicitly
# opted in for today by creating the flag file (see "enable_today.sh"
# below, or just run: touch .run_gemini_today).
#
# Consumes (deletes) the flag after running, so tomorrow requires a new,
# deliberate choice rather than silently repeating forever.
#
# WHY A FLAG FILE INSTEAD OF JUST "run every day": you asked to choose
# which days this runs, and RunGeminiTaskComparison spends real, scarce
# daily quota every time it runs -- an unconditional daily cron job would
# take that choice away from you. This gives you a one-line on switch
# (`touch .run_gemini_today`) with no crontab editing required to change
# your mind day to day.

set -euo pipefail

# ---------------------------------------------------------------------
# EDIT THIS to your actual project path if it differs.
# ---------------------------------------------------------------------
PROJECT_DIR="/Users/veronique/BASE/Code/CASAgentGovernor"

FLAG_FILE="$PROJECT_DIR/.run_gemini_today"
LOCK_FILE="$PROJECT_DIR/.gemini_run.lock"
LOG_DIR="$PROJECT_DIR/logs"
SCHEDULER_LOG="$LOG_DIR/scheduler.log"

mkdir -p "$LOG_DIR"

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') $1" >> "$SCHEDULER_LOG"
}

# ---------------------------------------------------------------------
# cron runs with a minimal PATH (often just /usr/bin:/bin) and does NOT
# inherit your interactive shell's environment. Both `mvn` and `java`
# commonly live outside that minimal PATH on macOS (Homebrew: /opt/
# homebrew/bin or /usr/local/bin; SDKMAN/jenv/etc. add their own dirs).
# Extend PATH defensively rather than assuming cron can find them.
# VERIFY these are actually where your tools live: run `which mvn` and
# `which java` in a normal terminal and adjust if your paths differ.
# ---------------------------------------------------------------------
export PATH="/opt/homebrew/bin:/usr/local/bin:$HOME/.sdkman/candidates/java/current/bin:$PATH"

# ---------------------------------------------------------------------
# GOOGLE_API_KEY: cron does NOT see variables exported in your .zshrc/
# .bash_profile. Keep the real key OUT of this script (and out of git)
# in a separate, permission-locked file this script sources instead.
# Create it once with:
#   echo 'export GOOGLE_API_KEY=your-real-key-here' > "$HOME/.gemini_api_key.env"
#   chmod 600 "$HOME/.gemini_api_key.env"
# ---------------------------------------------------------------------
API_KEY_FILE="$HOME/.gemini_api_key.env"

if [ ! -f "$API_KEY_FILE" ]; then
    log "ERROR: $API_KEY_FILE not found. Create it (see comment in this script) before enabling cron."
    exit 1
fi
# shellcheck disable=SC1090
source "$API_KEY_FILE"

if [ -z "${GOOGLE_API_KEY:-}" ]; then
    log "ERROR: GOOGLE_API_KEY is empty after sourcing $API_KEY_FILE."
    exit 1
fi

# ---------------------------------------------------------------------
# The actual opt-in gate.
# ---------------------------------------------------------------------
if [ ! -f "$FLAG_FILE" ]; then
    log "No run-today flag at $FLAG_FILE -- skipping today."
    exit 0
fi

# ---------------------------------------------------------------------
# Simple lock so an overlapping trigger (e.g. machine woke late and cron
# fired twice close together) can't run two copies against the same
# checkpoint/workspace simultaneously.
# ---------------------------------------------------------------------
if [ -f "$LOCK_FILE" ]; then
    log "Lock file $LOCK_FILE already present -- a run may still be in progress, or a previous run crashed without cleaning up. Skipping. Delete the lock file manually if you're sure nothing is running."
    exit 0
fi
touch "$LOCK_FILE"
trap 'rm -f "$LOCK_FILE"' EXIT

# Consume the flag now, before running -- so if the run itself fails
# partway (crash, killed process, etc.) it doesn't silently retry on
# the next cron tick without you choosing to opt in again.
rm -f "$FLAG_FILE"

RUN_LOG="$LOG_DIR/gemini_run_$(date +%Y-%m-%d_%H%M%S).log"

log "Starting Gemini run (opted in via flag file). Output: $RUN_LOG"

cd "$PROJECT_DIR"
mvn compile exec:java -Dexec.mainClass=org.cas.agentgovernor.experiment.RunGeminiTaskComparison \
    > "$RUN_LOG" 2>&1
EXIT_CODE=$?

log "Finished with exit code $EXIT_CODE. See $RUN_LOG for full output."

exit $EXIT_CODE
