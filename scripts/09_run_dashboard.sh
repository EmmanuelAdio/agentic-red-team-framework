#!/usr/bin/env bash
# Launch the Streamlit dashboard for the agentic red-team framework.
# Numbered 09 because scripts/01..08 are already taken.
#
# When REDTEAM_DASHBOARD_THEME=dark is set, the launcher *also*
# exports STREAMLIT_THEME_* vars so Streamlit's built-in widgets
# (st.dataframe, sidebar, st.metric) flip to dark alongside our
# injected CSS. Without this, only our HTML helpers go dark and the
# dataframe / native widgets stay white-on-light.
#
# The launcher re-sources `.env` (at the repo root) on every
# invocation via python-dotenv. Streamlit does not re-read env vars
# on its in-browser "Rerun" - dashboard env knobs only take effect at
# server start - so editing `.env` requires relaunching this script.
# Pass `--restart` to kill any process already listening on
# $STREAMLIT_PORT before launching, which is the standard
# "pick up my .env edits" workflow.
#
# Usage:
#   bash scripts/09_run_dashboard.sh              # plain launch
#   bash scripts/09_run_dashboard.sh --restart    # free the port first

set -euo pipefail

RESTART=0
for arg in "$@"; do
    case "$arg" in
        --restart) RESTART=1 ;;
        -h|--help)
            sed -n '2,22p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg" >&2
            exit 2
            ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# -- Load `.env` via python-dotenv ------------------------------------
#
# dotenv_values() returns only the keys defined in the file, so we
# skip empty values to preserve any shell-level overrides the caller
# set before invoking this script.
DOTENV_PATH="$REPO_ROOT/.env"
if [ -f "$DOTENV_PATH" ]; then
    echo "Loading .env via python-dotenv: $DOTENV_PATH"
    # Emit `export KEY='value'` lines for non-empty entries. Single
    # quotes are escaped via the standard sh trick: '\''
    eval "$(python - "$DOTENV_PATH" <<'PYEOF'
import sys
from dotenv import dotenv_values

values = dotenv_values(sys.argv[1])
for key, value in values.items():
    if value is None or value == "":
        continue
    escaped = value.replace("'", "'\\''")
    print(f"export {key}='{escaped}'")
PYEOF
)"
else
    echo "No .env file at $DOTENV_PATH - skipping dotenv load"
fi

PORT="${STREAMLIT_PORT:-8501}"
HEADLESS="${STREAMLIT_HEADLESS:-true}"

# -- Optional port restart --------------------------------------------
#
# Streamlit binds a fresh process per launch and does not re-read
# env vars on browser "Rerun". --restart frees the port so a new
# process picks up `.env` edits without a manual kill.
if [ "$RESTART" = "1" ]; then
    echo "Restart requested - looking for listeners on port $PORT"
    PIDS=""
    if command -v lsof >/dev/null 2>&1; then
        PIDS="$(lsof -ti TCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true)"
    elif command -v fuser >/dev/null 2>&1; then
        PIDS="$(fuser -n tcp "$PORT" 2>/dev/null | tr -d ':' || true)"
    elif command -v ss >/dev/null 2>&1; then
        # ss output: users:(("python",pid=1234,fd=5))
        PIDS="$(ss -ltnp "sport = :$PORT" 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 || true)"
    else
        echo "warning: no lsof/fuser/ss available - skipping restart" >&2
    fi

    if [ -z "$PIDS" ]; then
        echo "Port $PORT is already free."
    else
        for pid in $PIDS; do
            echo "Stopping PID $pid holding port $PORT"
            kill "$pid" 2>/dev/null || true
        done
        # Give graceful kills a moment, then escalate any survivors.
        sleep 1
        for pid in $PIDS; do
            if kill -0 "$pid" 2>/dev/null; then
                echo "PID $pid still alive - sending SIGKILL"
                kill -9 "$pid" 2>/dev/null || true
            fi
        done
    fi
fi

# Clear any STREAMLIT_THEME_* env vars left over from a previous
# dark-mode launch in the same shell. Without this, toggling
# REDTEAM_DASHBOARD_THEME from dark back to light keeps the
# Streamlit native chrome dark because the env vars override
# dashboard/.streamlit/config.toml and outlive the child process.
unset STREAMLIT_THEME_BASE \
      STREAMLIT_THEME_PRIMARY_COLOR \
      STREAMLIT_THEME_BACKGROUND_COLOR \
      STREAMLIT_THEME_SECONDARY_BACKGROUND_COLOR \
      STREAMLIT_THEME_TEXT_COLOR

if [ "${REDTEAM_DASHBOARD_THEME:-light}" = "dark" ]; then
    export STREAMLIT_THEME_BASE="dark"
    export STREAMLIT_THEME_PRIMARY_COLOR="#F0997B"
    export STREAMLIT_THEME_BACKGROUND_COLOR="#161614"
    export STREAMLIT_THEME_SECONDARY_BACKGROUND_COLOR="#1F1F1B"
    export STREAMLIT_THEME_TEXT_COLOR="#F1EFE8"
    echo "Launching in DARK mode (REDTEAM_DASHBOARD_THEME=dark)"
else
    echo "Launching in LIGHT mode (config.toml theme.base=light)"
fi

exec streamlit run dashboard/Home.py \
    --server.port="$PORT" \
    --server.headless="$HEADLESS" \
    --browser.gatherUsageStats=false
