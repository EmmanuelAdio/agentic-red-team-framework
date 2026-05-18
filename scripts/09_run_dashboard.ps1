# Launch the Streamlit dashboard for the agentic red-team framework.
#
# PowerShell sibling of `09_run_dashboard.sh`.
#
# Honours the same environment variables as the bash launcher so the
# two stay in lock-step:
#
#   STREAMLIT_PORT             - TCP port to bind  (default 8501)
#   STREAMLIT_HEADLESS         - server.headless   (default true)
#   REDTEAM_DASHBOARD_THEME    - light (default) | dark - when "dark",
#                                the launcher *also* exports
#                                STREAMLIT_THEME_* vars so Streamlit's
#                                built-in widgets (st.dataframe, the
#                                sidebar, st.metric) flip to dark
#                                alongside our injected CSS. Without
#                                this, only our HTML helpers go dark
#                                and the dataframe / native widgets
#                                stay white-on-light.
#   REDTEAM_DASHBOARD_DUCKDB   - 1 to opt into the DuckDB backend.
#
# The launcher re-sources `.env` (at the repo root) on every
# invocation via python-dotenv. Streamlit does not re-read env vars
# on its in-browser "Rerun" - dashboard env knobs only take effect at
# server start - so editing `.env` requires relaunching this script.
# Pass `-Restart` to kill any process already listening on
# $STREAMLIT_PORT before launching, which is the standard
# "pick up my .env edits" workflow.
#
# Usage (from the repo root, with the project's venv active):
#
#   .\scripts\09_run_dashboard.ps1                # plain launch
#   .\scripts\09_run_dashboard.ps1 -Restart       # free the port first
#
# Or with one-shot environment overrides:
#
#   $env:STREAMLIT_PORT = "8600" ; .\scripts\09_run_dashboard.ps1
#   $env:REDTEAM_DASHBOARD_THEME = "dark" ; .\scripts\09_run_dashboard.ps1

[CmdletBinding()]
param(
    # When set, terminates any process holding the target TCP port
    # before invoking Streamlit. Use after editing `.env` so the new
    # values actually reach the new server process.
    [switch]$Restart
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

# -- Load `.env` via python-dotenv ------------------------------------
#
# Streamlit's `runOnSave = true` reruns the Python script on file
# save but does not re-read process env. dotenv_values() returns
# only the keys defined in the file (it does not include the
# inherited process env), so we explicitly skip empty values to
# preserve any shell-level overrides the user set before invoking
# this script (e.g. `$env:STREAMLIT_PORT = "8600" ; .\09...ps1`).
$dotenvPath = Join-Path $RepoRoot ".env"
if (Test-Path $dotenvPath) {
    Write-Host "Loading .env via python-dotenv: $dotenvPath"
    $pyExpr = @'
import json, sys
from dotenv import dotenv_values
sys.stdout.write(json.dumps(dotenv_values(sys.argv[1])))
'@
    $envJson = & python -c $pyExpr $dotenvPath
    if ($LASTEXITCODE -ne 0) {
        throw "python-dotenv failed to parse $dotenvPath (exit $LASTEXITCODE). Is the project venv active and python-dotenv installed?"
    }
    $envVars = $envJson | ConvertFrom-Json
    foreach ($prop in $envVars.PSObject.Properties) {
        # Skip empty values - `.env.example` ships keys without
        # values, and we don't want those wiping a shell-level
        # override.
        if ($null -ne $prop.Value -and "$($prop.Value)" -ne "") {
            Set-Item -Path "env:$($prop.Name)" -Value $prop.Value
        }
    }
}
else {
    Write-Host "No .env file at $dotenvPath - skipping dotenv load"
}

$port = if ($env:STREAMLIT_PORT) { $env:STREAMLIT_PORT } else { "8501" }
$headless = if ($env:STREAMLIT_HEADLESS) { $env:STREAMLIT_HEADLESS } else { "true" }

# -- Optional port restart --------------------------------------------
#
# Streamlit binds a fresh process per launch and does not re-read
# env vars on browser "Rerun". The -Restart switch frees the port
# so a new process picks up `.env` edits without a manual taskkill.
if ($Restart) {
    Write-Host "Restart requested - looking for listeners on port $port"
    try {
        $conns = Get-NetTCPConnection -LocalPort ([int]$port) -State Listen -ErrorAction Stop
    }
    catch {
        # Get-NetTCPConnection throws when nothing is listening;
        # that is the happy path for `-Restart` on a free port.
        $conns = @()
    }
    $pidsToKill = $conns | Select-Object -ExpandProperty OwningProcess -Unique
    if (-not $pidsToKill) {
        Write-Host "Port $port is already free."
    }
    foreach ($procId in $pidsToKill) {
        try {
            $proc = Get-Process -Id $procId -ErrorAction Stop
            Write-Host ("Stopping PID {0} ({1}) holding port {2}" -f $procId, $proc.ProcessName, $port)
            Stop-Process -Id $procId -Force -ErrorAction Stop
        }
        catch {
            Write-Warning ("Could not stop PID {0}: {1}" -f $procId, $_.Exception.Message)
        }
    }
}

# Clear any STREAMLIT_THEME_* env vars left over from a previous
# dark-mode launch in the same shell. Without this, toggling
# `.env` from dark back to light keeps the Streamlit native chrome
# (sidebar, dataframe, st.metric) dark, because Streamlit reads
# these env vars as a config override on top of
# dashboard/.streamlit/config.toml and once set they outlive the
# child process.
$streamlitThemeVars = @(
    "STREAMLIT_THEME_BASE",
    "STREAMLIT_THEME_PRIMARY_COLOR",
    "STREAMLIT_THEME_BACKGROUND_COLOR",
    "STREAMLIT_THEME_SECONDARY_BACKGROUND_COLOR",
    "STREAMLIT_THEME_TEXT_COLOR"
)
foreach ($name in $streamlitThemeVars) {
    if (Test-Path "env:$name") {
        Remove-Item "env:$name"
    }
}

# When dark mode is requested, propagate the colour tokens through to
# Streamlit's own theme so the dataframe widget + native chrome
# render dark too.
if ($env:REDTEAM_DASHBOARD_THEME -and $env:REDTEAM_DASHBOARD_THEME.ToLower() -eq "dark") {
    $env:STREAMLIT_THEME_BASE                          = "dark"
    $env:STREAMLIT_THEME_PRIMARY_COLOR                 = "#F0997B"
    $env:STREAMLIT_THEME_BACKGROUND_COLOR              = "#161614"
    $env:STREAMLIT_THEME_SECONDARY_BACKGROUND_COLOR    = "#1F1F1B"
    $env:STREAMLIT_THEME_TEXT_COLOR                    = "#F1EFE8"
    Write-Host "Launching in DARK mode (REDTEAM_DASHBOARD_THEME=dark)"
}
else {
    Write-Host "Launching in LIGHT mode (config.toml theme.base=light)"
}

streamlit run dashboard/Home.py `
    --server.port=$port `
    --server.headless=$headless `
    --browser.gatherUsageStats=false
