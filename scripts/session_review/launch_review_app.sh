#!/usr/bin/env bash
# Start the FIVRS session review app (dock icon, or from a terminal).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
unset QT_PLUGIN_PATH QT_QPA_PLATFORM_PLUGIN_PATH QT_QPA_PLATFORMTHEME PYTHONPATH
exec /usr/bin/python3 "${SCRIPT_DIR}/review_app.py" "$@"
