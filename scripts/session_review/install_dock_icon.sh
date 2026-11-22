#!/usr/bin/env bash
# Add FIVRS Session Review to this user's applications and pin it to the dock, next to FIVRS.
# Paths come from where this checkout is, so it also works on the lab machine. Safe to re-run.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APPLICATIONS="${HOME}/.local/share/applications"
ENTRY="fivrs-session-review.desktop"
mkdir -p "${APPLICATIONS}"
chmod +x "${SCRIPT_DIR}/launch_review_app.sh"

cat > "${APPLICATIONS}/${ENTRY}" <<EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=FIVRS Session Review
GenericName=Drilling session review
Comment=Replay recorded UR5 drilling sessions: video, curves and 3D reconstruction
Exec=${SCRIPT_DIR}/launch_review_app.sh
Path=${SCRIPT_DIR}
Icon=${SCRIPT_DIR}/assets/session_review_icon.png
Terminal=false
StartupNotify=true
Categories=Education;Science;
Keywords=FIVRS;UR5;review;replay;drilling;session;
StartupWMClass=FIVRS Session Review
EOF
chmod +x "${APPLICATIONS}/${ENTRY}"
update-desktop-database "${APPLICATIONS}" 2>/dev/null || true

if command -v gsettings >/dev/null; then
    current="$(gsettings get org.gnome.shell favorite-apps)"
    updated="$(python3 -c '
import ast, sys
apps = ast.literal_eval(sys.argv[1].replace("@as ", ""))
entry = sys.argv[2]
if entry not in apps:
    after = "volumetric-drilling-study-gui.desktop"
    apps.insert(apps.index(after) + 1 if after in apps else len(apps), entry)
print(apps)
' "${current}" "${ENTRY}")"
    gsettings set org.gnome.shell favorite-apps "${updated}"
fi
echo "FIVRS Session Review is in the dock."
