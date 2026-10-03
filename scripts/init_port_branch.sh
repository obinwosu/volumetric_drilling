#!/usr/bin/env bash
# Point this working tree at upstream/master as branch cursor/ambf3-camera-nav-2b83
# WITHOUT discarding the already-applied port files, then stage everything for commit.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

command -v git >/dev/null || { echo "Install git first: sudo apt-get install -y git gh"; exit 1; }

if [[ ! -d .git ]]; then
  git init
fi

git remote get-url origin >/dev/null 2>&1 || \
  git remote add origin https://github.com/obinwosu/volumetric_drilling_VRE.git
git remote get-url upstream >/dev/null 2>&1 || \
  git remote add upstream https://github.com/LCSR-SICKKIDS/volumetric_drilling.git

git fetch upstream master

# Move HEAD to a new branch at upstream/master without touching the working tree.
git update-ref refs/heads/cursor/ambf3-camera-nav-2b83 refs/remotes/upstream/master
git symbolic-ref HEAD refs/heads/cursor/ambf3-camera-nav-2b83
# Index matches upstream; working tree (our port) stays intact → shows as modifications.
git reset

git add -A
echo "Branch: $(git branch --show-current)"
echo "Base:   $(git rev-parse --short HEAD) (upstream/master)"
echo
git status --short | head -100
echo
echo "Next:"
echo "  git commit -m \"Port camera_navigation features onto AMBF3 volumetric_drilling base\""
echo "  git push -u origin cursor/ambf3-camera-nav-2b83"
echo "  gh pr create --repo obinwosu/volumetric_drilling_VRE --base master --head cursor/ambf3-camera-nav-2b83 \\"
echo "    --title \"AMBF3 port of camera_navigation\" --body-file docs/AMBF3_PORT_CONFLICTS.md"
