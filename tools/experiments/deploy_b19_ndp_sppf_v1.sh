#!/usr/bin/env bash
# Copyable bootstrap; a caller supplies a verified full commit, never a moving branch as the revision.
set -euo pipefail
SHA="${1:?Supply the verified full 40-character commit from the delivery record}"
[[ "$SHA" =~ ^[0-9a-f]{40}$ ]] || { echo 'Expected a full SHA' >&2; exit 2; }
BASE=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26
WORK=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26_NDP_SPPF_v1
BRANCH=codex/exp-yolo26n-b19-ndp-sppf-v1
NAME=yolo26n_b19_ndp_sppf_v1
EXPECTED=https://github.com/supershaojie/Tunnel_Disease_YOLO26.git
[[ "$(git -C "$BASE" remote get-url origin)" == "$EXPECTED" ]] || {
    git -C "$BASE" remote -v; echo 'Origin mismatch; stopped without modifying remotes' >&2; exit 3;
}
mkdir -p -- "$BASE/.experiment-locks"
exec 9>"$BASE/.experiment-locks/$NAME.lock"
flock -n 9 || { echo 'This experiment is active; deployment refused' >&2; exit 3; }
# Also detect orphaned/direct Python stages whose launching shell no longer holds the lock.
command -v pgrep >/dev/null
if pgrep -af -- "$WORK/tools/experiments/(run|finish)_b19_ndp_sppf_v1[.]py"; then
    echo 'An experiment Python stage is still present; stopped for inspection.' >&2
    exit 3
fi
if ! git -C "$BASE" cat-file -e "$SHA^{commit}" 2>/dev/null; then
    git -C "$BASE" -c http.version=HTTP/1.1 -c http.lowSpeedLimit=1024 -c http.lowSpeedTime=120 \
        fetch --progress origin "$BRANCH"
fi
git -C "$BASE" cat-file -e "$SHA^{commit}"
git -C "$BASE" merge-base --is-ancestor 4a1b167dee2b0c7353d9d7ead2911383d1ebf1d6 "$SHA"
if [[ -e "$WORK" ]]; then
    [[ -f "$WORK/.git" ]] || { echo "Existing path is not the expected linked worktree: $WORK" >&2; exit 3; }
    ACTUAL_SHA="$(git -C "$WORK" rev-parse HEAD)"
    STATE="$(git -C "$WORK" status --porcelain --untracked-files=normal)"
    if [[ "$ACTUAL_SHA" != "$SHA" || -n "$STATE" ]]; then
        printf 'Expected %s; actual %s\n%s\n' "$SHA" "$ACTUAL_SHA" "$STATE"
        echo 'Stopped; preserve existing work and inspect before any explicit update.' >&2; exit 3
    fi
else
    git -C "$BASE" worktree add --detach "$WORK" "$SHA"
fi
[[ "$(git -C "$WORK" rev-parse HEAD)" == "$SHA" ]]
cd -- "$WORK"
B19_PYTHON="${B19_PYTHON:-/root/miniconda3/bin/python}"
PYTHONPATH="$WORK" "$B19_PYTHON" -c 'from pathlib import Path; import ultralytics; assert Path(ultralytics.__file__).resolve().parent == Path.cwd()/"ultralytics"; print(ultralytics.__file__)'
printf 'Verified deployment: %s at %s\n' "$WORK" "$SHA"
