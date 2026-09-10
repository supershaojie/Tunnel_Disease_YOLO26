#!/usr/bin/env bash
# Supply the verified full commit from the release handoff as the sole argument.
set -euo pipefail
SHA="${1:?Pass the verified full 40-character commit SHA}"
[[ "$SHA" =~ ^[0-9a-f]{40}$ ]] || exit 2
BASE=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26
WORK=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26_CCA_Fusion_v2
BRANCH=codex/exp-yolo26n-b19-cca-fusion-v2
REMOTE=https://github.com/supershaojie/Tunnel_Disease_YOLO26.git
exec 9>"$BASE/.yolo26n_b19_cca_fusion_v2.lock"
flock -n 9 || { echo 'CCA stage or deployment active; stopping.' >&2; exit 3; }
test "$(git -C "$BASE" remote get-url origin)" = "$REMOTE"
if ! git -C "$BASE" cat-file -e "$SHA^{commit}" 2>/dev/null; then
    GIT_TERMINAL_PROMPT=0 git -C "$BASE" -c http.version=HTTP/1.1 \
        -c http.lowSpeedLimit=1 -c http.lowSpeedTime=60 \
        fetch --progress --no-tags origin "$BRANCH"
fi
git -C "$BASE" cat-file -e "$SHA^{commit}"
if [[ -e "$WORK" ]]; then
    test "$(git -C "$WORK" rev-parse HEAD)" = "$SHA"
    test -z "$(git -C "$WORK" status --porcelain --untracked-files=no)"
    if git -C "$WORK" symbolic-ref -q HEAD; then
        echo 'Existing checkout is not detached; preserving it.' >&2; exit 3
    fi
    # Inspect even processes started outside the supported locking entry.
    if pgrep -af '[p]ython.*(run_b19_cca_fusion|finish_b19_cca_fusion)' ; then
        echo 'Existing CCA Python process; preserving checkout.' >&2; exit 3
    fi
    git -C "$WORK" status --short
else
    git -C "$BASE" worktree add --detach "$WORK" "$SHA"
fi
test -f "$WORK/tools/experiments/server_b19_cca_fusion_v2.sh"
bash -n "$WORK/tools/experiments/server_b19_cca_fusion_v2.sh"
test "$(git -C "$WORK" rev-parse HEAD)" = "$SHA"
printf 'Deployed detached commit %s at %s\n' "$SHA" "$WORK"
