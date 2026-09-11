#!/usr/bin/env bash
# Deploy one user-verified full commit; this entry never starts training.
set -euo pipefail
SHA="${1:?Pass the full commit SHA from the implementation report}"
[[ "$SHA" =~ ^[0-9a-f]{40}$ ]] || { printf 'Invalid SHA: <%s> length=%s\n' "$SHA" "${#SHA}" >&2; exit 2; }
BASE=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26
WORK=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26_MGC_SPPF_v1
BRANCH=codex/exp-yolo26n-b19-mgc-sppf-v1
REMOTE=https://github.com/supershaojie/Tunnel_Disease_YOLO26.git
exec 9>"$BASE/.yolo26n_b19_mgc_sppf_v1.lock"
flock -n 9 || { echo 'MGC stage/deployment already active.' >&2; exit 3; }
test "$(git -C "$BASE" remote get-url origin)" = "$REMOTE"
if pgrep -af '[p]ython.*(run_b19_mgc_sppf|verify_b19_mgc_sppf|finish_b19_mgc_sppf|diagnose_b19_mgc_sppf)'; then
    echo 'Active MGC process; preserving code.' >&2; exit 3
fi
for process in /proc/[0-9]*; do
    if [[ "$(readlink "$process/cwd" 2>/dev/null || true)" == "$WORK" ]]; then
        exe="$(readlink "$process/exe" 2>/dev/null || true)"
        if [[ "$exe" == *python* ]]; then
            printf 'Active Python in WORK: %s\n' "$process" >&2; exit 3
        fi
    fi
done
compare_sha() {
    local actual="$1" position
    if [[ "$actual" != "$SHA" ]]; then
        printf 'expected=<%s> length=%s\nactual=<%s> length=%s\n' "$SHA" "${#SHA}" "$actual" "${#actual}" >&2
        for ((position=0; position<${#SHA} && position<${#actual}; position++)); do
            [[ "${SHA:position:1}" == "${actual:position:1}" ]] || break
        done
        printf 'First difference at zero-based position %s\n' "$position" >&2
        exit 4
    fi
}
if ! git -C "$BASE" cat-file -e "$SHA^{commit}" 2>/dev/null; then
    # One bounded fetch; failed fetch exits before FETCH_HEAD is ever read.
    timeout --kill-after=5s 120s env GIT_TERMINAL_PROMPT=0 git -C "$BASE" -c http.version=HTTP/1.1 \
        -c http.lowSpeedLimit=1 -c http.lowSpeedTime=30 fetch --no-tags origin "$BRANCH"
    compare_sha "$(git -C "$BASE" rev-parse FETCH_HEAD)"
else
    printf 'Verified target object already local; no fetch needed.\n'
fi
git -C "$BASE" cat-file -e "$SHA^{commit}"
git -C "$BASE" merge-base --is-ancestor 4a1b167dee2b0c7353d9d7ead2911383d1ebf1d6 "$SHA"
git -C "$BASE" cat-file -e "$SHA:tools/experiments/server_b19_mgc_sppf_v1.sh"
if [[ -e "$WORK" ]]; then
    test "$(git -C "$WORK" remote get-url origin)" = "$REMOTE"
    test -z "$(git -C "$WORK" status --porcelain)"
    if git -C "$WORK" symbolic-ref -q HEAD; then
        echo 'Existing WORK is attached to a branch; preserving it.' >&2; exit 3
    fi
    git -C "$WORK" checkout --detach "$SHA"
else
    git -C "$BASE" worktree add --detach "$WORK" "$SHA"
fi
compare_sha "$(git -C "$WORK" rev-parse HEAD)"
bash -n "$WORK/tools/experiments/server_b19_mgc_sppf_v1.sh"
printf 'Deployed %s to %s. Formal training NOT STARTED.\n' "$SHA" "$WORK"
