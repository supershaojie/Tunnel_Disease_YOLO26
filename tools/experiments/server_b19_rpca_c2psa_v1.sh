#!/usr/bin/env bash
# Invoke in an interactive tmux Bash; every stage uses this experiment's own lock and output.
EXPERIMENT_NAME=yolo26n_b19_e1_rpca_c2psa_v1
EXPERIMENT_ENTRY=rpca_c2psa
source "$(dirname -- "${BASH_SOURCE[0]}")/server_b19_sir_sppf_v2.sh" "$@"
