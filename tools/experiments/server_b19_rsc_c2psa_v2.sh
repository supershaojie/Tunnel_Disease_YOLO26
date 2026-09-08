#!/usr/bin/env bash
# Fixed RSC v2 identity; the shared lifecycle locks only this experiment.
set -euo pipefail
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/server_b19_rsc_c2psa.sh" 2 "$@"
