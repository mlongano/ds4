#!/bin/bash
# Launches ds4-server against the RAM disk by default. If RAM allocation
# (mount + copy) fails for any reason, falls back to the SSD-backed model
# with a warning instead of refusing to start. Extra args are forwarded
# to ds4-server (--ctx, --port, --ssd-streaming, etc).
set -uo pipefail

DS4_DIR=/media/NVME_DATA/MOUNTS/Models/ds4
LOCAL_ROCR_LIB="$DS4_DIR/misc/rocm-local-runtime-fixed-prefix/lib"
cd "$DS4_DIR"

if [[ ! -r "$LOCAL_ROCR_LIB/libhsa-runtime64.so.1" ]]; then
    echo "ds4-server-launch: validated project-local ROCr runtime is missing" >&2
    exit 1
fi
export LD_LIBRARY_PATH="$LOCAL_ROCR_LIB:/opt/rocm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HSA_ENABLE_SDMA="${HSA_ENABLE_SDMA:-1}"
echo "ds4-server-launch: using project-local ROCr runtime: $LOCAL_ROCR_LIB" >&2

if MODEL=$("$DS4_DIR/ram-disk-up.sh"); then
    echo "ds4-server-launch: using RAM-backed model" >&2
else
    STATUS=$?
    echo "ds4-server-launch: WARNING: RAM disk allocation failed (exit $STATUS)," \
         "falling back to SSD-backed streaming" >&2
    MODEL="$(readlink -f "$DS4_DIR/ds4flash.gguf")"
fi

exec "$DS4_DIR/ds4-server" -m "$MODEL" "$@"
