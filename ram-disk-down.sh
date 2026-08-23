#!/bin/bash
# Idempotently release the /mnt/ram tmpfs, freeing its RAM.
# Refuses if any process still has a file on it open.
set -euo pipefail

RAM_DIR=/mnt/ram
# One-shot opt-out, written by the inference panel's "Free the GPU" action.
# Stopping ds4-server releases VRAM; it has no reason to also drop 81 GiB of
# host RAM, and re-copying the model costs 40-50s on the way back. When the
# marker is present this stop keeps the tmpfs -- and consumes the marker, so
# a stale one can only ever skip a single teardown. The panel's dedicated
# "Release RAM disk" button runs this script with no marker.
KEEP_MARKER="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/inference-panel/keep-ramdisk"

if [ -e "$KEEP_MARKER" ]; then
    rm -f "$KEEP_MARKER"
    echo "ds4-ram: keeping $RAM_DIR mounted (requested by the inference panel)" >&2
    exit 0
fi

if ! mountpoint -q "$RAM_DIR"; then
    echo "ds4-ram: nothing to release, $RAM_DIR is not mounted" >&2
    exit 0
fi

if sudo fuser -m "$RAM_DIR" >/dev/null 2>&1; then
    echo "ds4-ram: refusing to unmount, $RAM_DIR is still in use:" >&2
    sudo fuser -mv "$RAM_DIR" >&2 || true
    exit 1
fi

echo "ds4-ram: unmounting $RAM_DIR, freeing RAM" >&2
sudo umount "$RAM_DIR"
