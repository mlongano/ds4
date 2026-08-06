#!/bin/bash
# Idempotently release the /mnt/ram tmpfs, freeing its RAM.
# Refuses if any process still has a file on it open.
set -euo pipefail

RAM_DIR=/mnt/ram

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
