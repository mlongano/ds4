#!/bin/bash
# Idempotently mount a tmpfs at /mnt/ram and copy ds4flash.gguf into it.
# Prints the RAM-backed model path on stdout.
set -euo pipefail

RAM_DIR=/mnt/ram
RAM_SIZE=90G
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$(readlink -f "$SCRIPT_DIR/ds4flash.gguf")"
DST="$RAM_DIR/ds4flash.gguf"
PROGRESS_FILE="${XDG_RUNTIME_DIR:-/tmp}/ds4-ram-copy.progress"
PROGRESS_PID=""

if [ ! -f "$SRC" ]; then
    echo "ds4-ram: source model not found: $SRC" >&2
    exit 1
fi

# Serialize concurrent callers (e.g. pi-inference starting ds4-server.service
# at the same moment ds4-chat is independently allocating the RAM disk):
# without this, two invocations racing to cp the same 80GB into the same
# .tmp path can interleave writes and corrupt it, or one's mv can land
# mid-copy under the other. A blocked second caller just waits here, then
# finds the file already correctly sized below and skips straight past.
LOCK_FILE="${XDG_RUNTIME_DIR:-/tmp}/ds4-ram-disk.lock"
exec 200>"$LOCK_FILE"
flock 200

MOUNTED_BY_US=0
if ! mountpoint -q "$RAM_DIR"; then
    echo "ds4-ram: mounting tmpfs at $RAM_DIR ($RAM_SIZE)" >&2
    sudo mount -t tmpfs -o size=$RAM_SIZE,mode=1777 tmpfs "$RAM_DIR"
    MOUNTED_BY_US=1
fi

cleanup_on_failure() {
    if [ -n "$PROGRESS_PID" ]; then
        kill "$PROGRESS_PID" 2>/dev/null || true
        wait "$PROGRESS_PID" 2>/dev/null || true
    fi
    rm -f "$DST.tmp" "$PROGRESS_FILE"
    if [ "$MOUNTED_BY_US" = 1 ]; then
        sudo umount "$RAM_DIR" 2>/dev/null || true
    fi
}
trap cleanup_on_failure ERR

SRC_SIZE=$(stat -c%s "$SRC")
if [ ! -f "$DST" ] || [ "$(stat -c%s "$DST")" != "$SRC_SIZE" ]; then
    echo "ds4-ram: copying model to RAM ($(( SRC_SIZE / 1073741824 )) GiB)" >&2
    # Report copy progress (bytes written to the .tmp file so far) two ways:
    # to a small file that /v1/transition reads (for pi-inference's live
    # progress output), and directly to our own stderr as a single updating
    # line (for direct callers like ds4-chat that never go through
    # pi-inference at all).
    (
        while sleep 1; do
            [ -f "$DST.tmp" ] || continue
            CUR=$(stat -c%s "$DST.tmp" 2>/dev/null) || continue
            PCT=$(( CUR * 100 / SRC_SIZE ))
            printf '%d%% (%d/%d GiB)\n' "$PCT" "$(( CUR / 1073741824 ))" "$(( SRC_SIZE / 1073741824 ))" > "$PROGRESS_FILE"
            printf '\rds4-ram: copying model to RAM: %d%% (%d/%d GiB)...' \
                "$PCT" "$(( CUR / 1073741824 ))" "$(( SRC_SIZE / 1073741824 ))" >&2
        done
    ) &
    PROGRESS_PID=$!
    cp --sparse=never "$SRC" "$DST.tmp"
    kill "$PROGRESS_PID" 2>/dev/null || true
    wait "$PROGRESS_PID" 2>/dev/null || true
    PROGRESS_PID=""
    rm -f "$PROGRESS_FILE"
    printf '\n' >&2
    mv "$DST.tmp" "$DST"
fi

trap - ERR
echo "ds4-ram: ready at $DST" >&2
echo "$DST"
