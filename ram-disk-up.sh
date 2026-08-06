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
    # Report copy progress (bytes written to the .tmp file so far) to a
    # small file that /v1/transition reads, so pi-inference's live progress
    # output shows a real percentage instead of a static "waiting" line for
    # the ~40s this copy takes.
    (
        while sleep 1; do
            [ -f "$DST.tmp" ] || continue
            CUR=$(stat -c%s "$DST.tmp" 2>/dev/null) || continue
            PCT=$(( CUR * 100 / SRC_SIZE ))
            printf '%d%% (%d/%d GiB)\n' "$PCT" "$(( CUR / 1073741824 ))" "$(( SRC_SIZE / 1073741824 ))" > "$PROGRESS_FILE"
        done
    ) &
    PROGRESS_PID=$!
    cp --sparse=never "$SRC" "$DST.tmp"
    kill "$PROGRESS_PID" 2>/dev/null || true
    wait "$PROGRESS_PID" 2>/dev/null || true
    PROGRESS_PID=""
    rm -f "$PROGRESS_FILE"
    mv "$DST.tmp" "$DST"
fi

trap - ERR
echo "ds4-ram: ready at $DST" >&2
echo "$DST"
