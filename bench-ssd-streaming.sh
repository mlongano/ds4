#!/bin/bash
set -euo pipefail
cd /media/NVME_DATA/MOUNTS/Models/ds4
PORT=8080
PROMPT="Why is the sky blue? Keep it short."

echo "============================================"
echo " ds4 SSD Streaming Benchmark (65K context)"
echo "============================================"
printf "%-14s %8s  %10s  %8s\n" "Mode" "Wall" "Prefill" "Gen"
echo "--------------------------------------------"

for MODE in cold warm_64 warm_128 warm_256; do
    rm -f /tmp/ds4.lock

    case $MODE in
        cold)     EXTRA="--ssd-streaming-cold" ;;
        warm_64)  EXTRA="--ssd-streaming-preload-experts 64" ;;
        warm_128) EXTRA="--ssd-streaming-preload-experts 128" ;;
        warm_256) EXTRA="--ssd-streaming-preload-experts 256" ;;
    esac

    echo -n "  $MODE... " >&2

    DS4_ROCM_STREAM_FREE_RESERVE_GB=2 ./ds4-server --ctx 65536 --port $PORT \
        --ssd-streaming $EXTRA > /tmp/ds4-bench.log 2>&1 &
    PID=$!

    READY=false
    for i in $(seq 1 90); do
        sleep 2
        if curl -sSf "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then
            READY=true; break
        fi
        if ! kill -0 $PID 2>/dev/null; then
            echo "CRASHED" >&2
            tail -3 /tmp/ds4-bench.log >&2
            break
        fi
    done

    if ! $READY; then
        kill $PID 2>/dev/null; wait $PID 2>/dev/null
        printf "%-14s %8s  %10s  %8s\n" "$MODE" "-" "-" "-"
        sleep 3; continue
    fi

    # Warm-up
    curl -s --max-time 180 "http://localhost:$PORT/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":5,\"temperature\":0}" > /dev/null 2>&1 || true

    # Timed benchmark
    START=$(date +%s%3N)
    BODY=$(curl -s --max-time 180 "http://localhost:$PORT/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"max_tokens\":128,\"temperature\":0}" 2>/dev/null)
    END=$(date +%s%3N)
    WALL=$(( (END - START) / 1000 ))

    STATS=$(grep 'prefill:' /tmp/ds4-bench.log | tail -1)
    PF=$(echo "$STATS" | sed -n 's/.*prefill: \([0-9.]*\).*/\1/p')
    GEN=$(echo "$STATS" | sed -n 's/.*generation: \([0-9.]*\).*/\1/p')

    CONTENT=$(echo "$BODY" | python3 -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['message']['content'].strip())" 2>/dev/null || echo "?")
    echo "  reply: ${CONTENT:0:80}..." >&2

    printf "%-14s %5ss  %8s t/s  %6s t/s\n" "$MODE" "${WALL}" "${PF:-?}" "${GEN:-?}"

    kill $PID 2>/dev/null; wait $PID 2>/dev/null
    sleep 3
done

rm -f /tmp/ds4-bench.log /tmp/ds4.lock
echo "============================================"
