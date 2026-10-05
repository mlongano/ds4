#!/bin/bash
cd /media/NVME_DATA/MOUNTS/Models/ds4
PROMPT="Explain why the sky is blue in two sentences."

echo "============================================================="
echo " ds4 Context Size Benchmark (CLI -p mode, cold streaming)"
echo "============================================================="
printf "%-8s %10s %8s\n" "Context" "Prefill" "Gen"
echo "-------------------------------------------------------------"

for CTX in 4096 8192 16384 32768 65536; do
    rm -f /tmp/ds4.lock
    echo -n "  ctx=${CTX}... " >&2

    OUTPUT=$(DS4_ROCM_STREAM_FREE_RESERVE_GB=2 ./ds4 -p "$PROMPT" \
        --ctx $CTX --ssd-streaming --ssd-streaming-cold 2>&1) || true

    PF=$(echo "$OUTPUT" | sed -n 's/.*prefill: \([0-9.]*\) t\/s.*/\1/p' | tail -1)
    GEN=$(echo "$OUTPUT" | sed -n 's/.*generation: \([0-9.]*\) t\/s.*/\1/p' | tail -1)
    REPLY=$(echo "$OUTPUT" | tail -3 | head -1)

    printf "%-8s %8s t/s %6s t/s\n" "$CTX" "${PF:-?}" "${GEN:-?}"
    echo "    ${REPLY:0:100}" >&2
    sleep 2
done

rm -f /tmp/ds4.lock
echo "============================================================="
