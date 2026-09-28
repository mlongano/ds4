set pagination off
python
import gdb
for t in gdb.selected_inferior().threads():
    if t.ptid[1] == 3012274:
        t.switch()
        break
end
printf "registers:\n"
info registers r12 r13 r14 r15 rbp rsp rip
printf "wrapper="
p/x (void*)$r12
printf "wrapper memory:\n"
x/8gx $r12
printf "blit="
p/x *(void**)$r12
printf "blit header/state:\n"
x/24gx *(void**)$r12
printf "bytes_queued="
p/x *(unsigned long long*)(*(char**)$r12+0x40)
printf "vector_begin="
p/x *(void**)(*(char**)$r12+0x48)
printf "vector_end="
p/x *(void**)(*(char**)$r12+0x50)
printf "queue_read_ptr="
p/x *(void**)(*(char**)$r12+0x94)
printf "hw_read_after_abort="
p/x **(unsigned long long**)(*(char**)$r12+0x94)
printf "cached_reserve="
p/x *(unsigned long long*)(*(char**)$r12+0xa8)
printf "cached_commit="
p/x *(unsigned long long*)(*(char**)$r12+0xb0)
printf "tracker_first="
p/x **(unsigned long long**)(*(char**)$r12+0x48)
printf "tracker_last="
p/x *(*(unsigned long long**)(*(char**)$r12+0x50)-1)
