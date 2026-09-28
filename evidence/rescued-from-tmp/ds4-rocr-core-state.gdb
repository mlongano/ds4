set pagination off
python
import gdb
for t in gdb.selected_inferior().threads():
    if t.ptid[1] == 508921:
        t.switch()
        break
end
printf "wrapper="
p/x (void*)$r12
printf "wrapper memory:\n"
x/8gx $r12
printf "blit="
p/x *(void**)$r12
printf "blit header/state:\n"
x/28gx *(void**)$r12
printf "bytes_queued="
p/x *(unsigned long long*)(*(char**)$r12+0x40)
printf "vector_begin="
p/x *(void**)(*(char**)$r12+0x48)
printf "vector_end="
p/x *(void**)(*(char**)$r12+0x50)
printf "vector_capacity="
p/x *(void**)(*(char**)$r12+0x58)
printf "queue_read_ptr="
p/x *(void**)(*(char**)$r12+0x94)
printf "hw_read="
p/x **(unsigned long long**)(*(char**)$r12+0x94)
printf "cached_commit="
p/x *(unsigned long long*)(*(char**)$r12+0xb0)
printf "cached_reserve="
p/x *(unsigned long long*)(*(char**)$r12+0xb8)
printf "tracker_first="
p/x **(unsigned long long**)(*(char**)$r12+0x48)
printf "tracker_last="
p/x *(*(unsigned long long**)(*(char**)$r12+0x50)-1)
