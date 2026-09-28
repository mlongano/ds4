set pagination off
python
import gdb
for t in gdb.selected_inferior().threads():
    if t.ptid[1] == 508921:
        t.switch()
        break
end
info registers rip rsp rbp rbx r12 r13 r14 r15
x/64gx $rsp
set $p=$rbp
x/2gx $p
set $p=*(void**)$p
x/2gx $p
set $p=*(void**)$p
x/2gx $p
set $p=*(void**)$p
x/2gx $p
set $p=*(void**)$p
x/2gx $p
set $p=*(void**)$p
x/2gx $p
set $p=*(void**)$p
x/2gx $p
set $p=*(void**)$p
x/2gx $p
