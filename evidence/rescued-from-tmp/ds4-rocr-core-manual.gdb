set pagination off
python
import gdb
for t in gdb.selected_inferior().threads():
    if t.ptid[1] == 508921:
        t.switch()
        break
end
add-symbol-file /home/mauro/.cache/debuginfod_client/2e2d3183cee7ff4087f70e26daa249261f2092b1/debuginfo 0x00007fa66080f040
set $pc=0x00007fa6608366cc
set $rbp=0x00007f908f7f6200
set $rsp=0x00007f908f7f61f8
info line *$pc
info registers rax rbx rcx rdx rsi rdi r8 r9 r10 r11 r12 r13 r14 r15 rbp rsp rip
info args
info locals
p/x hw_read_index
p/x commit
p/x this
p/x this->cached_commit_index_
p/x this->cached_reserve_index_
p/x this->bytes_queued_
p this->bytes_written_.data_.size()
p/x this->bytes_written_.data_[0]
p/x this->bytes_written_.data_[this->bytes_written_.data_.size()-1]
