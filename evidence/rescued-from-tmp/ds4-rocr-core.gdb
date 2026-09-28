set pagination off
set debuginfod enabled on
python
import gdb
for t in gdb.selected_inferior().threads():
    if t.ptid[1] in (508921, 3012274):
        t.switch()
        print("SELECTED_LWP", t.ptid[1], "GDB_THREAD", t.num)
        break
end
bt 12
frame 5
info args
info locals
p/x hw_read_index
p/x commit
p/x this->cached_commit_index_
p/x this->cached_reserve_index_
p/x this->bytes_queued_
p this->bytes_written_.data_.size()
p/x this->bytes_written_.data_[0]
p/x this->bytes_written_.data_[this->bytes_written_.data_.size()-1]
