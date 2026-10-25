# Buffer lưu cạnh checkpoint trên máy local

Checkpoint chứa trọng số, optimizer state, cấu hình và số vòng đã chạy. Buffer tái chơi **không** nằm trong checkpoint: nó được ghi thành `buffer_last.npz` cạnh checkpoint, chỉ trên đĩa local, và `--resume` nạp lại tệp đó.

Lý do: nhét 36 MB buffer vào `last.pt` làm mỗi lần ghi checkpoint trên Google Drive chậm đi rõ rệt, trong khi không lưu buffer thì việc mở rộng ngân sách 1000 → 1200/1500 ván sẽ bắt đầu bằng một buffer rỗng.

## Considered Options

- **Nhét buffer vào `last.pt`**: một tệp duy nhất, nhưng checkpoint phình từ vài chục MB lên hàng trăm MB và mỗi iteration ghi Drive một lần.
- **Không lưu buffer**: đơn giản nhất, nhưng `--resume` mất toàn bộ dữ liệu đã sinh — trái với yêu cầu nối tiếp từ checkpoint chứ không chạy lại từ vòng 0.
- **Ghi buffer ra Drive**: Drive I/O chậm, làm nghẽn mỗi iteration.

## Consequences

- `--resume` cần cả hai tệp; thiếu `buffer_last.npz` thì vẫn chạy tiếp được nhưng bắt đầu với buffer rỗng — phải in cảnh báo rõ ràng.
- Buffer trên máy thứ hai (ví dụ Colab) không có sẵn; đó là lý do tồn tại `--buffer-in` cho shard.
