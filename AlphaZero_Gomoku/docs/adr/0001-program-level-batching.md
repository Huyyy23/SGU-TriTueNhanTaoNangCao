# Gom lô ở cấp chương trình với MCTS API từng bước

Bàn 11x11 cần 200 lượt mô phỏng mỗi nước, và 1000 ván tự chơi chỉ chạy nổi nếu GPU được dùng theo lô. `az/mcts.py` vì thế **không** phải một hàm `search()` chạy khép kín: nó lộ hai phương thức đối xứng `request()` / `receive(policy, value)`, để `az/selfplay.py` chạy G ván xen kẽ trong **một tiến trình**, mỗi ván giữ đúng một lá đang chờ, rồi gửi tất cả lá đang chờ trong một lượt suy luận. Chế độ tất định (một lá mỗi lượt gọi) chỉ là vòng lặp `request`→`receive` ngay, dùng cho kiểm thử.

Lý do: cách này cho lô thật mà không phải trả giá IPC và nhân bản model — hai thứ mà multiprocessing trên Windows (spawn) bắt buộc phải có. Vì mỗi cây chỉ có một lá đang chờ tại một thời điểm nên **không cần virtual loss**.

## Considered Options

- **Multiprocessing, mỗi worker một cây, batch server ở tiến trình chính**: đúng chữ "multiprocessing" trong `AGENTS.md`, nhưng nhân bản model, serialize qua IPC, và spawn trên Windows — đắt hơn lợi ích biên trên bàn 11x11 với mạng 6×64. Giữ làm wrapper mỏng của chế độ một tiến trình nếu đo thấy vẫn thiếu GPU.
- **MCTS tự gom lô bên trong**: trộn trách nhiệm, và dễ trôi về gom con của một node — đúng thứ cần tránh.

## Consequences

- Kích thước lô dao động từ 1 đến G (ván vừa kết thúc thì ít lá chờ hơn), nên phải log kích thước lô trung bình thay vì giả định nó bằng G.
- Bộ đếm `nn_calls` là chỉ số kiểm chứng: một ván phải tốn xấp xỉ `n_simulations × số nước`, không phụ thuộc G.
