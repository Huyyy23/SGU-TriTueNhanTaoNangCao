# Gomoku 11x11 + AlphaZero

Bối cảnh duy nhất của repo: bàn Gomoku 11x11 với agent tự học bằng self-play, hai bộ luật Normal và Renju. Từ điển này chốt nghĩa các từ mà tài liệu và mã phải dùng giống nhau. Quyết định thiết kế nằm ở `docs/adr/`, không nằm ở đây.

## Language

### Tìm kiếm

**Lá**:
Node trong cây tìm kiếm chưa được mở rộng, tức chưa có danh sách nước đi con.
_Avoid_: node chưa thăm, frontier, nút lá

**Lượt suy luận**:
Một lần gọi mạng nơ-ron. Là đơn vị đếm chi phí của cả hệ thống (`nn_calls`), không phải số lượt mô phỏng.
_Avoid_: lần forward, batch inference, lượt đánh giá

**Lô cấp chương trình**:
Tập hợp các lá đang chờ đánh giá, gom từ nhiều ván chạy song song, và được gửi trong đúng một lượt suy luận. Không phải tập nước đi con của một node.
_Avoid_: batch cấp cây, node batch

**Nhiệt độ (τ)**:
Tham số quyết định cách chọn nước từ số lần thăm ở gốc cây. Bậc thang: bằng 1 trong một số nước đầu, sau đó chọn nước nhiều lượt thăm nhất.
_Avoid_: temperature decay, giảm dần

**Nước cấm**:
Nước đi mà luật Renju không cho phép Đen thực hiện. Nước cấm không bao giờ được mở rộng trong cây.
_Avoid_: illegal move, nước không hợp lệ

### Học

**Mẫu**:
Một bộ ba gồm đặc trưng bàn cờ, phân phối mục tiêu π, và kết quả z, kèm nhãn bộ luật.
_Avoid_: transition, experience, dữ liệu huấn luyện

**π (phân phối mục tiêu)**:
Phân phối xác suất trên 121 ô mà mô hình được huấn luyện để khớp, lấy từ số lần thăm ở gốc cây với nhiệt độ bằng 1 nên luôn mềm.
_Avoid_: visit distribution, one-hot policy, nhãn chính sách

**z (kết quả)**:
Giá trị kết quả ván, tính theo góc nhìn người đang đi tại vị trí được lưu: thắng +1, thua -1, hoà 0.
_Avoid_: reward, outcome, label

**Góc nhìn người đang đi**:
Quy ước dấu cho mọi giá trị trong cây và trong dữ liệu: giá trị luôn là lợi ích của bên sắp đi. Giá trị của node con là góc nhìn đối thủ nên phải đảo dấu khi truyền ngược.
_Avoid_: góc nhìn Đen, góc nhìn cố định

**Buffer**:
Kho mẫu giới hạn dung lượng, dùng chung cho các vòng huấn luyện, tách riêng theo bộ luật.
_Avoid_: replay memory, dataset

### Kiểm chứng

**Evaluator giả**:
Bộ đánh giá thay thế mạng nơ-ron trong kiểm thử, trả phân phối đều và một giá trị cố định, để tầng tìm kiếm và tự chơi test được mà không cần huấn luyện.
_Avoid_: fake model, dummy net

**Chế độ tất định**:
Cách chạy tự chơi một ván một lúc, mỗi lượt suy luận phục vụ đúng một lá, không gom lô. Dùng cho kiểm thử và làm mốc so sánh với chế độ gom lô.
_Avoid_: sequential mode, debug mode

**Ngân sách ván**:
Số ván tự chơi mà một lần chạy phải hoàn thành, bằng số vòng nhân số ván mỗi vòng. Đọc từ cấu hình, không bao giờ là hằng số trong mã.
_Avoid_: 1000 ván, số ván cố định

**Agent AlphaZero**:
Người chơi đóng gói cây tìm kiếm cùng mạng đã huấn luyện, chỉ lộ ra một phương thức chọn nước cho bộ chạy ván.
_Avoid_: AI player, bot, engine
