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

**Vòng**:
Một chu kỳ huấn luyện: tự chơi xong một lượt ván, thêm mẫu vào Buffer, chạy một số bước tối ưu, rồi lưu Checkpoint. Ngân sách của một lần chạy là số vòng.
_Avoid_: epoch, chu kỳ train

**Checkpoint**:
Bản lưu để chạy tiếp: trọng số mô hình, trạng thái bộ tối ưu, Số bước tối ưu đã chạy, số vòng đã xong và cấu hình. Buffer đi kèm thành một tệp riêng.
_Avoid_: model file, snapshot, trọng số

**Nối tiếp**:
Chạy tiếp một lần huấn luyện đã dừng, từ Checkpoint và Buffer đã lưu: không chạy lại các vòng đã xong và không đặt lại lịch tốc độ học.
_Avoid_: restart, khôi phục, load model

**Số bước tối ưu**:
Số lần cập nhật trọng số đã thực hiện, cộng dồn qua các vòng và qua cả những lần nối tiếp. Là biến độc lập của lịch tốc độ học.
_Avoid_: step, số batch

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

**loss_eval_on_fixed_batch**:
Mất mát đo trên tập 512 mẫu cố định (seed cố định), không dùng để huấn luyện. Là thước đo duy nhất để kết luận mô hình hội tụ trong báo cáo vì loss trên batch ngẫu nhiên dao động quá mạnh.
_Avoid_: validation loss, eval loss ngẫu nhiên

**Mã kết thúc (end_reason)**:
Nguyên nhân kết thúc ván cờ được chuẩn hoá trong log: five_in_row, board_full, forbidden_move, timeout. Dùng để phân loại ván đấu và giải trình nguyên nhân thua cuộc.
_Avoid_: game result string, termination type

**Lợi thế đi trước**:
Hiện tượng bên cầm Đen có tỉ lệ thắng áp đảo trong luật Gomoku tiêu chuẩn. Báo cáo bắt buộc bóc tách tỉ lệ thắng khi cầm Đen và khi cầm Trắng riêng biệt.
_Avoid_: win rate tổng thể không phân loại

**Agent AlphaZero**:
Người chơi đóng gói cây tìm kiếm cùng mạng đã huấn luyện, chỉ lộ ra một phương thức chọn nước cho bộ chạy ván.
_Avoid_: AI player, bot, engine

### So sánh phiên bản

**Baseline**:
Tác tử `random`, mốc thấp nhất mà mọi tác tử khác phải vượt. Không dùng từ này cho `az_v0`.
_Avoid_: baseline ngầm hiểu là AlphaZero

**az_v0**:
AlphaZero gốc: `checkpoints/last.pt` sau 20 vòng × 50 ván trên luật Normal, không có thành phần chiến thuật nào. Là mốc cố định, không sửa, không train đè.
_Avoid_: model cũ, bản gốc, alpha zero

**az_v1**:
AlphaZero cộng thành phần chiến thuật, dùng **đúng trọng số của az_v0** và chỉ khác ở tầng tìm kiếm. Nhờ vậy phép so sánh az_v0 với az_v1 cô lập đúng một biến.
_Avoid_: model mới, bản cải tiến, v2

**az_renju**:
Mô hình Renju tạo bằng cách fine-tune từ az_v0, có Buffer riêng. Không phải một phiên bản cải tiến, không nhân bản chuỗi az_v1 sang Renju.
_Avoid_: az_v1 renju, model renju từ đầu

**Ablation**:
Thí nghiệm tắt rồi bật đúng một thành phần trong khi mọi yếu tố khác giữ nguyên. Khác với phép so sánh az_v0 / az_v1, vốn là thí nghiệm chính của câu hỏi nghiên cứu.
_Avoid_: so sánh phiên bản, test

**Score**:
Điểm đối kháng của một tác tử: (số thắng + 0,5 × số hoà) chia tổng số ván. Là chỉ số xếp hạng, khác win rate thuần.
_Avoid_: win rate (khi ý nói score)

### Chiến thuật

**Bộ nhận diện mẫu**:
Module `gomoku/patterns.py` nhận diện hình dạng quân trên bàn (tam mở, tứ mở, tứ bị chặn, overline) cho một màu tại một ô. Một bản duy nhất dùng chung cho luật Renju, tác tử heuristic và tầng tìm kiếm chiến thuật.
_Avoid_: heuristic module, pattern matcher riêng cho từng nơi

**Tactical override**:
Bước ghi đè chạy sau tìm kiếm: nếu có nước thắng ngay thì đánh, nếu đối thủ có nước thắng ngay thì chặn. Chỉ bật trong az_v1, tắt trong az_v0.
_Avoid_: hardcode chiến thuật, rule injection

**Threat-space search**:
Tìm kiếm chỉ trên chuỗi nước đe dọa và nước đáp trả bắt buộc, bỏ qua các nhánh không tạo đe dọa. Không phải Minimax tổng quát và không chạy trên toàn bộ cây MCTS.
_Avoid_: alpha-beta, tactical search (dùng lẫn)
