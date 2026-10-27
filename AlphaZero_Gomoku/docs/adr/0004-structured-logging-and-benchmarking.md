# Kiến trúc ghi log ba tầng và chuẩn hoá dữ liệu đo đạc thực nghiệm

Để phục vụ báo cáo khoa học và phân tích thực nghiệm mà không dựa vào phỏng đoán hay trí nhớ, tầng AlphaZero và toàn bộ pipeline chuẩn hoá hệ thống log thành ba tầng độc lập, ghi theo định dạng append-only JSONL, và bắt buộc flush ngay sau mỗi sự kiện:

1. **Tầng benchmark phần cứng (`logs/benchmarks/`)**: ghi kết quả đo đạc latency theo kích thước lô (batch = 1, 4, 16, 64) trên GPU/CPU, thông lượng MCTS (số sim/s), và tốc độ self-play thực tế. Cung cấp số liệu định lượng cho việc giải trình lựa chọn gom lô cấp chương trình.
2. **Tầng tiến trình huấn luyện (`logs/train_<run_id>.jsonl`)**: mỗi dòng là một iteration, gồm thời gian self-play/train, `buffer_size`, số mẫu duy nhất được dùng (`unique_samples_used`), learning rate, policy loss, value loss, và đặc biệt là `loss_eval_on_fixed_batch` (đo trên tập 512 mẫu cố định) — thước đo hội tụ ổn định duy nhất cho báo cáo.
3. **Tầng ván đấu (`logs/games/<run_id>.jsonl`)**: do `GameLogger` trong `az/game_log.py` phụ trách, dùng chung cho Runner, Self-play và Đánh giá (Arena). Mỗi dòng lưu một ván cờ với các trường quyết định: `first_player_won`, `end_reason`, `nn_calls`, `mean_batch`, và tuỳ chọn `--log-moves`. Kết thúc đợt đánh giá tự động sinh `summary.json` tách riêng tỷ lệ thắng khi đi trước và đi sau.

## Considered Options

- **Dùng TensorBoard / WandB đơn thuần**: khó trích xuất tự động thành bảng Markdown cho báo cáo LaTeX/Markdown của đồ án, phụ thuộc thư viện ngoài và khó chạy offline an toàn trên máy phòng lab.
- **Lưu database SQLite**: dư thừa abstraction cho mô hình huấn luyện tuần tự/gom shard, khó inspect bằng công cụ dòng lệnh tiêu chuẩn (`head`, `jq`, `grep`).
- **Gộp chung 1 file log duy nhất**: làm phình kích thước file, script vẽ biểu đồ loss phải parse qua hàng trăm nghìn dòng ván đấu không cần thiết.

## Consequences

- Mọi con số trong báo cáo đều có nguồn gốc truy vết ngược về từng dòng log cụ thể (`tests/test_report.py` có thể assert tự động).
- Báo cáo phân tích được chính xác nguyên nhân thua và bóc tách được lợi thế đi trước đặc thù của cờ Gomoku.
