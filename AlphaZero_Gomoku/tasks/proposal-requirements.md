# Yêu cầu cho proposal — Gomoku 11x11 + AlphaZero

Danh sách thông tin proposal phải có. Bản nháp đã dựng ở `tasks/proposal.md`.
Mỗi mục ghi: cần có gì, lấy từ đâu, còn thiếu gì.

## Trạng thái sau khi chốt với người dùng

| Thông tin | Trả lời | Mục checklist đóng |
|---|---|---|
| Người đọc | Giảng viên bộ môn AI, không có mẫu | Mục 1, cấu trúc tự dựng theo chuẩn proposal kỹ thuật |
| Mục tiêu | Tạo game chơi được, có agent ra quyết định | Mục 2, 12, 13 |
| Câu hỏi nghiên cứu | Đã chốt: "Với ngân sách 1000 ván tự chơi, agent Gomoku 11x11 đạt được trình độ nào, và win rate tăng thế nào theo từng 100 ván?" | Mục 3, đóng |
| Deadline | Chưa có; proposal cố ý không gán lịch theo tuần, điền sau khi chốt hạn nộp | Mục 15, để trống |
| Hạ tầng | Colab free, không đưa vào proposal | Mục 14 chỉ nêu máy cá nhân có GPU |
| Nhóm | 3 người; vai trò A/B/C đã chốt trong proposal mục 7.2 (A lõi game và luật + UI, B tầng AlphaZero, C hạ tầng và đánh giá + báo cáo); tên đã chốt: A = Khang, B = Huy, C = Hường | Mục 15, đóng khi xác nhận |
| Tài liệu tham khảo | Đã chốt 7 mục, proposal mục 9 | Mục 4, đóng |

## Khối 1 — Ý tưởng (người dùng cung cấp)

| # | Mục | Cần có gì | Nguồn | Còn thiếu |
|---|---|---|---|---|
| 1 | Bối cảnh | Ai đọc, theo mẫu nào, hạn nộp, đồ án môn / nghiên cứu / sản phẩm | — | Toàn bộ |
| 2 | Mục tiêu | Mục tiêu kỹ thuật + mục tiêu của proposal (học tập, publication, demo, nghiên cứu nội bộ) | Mục tiêu kỹ thuật đã có trong `AGENTS.md` | Mục tiêu của proposal, tiêu chí người duyệt chấp nhận |
| 3 | Câu hỏi nghiên cứu | Vì sao Gomoku 11x11, vì sao cần Renju, đóng góp của dự án là gì | — | Toàn bộ |

## Khối 2 — Kỹ thuật

| # | Mục | Cần có gì | Nguồn | Còn thiếu |
|---|---|---|---|---|
| 4 | Dẫn nguồn tài liệu | Danh mục paper (AlphaZero 2018, AlphaGo Zero 2017, EfficientZero 2020, luật Renju) + bảng "module → lấy src ở đâu → dùng làm gì → giấy phép" | `tasks/plan.md` (T22 dựng `docs/provenance.md`) | Link chính xác từng paper, commit SHA của repo ngoài nếu có |
| 5 | Kiến trúc + phương pháp | 4 plane input, node-batched PUCT, đối xứng 8-fold, một model cho Normal và Renju, buffer tách theo rule set | `tasks/plan.md` mục Architecture Decisions | Không |
| 6 | Kế hoạch triển khai | 25 task, 10 phase, checkpoint A–J, mốc thời gian | `tasks/todo.md` | Cột deadline thật cho từng checkpoint |

## Khối 3 — Thí nghiệm và kết quả

| # | Mục | Cần có gì | Nguồn | Còn thiếu |
|---|---|---|---|---|
| 7 | Chơi thử và nhật kí | Quy trình chơi thử, ghi log, thắng/thua, trả lời "thua vì sao" | T19 (schema JSONL: `end_reason` = `five_in_row` / `forbidden_move` / `board_full` / `timeout`) | Ngưỡng coi là lỗi vs hợp lý |
| 8 | Ai đi trước, ai đi sau | Quy ước luật: 1000 ván xen kẽ, đúng 500 ván mỗi bên đi trước | T21 | — |
| 9 | Chiến lược nào đi trước | Bảng tổng hợp: thắng khi đi trước, thắng khi đi sau, phân bố `end_reason`, phân tách theo màu | T21 (`summary.json`) | — |
| 10 | Thí nghiệm ghi trên nhật kí | Mỗi lần chạy thí nghiệm ghi lại: cấu hình, seed, kết quả, link log ván | T19 + T21 (log ván + `summary.json` cạnh nhau) | Quy ước đặt tên run để truy vết |
| 11 | Lưu log dữ liệu thắng thua | Kho log: `logs/games/*.jsonl` (mỗi ván 1 dòng) + `logs/eval_*.jsonl` + `summary.json` | T19, T21 | Chính sách giữ log bao lâu, có push lên Drive/GitHub không |

## Khối 4 — Kết quả kỳ vọng và cam kết

| # | Mục | Cần có gì | Nguồn | Còn thiếu |
|---|---|---|---|---|
| 12 | Tiêu chí đánh giá | Ngưỡng đo: agent 11x11 thắng random ≥ 18/20 ván, arena win rate ≥ 0.55, hoạt động được cả Normal và Renju | `tasks/plan.md` Checkpoint | Ngưỡng cụ thể cho từng tiêu chí |
| 13 | Deliverable | Code repo + checkpoint + UI + báo cáo + (tùy) bài báo, mỗi cái có hạn nộp | — | Toàn bộ |
| 14 | Tài nguyên | RTX 4060 16 luồng + Colab free (1 instance GPU + 1 CPU-only), chi phí = 0 nếu không dùng Colab Pro | `tasks/plan.md` Environment | Có mua Colab Pro không |
| 15 | Rủi ro + timeline | Bảng 11 rủi ro kèm biện pháp, gantt 10 phase | `tasks/plan.md` Risks | Ngày bắt đầu và ngày nộp |

## Điều kiện trước khi viết proposal

Cần người dùng cung cấp: mẫu proposal (nếu có) · mục tiêu thật + tiêu chí duyệt · deadline từng mốc · ngân sách (có Colab Pro không) · nhóm thực hiện và phân chia công việc.
