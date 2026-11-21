# Báo cáo cleanup cuối cùng AZ Normal

**Phạm vi:** chỉ xóa run `selfplay_20k_cpp_2w` và checkpoint branch `az_25k_continuation_v1`; giữ nguyên các database dùng chung, kết quả tournament, Renju và baseline khác. Không chạy self-play/training.

## Artifact đã xóa

| Đường dẫn | Trạng thái trước cleanup | Số file | Dung lượng |
|---|---:|---:|---:|
| `checkpoints_selfplay_20k_2w/` | Checkpoint, replay, manifests, 40 recovery outbox | 336 | 5.256.734.579 bytes |
| `selfplay_20k_backup_e01gnlu4/` | Bản backup SQLite + bản sao checkpoint; backup SQLite chứa cả lịch sử nhiều run | 337 | 7.103.379.315 bytes |
| `checkpoints/az_25k_continuation_v1/` | Generation/recovery cũ đã thay bởi checkpoint Normal chính thức | 5.625 | 69.922.451.574 bytes |
| `configs/normal_continuation.yaml` | Config continuation cũ, không có caller hoạt động | 1 | ước lượng không đáng kể |

Tổng dung lượng file đã giải phóng: **82.282.565.468 bytes (~82,28 GB decimal / 76,65 GiB)**. Đây là tổng kích thước file đã đo trước khi xóa; không bao gồm metadata filesystem.

Các artifact nằm trong hai cây checkpoint và backup đã bị xóa toàn bộ. Tìm kiếm `selfplay_20k_cpp_2w` ngoài cây đã xóa, backup, SQLite và hai báo cáo forensic/migration chỉ còn tham chiếu mô tả trong `SPEC.md`; không còn log/metrics/cache riêng run 20k được tìm thấy.

## Tournament 20k và provenance

Bốn experiment trong `data/tournament.sqlite` được giữ nguyên kết quả game/move/event; chỉ đường dẫn checkpoint đã bị xóa được thay bằng định danh `historical://checkpoint-sha256/<SHA>`. `agents.checkpoint_sha256` được giữ. Mỗi experiment có metadata `checkpoint_available=false`, `reproducibility=historical_non_reproducible`; không gán checkpoint đó cho AZ Normal chính thức.

| Experiment | Games | AZ W-D-L |
|---|---:|---:|
| `selfplay_20k_cpp_2w-promotion-10-heuristic-0` | 100 | 3-0-97 |
| `selfplay_20k_cpp_2w-promotion-20-heuristic-0` | 100 | 14-0-86 |
| `selfplay_20k_cpp_2w-promotion-30-heuristic-0` | 100 | 18-0-82 |
| `selfplay_20k_cpp_2w-promotion-40-heuristic-0` | 100 | 21-0-79 |

Không có rows Elo trong `rating_systems`/`rating_events` của DB này; vì vậy báo cáo không tuyên bố có Elo đã lưu. `data/tournament.sqlite` integrity và foreign keys sạch sau migration. W/D-L, match/game identity, moves, seeds, checkpoint SHA và kết quả không bị thay đổi; path trong identifiers/payload được chuyển thành SHA identity để không còn trỏ tới file mất.

`data/selfplay.sqlite` không có `selfplay_20k_cpp_2w` run record. Không xóa file SQLite dùng chung.

## AZ Normal chính thức

- Pointer: `checkpoints/normal/latest.json`
- Generation: `ba897320ab8d406c881de90496d7f3a0`
- Checkpoint SHA-256: `73c7d292a1588255cbcfe9a2d1fbc6dd471cb3e2e6cedb9850d0b47f05e899e9`
- Replay SHA-256: `4a4b13e229a51395d013e93987b5dd9e602140b9aaccb9b06b0e4bd6e3df4120`
- Load qua `az.trainer.resolve_generation` và `az.model.load_checkpoint` thành công: `AZNet`, iteration 700, global step 70.000, ruleset `normal`.
- SHA khớp với generation 60k được kiểm toán trong `docs/az-normal-migration-report.md`; đây là xác nhận identity của weights, không phải khẳng định sức chơi hay chất lượng mới.

Production paths `az`, `scripts`, `configs`, `players`, `ui`, `gomoku` không còn tham chiếu `az_25k_continuation_v1`, `normal_continuation`, hoặc cây run 20k. Normal default vẫn đọc pointer chính thức; thiếu/hỏng checkpoint là lỗi thay vì tự khởi tạo. Chế độ khởi tạo mới vẫn chỉ explicit `--fresh`.

## SQLite, references và dữ liệu giữ lại

- `data/normal.sqlite`, `data/selfplay.sqlite`, `data/tournament.sqlite`: `PRAGMA integrity_check = ok`; `foreign_key_check` rỗng.
- Tournament database không còn đường dẫn đến `checkpoints_selfplay_20k_2w` trong trường checkpoint. Command snapshot cũ còn ghi `--out-dir checkpoints_selfplay_20k_2w` ở config/event history; đây là provenance của lệnh đã chạy, không phải checkpoint reference.
- Giữ `data/az_25k_continuation_v1.sqlite`, `data/az_25k_continuation_v1_tournament.sqlite`, các SQLite evaluation, logs và báo cáo/dataset thí nghiệm liên quan: chúng lưu lịch sử lineage, metrics, hoặc input/output phân tích; không phải model runtime. Các checkpoint intermediate từng được tham chiếu trong đó đã bị xóa, nên phép tái lập dựa vào các generation này hiện **historical / non-reproducible**. Không viết lại kết quả hay chuyển chúng sang AZ Normal. `docs/az-80k-checkpoint-forensics.md` và `docs/az-normal-migration-report.md` được giữ nguyên làm snapshot forensic của thời điểm trước cleanup.
- Giữ `checkpoints_selfplay_20k/` (không có hậu tố `_2w`): đây là cây riêng khác, chỉ một generation nhỏ, không có tham chiếu được tìm thấy và không chứng minh được cùng ownership với run đã nêu. Không xóa đoán.
- Giữ `checkpoints/normal/`, checkpoint Renju, baselines và `data/normal.sqlite`.

## Kiểm thử và giới hạn

- `pytest` trong `.venv`: **101 passed**, 1 cảnh báo deprecation từ pygame/pkg_resources.
- Load checkpoint/replay chính thức và hash đối chiếu thành công.
- Integrity/FK trên ba SQLite hoạt động sạch; các đường dẫn hoạt động không còn alias continuation.
- Không chạy self-play hoặc training; không đánh giá lại tournament, không tính Elo mới, không kiểm chứng UI trên màn hình thật.
- Các tài liệu/datasets lịch sử còn giữ đường dẫn cũ để bảo toàn provenance; chúng không được dùng làm checkpoint mặc định và các kết quả cần generation intermediate đã xóa không tái lập được.
