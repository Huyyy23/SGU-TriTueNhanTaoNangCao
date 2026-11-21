# Migration checkpoint AZ Normal và cleanup run 20k

## Kết quả và giới hạn

- Đường dẫn Normal chính thức: `checkpoints/normal/latest.json`, generation `ba897320ab8d406c881de90496d7f3a0`. Checkpoint file SHA-256 `73c7d292a1588255cbcfe9a2d1fbc6dd471cb3e2e6cedb9850d0b47f05e899e9`, **trùng byte** với checkpoint 60k được kiểm toán tại `checkpoints/az_25k_continuation_v1/`. Replay SHA-256 `4a4b13e229a51395d013e93987b5dd9e602140b9aaccb9b06b0e4bd6e3df4120` khớp nguồn. Mốc 60k là danh nghĩa; không cộng run 20k độc lập.
- Bản canonical chứa recovery outbox iteration 699 và generation candidate `1b8d8a4604584f0e93ce111f4e0121eb` cần thiết để kiểm tra event đánh giá iteration 700. Lịch sử generation cũ hơn vẫn nằm trong cây nguồn lịch sử; lịch sử rollback tại canonical không bao gồm toàn bộ các generation cũ.
- `data/normal.sqlite` là bản sao bằng SQLite backup API từ `data/az_25k_continuation_v1.sqlite`; `integrity_check=ok`, `foreign_key_check` rỗng. Database lịch sử được giữ nguyên. Bản canonical vẫn giữ run ID lịch sử để xác thực recovery; tên này là provenance, không phải một model chính thức khác.
- Chưa xóa hết filesystem artifact 20k: checkpoint, replay và recovery `checkpoints_selfplay_20k_2w/` được giữ vì tournament tham chiếu bốn generation và checkpoint còn tham chiếu outbox. Xóa mà không xác minh thêm sẽ làm hỏng liên kết lịch sử hoặc recovery. Các artifact này không phải checkpoint chính thức và không được hiển thị như lựa chọn model mặc định. Bản backup cũng còn dữ liệu run 20k, không nằm trong DB hoạt động.

## Backup và SQLite cleanup

Backup `selfplay_20k_backup_e01gnlu4/`: SQLite qua backup API, integrity và foreign keys sạch; bản sao cây checkpoint gồm 336 file, 5.256.734.579 byte, SHA-256 từng file khớp nguồn tại thời điểm backup. Không xóa DB dùng chung hoặc tournament evidence.

Xóa bằng transaction đúng run `selfplay_20k_cpp_2w`; thống kê dưới là **tổng dòng toàn bảng** của `data/selfplay.sqlite`:

| Bảng | Trước | Sau |
|---|---:|---:|
| `runs` | 316 | 315 |
| `games` | 69.270 | 49.270 |
| `moves` | 1.023.202 | 713.023 |
| `mcts_roots` | 576.568 | 266.389 |
| `mcts_traces` | 0 | 0 |
| `training_steps` | 714 | 674 |
| `events` | 25.827 | 25.822 |
| `artifacts` | 0 | 0 |
| `source_imports` | 479 | 479 |
| `schema_migrations` | 1 | 1 |

Sau transaction, SQLite integrity `ok`, foreign-key check rỗng; truy vấn riêng run cho `runs`, `games`, `training_steps`, `events` đều bằng 0. Dữ liệu các run khác và bốn experiment tournament được giữ. Không có file checkpoint/outbox 20k nào bị xóa vì các phụ thuộc nêu trên; user yêu cầu giữ artifact khi chưa xác minh được dependency.

## Training và consumer

- Training Normal mặc định nạp generation chính thức, replay, optimizer, RNG và owner; checkpoint thiếu/hỏng là lỗi. `--fresh` là tường minh và không ghi đè output đã có. CLI không còn tạo nhánh bootstrap Normal; dữ liệu bootstrap lịch sử vẫn đọc được.
- `--iterations` là **iteration đích tuyệt đối**; bỏ qua mặc định tiếp tục thêm một iteration. Checkpoint hiện tại ở iteration 700, cấu hình 50 game/iteration, nên `--iterations 701` yêu cầu thêm 50 game. `--games` khác cấu hình nguồn bị từ chối để không làm sai promotion counter và recovery: **chưa hỗ trợ N tùy ý, không chia hết cho 50**, trong cùng run.
- Manifest/metrics mới ghi parent checkpoint SHA/generation, run ID và thống kê ván sinh, game ID đã lấy mẫu duy nhất kể từ khi bắt đầu tracking, cùng số sample không xác định được game ID. Lịch sử chưa xác minh giữ `unknown`; không suy diễn game ID là lineage đã xác minh. Số ván sinh, được lấy mẫu và thuộc lineage không đồng nhất.
- Factory, CLI arena/tournament và GUI Normal mặc định dùng pointer chính thức; vẫn cho chỉ định checkpoint để tái lập baseline lịch sử. GUI chỉ tự phát hiện pointer Normal chính thức. Renju vẫn dùng checkpoint riêng.
- Commit latest dùng generation/manifest/hash và atomic pointer; kết quả quality gate là khái niệm độc lập, không suy ra model tốt hơn chỉ vì commit thành công.

## Kiểm chứng

- `pytest ... tests/test_train_resume_logging.py tests/test_trainer.py tests/test_tournament.py tests/test_ui.py tests/test_az_player_factory.py`: **101 passed, 1 warning** trước cutover CLI cuối.
- `pytest ... tests/test_train_resume_logging.py tests/test_trainer.py`: **39 passed** sau cutover CLI.
- `pytest ... tests/test_ui.py tests/test_play_cli.py tests/test_agent_factory.py tests/test_arena.py`: **37 passed, 1 warning** sau cập nhật GUI.
- Resume `--iterations 700` trên bản sao tạm checkpoint/recovery/SQLite: load iteration 700 và global step 70.000; không sinh ván hay train. Một lần kiểm đầu phát hiện event promotion trỏ candidate chưa sao chép; đã bổ sung candidate và kiểm tra lại thành công.
- Pygame với SDL dummy mở vòng lặp menu và thoát, checkpoint mặc định là pointer Normal; không kiểm chứng hình ảnh trên màn hình thật.
- Không chạy self-play hoặc training thật. Không chứng minh độ bền khi mất điện, toàn bộ lineage lịch sử, sức chơi, hay cleanup hoàn toàn các artifact còn phụ thuộc.
