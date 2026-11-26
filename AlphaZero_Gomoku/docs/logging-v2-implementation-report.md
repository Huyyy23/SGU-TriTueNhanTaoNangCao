# Logging V2 — báo cáo triển khai và kiểm toán

**Trạng thái: cutover và import lịch sử đã triển khai; acceptance chưa hoàn tất.** Không chạy lại thí nghiệm self-play/tournament, không xóa nguồn cũ, không sửa checkpoint/replay hoặc kết quả AI. STANDARD overhead sau cutover chưa được benchmark nên chưa thể tuyên bố đạt KPI ≤5%.

## 1. Tóm tắt

- `data/selfplay.sqlite` là nguồn dữ liệu cho self-play/training; `data/tournament.sqlite` là nguồn cho tournament/Elo. Các file DB/sidecar bị `.gitignore` loại khỏi Git; không có DB tracked cần untrack. Không ghi đè DB khi import/export.
- Training, arena, tournament, ablation/benchmark event writers và `analyze_t35.py` dùng SQLite mặc định. Tournament resume đọc các ván đã commit trong DB. Export/import bundle thí nghiệm tournament là thao tác tường minh, additive, hash/idempotent.
- Đã import 1.133 trong 1.136 nguồn log nhận diện được từ `logs/` và `evidence/`. Ba nguồn xung đột/không đồng nhất bị giữ nguyên, không ép ghép. Toàn bộ file gốc vẫn còn nguyên.
- Hai DB hiện qua `PRAGMA integrity_check` (`ok`) và `PRAGMA foreign_key_check` (0 lỗi). Dung lượng sau import: selfplay 521,580,544 byte; tournament 3,043,328 byte.
- Chưa hoàn tất MCTS root logging trong tournament, TRACE, self-play exchange, crash injection, reconciliation checkpoint/replay, và benchmark overhead sau cutover. Không được xem đây là hoàn thành toàn bộ Logging V2.

## 2. Hiện trạng logging trước migration

Audit source/runtime paths nhận diện các hệ thống cũ sau:

- `az/game_log.py`: GameRecord/MoveRecord và trước đây có writer/read/summary JSONL. File writer đã được gỡ; record vẫn là mô hình dữ liệu trong memory.
- `scripts/train.py`, `az/selfplay.py`, `az/renju_finetune.py`: trước đây có log JSONL hoặc event stdout; giờ training/self-play/finetune metrics đi vào selfplay SQLite; checkpoint/replay vẫn độc lập.
- `scripts/arena.py`, `scripts/tournament.py`: trước đây có game JSONL, manifest, CSV, summary; nay kết quả/event đi vào tournament SQLite.
- `scripts/ablation.py`, `scripts/bench_batching.py`, `scripts/bench_selfplay_round2.py`: kết quả/log nghiên cứu được ghi thành SQLite events. JSON export chỉ phát sinh nếu truyền tùy chọn `--out`.
- `az/backend_logging.py`: bản ghi backend cũ ở `logs/backend_runs.sqlite3` được import theo workload vào DB chính; writer mới dùng event của DB chính.
- `scripts/analyze_t35.py`, resume và Elo: analytics/resume/recalculation đọc database; JSON export là tùy chọn, không mặc định.
- Native `native/gomoku_mcts_native.cpp` không tự ghi file/console; instrumentation MCTS thuộc Python wrapper/API hiện có.

Inventory toàn bộ 1.244 file trong `logs/` và `evidence/`; importer nhận diện 1.136 file (`.json`, `.jsonl`, `.csv`, `.log`, `.txt`, SQLite). Nhóm: 512 metadata JSON, 65 opaque (64 log + 1 TXT), 270 game, 190 CSV, 42 benchmark, 42 training, 7 training-metric, 1 backend SQLite, 7 tournament-game. 108 file còn lại (mã nguồn/cache, checkpoint, YAML/ảnh, archive/notebook và extension không hỗ trợ) được giữ nguyên, không đưa vào DB. Đây là phân loại định dạng, không phải kết luận file nào có thể xóa.

## 3. Vấn đề và nguyên nhân

- Nhiều định dạng tách rời làm resume, metrics và báo cáo phải đọc lại JSONL/CSV.
- File snapshot/resume có thể đại diện dữ liệu độc lập hoặc bổ sung; cùng content không đủ chứng minh cùng một thí nghiệm.
- Historical training thường thiếu run identity; importer gắn identity ổn định theo source path thay vì bịa run ID gốc.
- Một số aggregate CSV/metadata không đủ dữ liệu gốc để tái dựng games/Elo; được giữ dạng raw event, không suy diễn kết quả.
- Root/TRACE không có producer xuyên suốt cả Python và native tournament. Không được báo cáo BLOB/MCTS metrics nếu source không có.

## 4. Kiến trúc SQLite V2

- `selfplay.sqlite`: runs, games, moves, MCTS roots/traces, training steps, artifacts, events, source imports và schema migrations.
- `tournament.sqlite`: experiments, agents, matches, games, moves, MCTS roots/traces, rating systems/events, events, source imports và schema migrations.
- WAL, foreign keys, stable IDs, uniqueness, schema-version preflight, BLOB encode/decode, backup helper và prepared SQL statements nằm trong `az/logging_v2.py`.
- STANDARD self-play ghi move/root telemetry sau khi search/game data đã thu được; NONE bỏ thu thập telemetry không cần thiết và durable writes. Không thay đổi MCTS, RNG, luật hoặc số simulations.
- Tournament STANDARD ghi DB; resume lấy config/result đã commit từ cùng DB và cần `--run-id` rõ ràng. NONE chỉ là baseline không lưu dữ liệu.
- STANDARD training resume yêu cầu `--run-id` ban đầu; hệ thống chỉ dùng lại dòng run hiện có nếu config/backend/rules/seed/metadata khớp. Checkpoint cũ không có dòng DB sẽ bắt đầu một run với identity mới; importer không suy đoán lineage.
- `az.experiment_exchange` export/import một tournament experiment sang bundle SQLite tường minh. Import additive; xung đột rollback, hash + experiment identity chống duplicate; không thay DB đích. Self-play run bundle chưa có.

## 5. Schema và indexing

Schema hiện có các bảng theo hai nhóm tối thiểu trên; `schema_migrations` ghi version 1. Các foreign key/unique constraints và index hỗ trợ run, game, event, experiment, agent, match và rating queries. Chưa có migration nâng version 1 lên version sau.

Historical source được ghi dấu ở `source_imports` (source path/identity, SHA-256, trạng thái, số record, loại dữ liệu). Raw source record cùng source hash/line được lưu dưới dạng event để giữ provenance; game/training/tournament rows chỉ được tạo khi dữ liệu đủ trường. CSV/metadata không đủ chi tiết vẫn query được dưới dạng raw event, không được giả làm game rows.

Sau migration, `mcts_roots`/`mcts_traces` và Elo tables còn trống trong dữ liệu production local: source import không tự tạo telemetry hoặc rating history không tồn tại.

## 6. File/module đã thay đổi

- `.gitignore`: bỏ SQLite DB, sidecars, journals và bundle local khỏi Git; không xóa DB local.
- `az/logging_v2.py`, `az/backend_logging.py`: schemas, định tuyến structured event, telemetry gốc, helper sao lưu và kiểm tra toàn vẹn dữ liệu.
- `az/mcts.py`, `native/gomoku_mcts_native.cpp`, `az/selfplay.py`, `scripts/train.py`: chỉ thu thập telemetry để đọc và lưu self-play/training vào SQLite; không đổi search/training targets.
- `az/game_log.py`, `az/evaluator.py`: bỏ JSONL file writer/read/summary; game records/summaries còn xử lý trong memory.
- `scripts/tournament.py`, `scripts/arena.py`, `az/renju_finetune.py`: chuyển kết quả/tổng hợp sang SQLite; checkpoint/replay output giữ nguyên.
- `scripts/ablation.py`, `scripts/bench_batching.py`, `scripts/bench_selfplay_round2.py`: event mặc định vào DB; explicit JSON export không mặc định.
- `scripts/analyze_t35.py`: analytics đọc tournament DB, export chỉ khi yêu cầu.
- `az/elo.py`, `scripts/recalculate_elo.py`: Elo replay từ game rows.
- `az/experiment_exchange.py`, `scripts/exchange_experiment.py`: exchange tournament bundle.
- `scripts/migrate_logs_to_sqlite.py`: inventory/import JSONL, JSON arrays/objects, CSV, log/TXT và SQLite backend cũ.
- Kiểm thử: `test_logging.py`, `test_logging_v2.py`, `test_migrate_logs_to_sqlite.py`, `test_train_resume_logging.py`, `test_elo.py`, `test_experiment_exchange.py`, `test_tournament.py`, `test_arena.py`, `test_ablation.py`, `test_backend_auto.py`, `test_selfplay.py`, `test_native_mcts.py`.
- `README.md`, `SPEC.md`, `docs/tournament_guide.md`, `docs/adr/0004-structured-logging-and-benchmarking.md` và báo cáo này.

## 7. Migration inventory và reconciliation

Đã import additive, không sửa/xóa/đổi tên source:

| Nhóm | Kết quả |
|---|---|
| Game/self-play | 268 file normalized; 25.220 game rows cuối DB, bao gồm dữ liệu đã có trước lần import mở rộng |
| Training | 42 source file; 196 training-step rows; metric records không đủ cấu trúc được giữ event |
| Tournament | 6 file normalized, 20 game rows; 183 CSV và 465 metadata source được lưu raw event |
| Benchmark | 42 file/source event; không biến aggregate throughput thành game records |
| Backend | Legacy `logs/backend_runs.sqlite3` được route theo selfplay/tournament vào hai DB |
| Opaque/evidence | 65 log/TXT sources cùng metadata/snapshot được lưu raw event; file nguồn giữ nguyên |
| Checkpoint/replay | Không import, không sửa; `.pt`/NPZ vẫn ở định dạng/đường dẫn riêng |

Tổng kết importer: 1.133/1.136 source files được nhập; 3 bị giữ lại:

1. Hai recovery game histories có cùng synthetic source-run/game identity nhưng payload khác nhau; importer từ chối collision.
2. `logs/integration_merged_20261114/games.jsonl` trộn identity/ruleset; không thể xác định một experiment duy nhất.

Source identity dùng path tương đối repo cùng source hash; game/run IDs riêng biệt không bị gộp chỉ vì moves giống nhau. Backend DB được route vào hai DB chính nên một source có thể có provenance row ở mỗi DB. `PRAGMA integrity_check`=`ok`; foreign-key violations=0 cho cả hai database. Re-import idempotency, duplicate source và distinct-source behavior được kiểm thử. Không chạy cleanup; không tạo restore backup để xóa nguồn vì không có nguồn nào bị xóa.
Hai primary database chưa tồn tại trước lần import đầu; importer tạo mới, các lượt sau chỉ append theo source identity/hash. Không có DB cũ bị ghi đè và không cần backup DB trước khi import ban đầu. Vì không cleanup file nguồn nên không tạo/xóa backup nguồn.
Final database rows: selfplay runs=310, games=25,220, moves=446,634, training_steps=196, events=25,800, source_imports=479; tournament experiments=6, agents=7, matches=10, games=20, moves=508, events=1,447, source_imports=655. Combined provenance rows exceed distinct source files by one because the legacy backend database is routed to both primary databases.

## 8. File đã chuyển đổi, giữ lại hoặc chưa thể chuyển

- Đã chuyển: normalized game/training/tournament records; raw metadata, opaque logs, CSV, benchmark, backend selection và unsupported-but-readable record payloads qua events/provenance.
- Giữ nguyên: toàn bộ nguồn ở `logs/` và `evidence/`, checkpoint, replay buffer, archived snapshots, database cũ và notebook/archive không nằm trong parser scope.
- Chưa thể normalize: 3 nguồn identity-conflict/mixed; CSV aggregate không có game-level provenance; archive/format không được parser nhận diện. Chúng được báo cáo, không bị bỏ/xóa.
- Một số historical MCTS root payload giữ trong raw source event, chưa chuyển thành compressed MCTS-root BLOB. Không thể claim producer/query schema đã chuẩn hóa payload đó.

## 9. Self-play benchmark trước/sau

Không chạy lại thí nghiệm theo yêu cầu. Bảng dưới là benchmark đã đo **trước cutover cuối cùng**; cột sau cutover cố ý để “chưa đo”, không phải kết quả giả định.

| Workload | NONE mean elapsed/game (s) | STANDARD mean elapsed/game (s) | Games/min NONE → STANDARD | Overhead trước cutover | Sau cutover |
|---|---:|---:|---:|---:|---|
| Python / Normal | 2.5962 ± 0.0424 | 2.6350 ± 0.0208 | 23.11 → 22.77 | 1.49% | Chưa đo |
| C++ MCTS / Python patterns / Normal | 0.5075 ± 0.0061 | 0.5240 ± 0.0090 | 118.22 → 114.51 | 3.24% | Chưa đo |
| Python / Renju | 79.2533 ± 0.4710 | 79.1801 ± 0.0760 | 0.7571 → 0.7578 | −0.09% | Chưa đo |
| C++ MCTS / Python patterns / Renju | 78.2759 ± 0.4439 | 78.5390 ± 0.3591 | 0.7665 → 0.7640 | 0.34% | Chưa đo |

Protocol cũ: ba cặp lặp, một game/lặp, 200 simulations/move, một worker, cùng seed và uniform evaluator. Selfplay dùng synthetic evaluator, không chứng minh workload neural-network training. Mỗi phép đo cũ ghi root-writer/database metrics; không có post-cutover numbers, queue latency, peak RAM hay variance theo worker để nghiệm thu KPI hiện tại.

## 10. Tournament benchmark trước/sau

Không chạy lại tournament. Benchmark trước cutover:

| Workload | NONE elapsed/run, 2 games (s) | STANDARD elapsed/run (s) | Games/min NONE → STANDARD | Overhead trước cutover | Sau cutover |
|---|---:|---:|---:|---:|---|
| Python MCTS / Normal | 12.1857 ± 0.5964 | 12.3318 ± 0.4439 | 9.85 → 9.73 | 1.20% | Chưa đo |
| C++ MCTS / Python patterns / Normal | 9.6872 ± 0.1552 | 9.6812 ± 0.0775 | 12.39 → 12.40 | −0.06% | Chưa đo |
| Python MCTS / Renju | 59.2268 ± 0.6681 | 59.4036 ± 0.2607 | 2.03 → 2.02 | 0.30% | Chưa đo |
| C++ MCTS / Python patterns / Renju | 58.1439 ± 0.4884 | 58.4898 ± 0.4186 | 2.06 → 2.05 | 0.59% | Chưa đo |

Protocol cũ: ba cặp lặp, hai color-swapped games/cặp, `az_v0` vs random, 200 simulations, một worker, seed 2026. Số âm là noise, không phải speedup. Kết quả cũ không chứng minh overhead sau khi writer/readers được chuyển đổi; KPI ≤5% chưa được xác minh.

## 11. Python/C++ correctness

Không đổi luật, MCTS selection/backup, RNG, backend selection, model, training target hoặc tournament scheduling. Root-stat parity Normal/Renju được xác minh trong task này bằng native build; focused test `3 passed`, full trajectory parity chưa được chứng minh. Không chạy AI research games.

## 12. Crash/resume/data integrity tests

Đã xác minh trong tác vụ này: bộ kiểm thử cutover tập trung `109 passed`; importer sau khi hỗ trợ TXT `11 passed`; native parity/root-telemetry `3 passed, 6 deselected`; `test_az_player.py` `2 passed, 2 skipped`; các kiểm tra identity/metadata khi resume training nằm trong bộ kiểm thử. Analytics chỉ đọc của một experiment đã import trả về 6 games/3 pair groups; bộ lọc seed T35 mặc định không tìm thấy experiment phù hợp. Không chạy lại AI research games.

Chưa có crash injection quanh transaction commit, queue-full, lock timeout, disk-full, writer failure, abrupt WAL recovery, concurrent production writers hoặc restore thử trên database research hiện tại. Không test bằng cách làm gián đoạn hoặc ghi đè database local. Existing backup/restore test dùng temporary DB.

## 13. Elo verification

Controlled fixture: một Black win từ 1200/1200, K=32 cho 1216/1184; recalculation idempotent và tạo hai rating events. Historical tournament game rows chưa được dùng để ghi rating history: `rating_systems=0`, `rating_events=0` sau import. Recalculate Elo trên production history chưa chạy để không ghi thêm kết quả suy diễn. Rating pools Normal/Renju vẫn độc lập; advanced Bradley–Terry/Davidson protocol chưa được triển khai.

## 14. Tiêu chí nghiệm thu

| Tiêu chí | Trạng thái |
|---|---|
| Hai SQLite DB local, không commit | Đạt theo `.gitignore`; không có DB tracked cần untrack |
| Self-play/training → selfplay DB | Đạt ở entrypoints mặc định; STANDARD resume dùng lại run ID được chỉ định và từ chối khi metadata sai khác; checkpoint/replay không đổi |
| Tournament/Elo/analytics → tournament DB | Đạt ở entrypoints mặc định; resume/query dùng DB |
| Python/C++ chung schema | Root-stat API parity đã có; full trajectory parity chưa đủ |
| STANDARD root logging đầy đủ | Self-play producer có; tournament root producer chưa wired |
| TRACE selection/expansion/backup | Chưa triển khai |
| Không log file mặc định mới | Đạt ở cutover entrypoints; explicit JSON export/stdout không phải log mặc định |
| Import lịch sử + reconciliation | 1.133 file import; 3 file được giữ lại do conflict/mixed identity |
| Không xóa nguồn cũ | Đạt; không cleanup |
| Export/import thành viên | Tournament bundle đạt; self-play run exchange chưa có |
| Crash/resume/recovery | Resume/idempotency/backup fixture có; crash-injection chưa có |
| Elo từ historical production results | Chưa chạy; source data không bị sửa |
| STANDARD overhead ≤5% sau cutover | Chưa đo; không nghiệm thu |
| Integrity | Cả hai DB integrity `ok`, FK violations 0 |
| Tài liệu truy vấn/bảo trì | README, tournament guide, SPEC và ADR cập nhật |

## 15. Vấn đề đã biết và rủi ro

- Benchmark post-cutover không chạy vì yêu cầu không rerun thí nghiệm; overhead target không được xác nhận.
- Tournament MCTS roots/TRACE và self-play experiment exchange còn thiếu.
- Một số CSV/metadata chỉ giữ raw event; không thể tính lại game-level Elo từ aggregate.
- Ba source identity conflict/mixed chưa normalize; không tự quyết định gộp hoặc bỏ.
- Checkpoint hash/artifact provenance chưa được reconcile với toàn bộ replay/checkpoint lịch sử.
- Schema có version 1 nhưng chưa có upgrade migration.
- Historical root payloads trong raw events không được chuyển thành normalized compressed BLOB.
- Backup/restore chỉ kiểm thử helper trên temporary DB; production DB chưa restore thử.
- Không thể suy ra provenance của checkpoint/replay cũ nếu nguồn không ghi run ID ổn định; muốn resume checkpoint như vậy cần identity DB mới được chỉ định rõ.
- Không có đề xuất cleanup trong task này; nguồn chỉ có thể được xem xét riêng sau backup và đối chiếu.

## 16. Hướng dẫn sử dụng, truy vấn và bảo trì

Mặc định database local:

```powershell
.venv\Scripts\python.exe scripts/train.py --db data\selfplay.sqlite --config configs\normal.yaml
.venv\Scripts\python.exe scripts/tournament.py --db data\tournament.sqlite --run-id RUN_ID --dry-run
.venv\Scripts\python.exe scripts/tournament.py --db data\tournament.sqlite --run-id RUN_ID --resume
```

Inventory chỉ đọc mặc định; import là additive và không xóa nguồn:

```powershell
.venv\Scripts\python.exe scripts/migrate_logs_to_sqlite.py
.venv\Scripts\python.exe scripts/migrate_logs_to_sqlite.py --import
```

Exchange tournament experiment chỉ khi yêu cầu, bundle không ghi đè file đã tồn tại:

```powershell
.venv\Scripts\python.exe scripts/exchange_experiment.py export --db data\tournament.sqlite --experiment-id RUN_ID --out exports\RUN_ID.sqlite
.venv\Scripts\python.exe scripts/exchange_experiment.py import --db data\tournament.sqlite --bundle exports\RUN_ID.sqlite
```

Research queries và integrity checks:

```sql
SELECT run_id, started_at, backend, rule_set, seed FROM runs ORDER BY started_at;
SELECT iteration, global_step, metrics_json FROM training_steps WHERE run_id=? ORDER BY global_step;
SELECT game_id, winner, end_reason, moves_count FROM games WHERE run_id=? ORDER BY local_game_id;
SELECT experiment_id, rule_set, winner, count(*) FROM games GROUP BY experiment_id, rule_set, winner;
PRAGMA foreign_key_check;
PRAGMA integrity_check;
```

`scripts/analyze_t35.py --db data/tournament.sqlite` queries SQLite; JSON output requires explicit `--out`. `scripts/recalculate_elo.py` requires explicit experiment/K/initial rating. Checkpoints remain under `checkpoints/`; replay remains in its current format. Never delete `-wal`/`-shm` while a database is open. Git sync does not import, replace or merge local SQLite; use the explicit experiment exchange command.

## 17. Kiểm tra migration và cleanup tiếp theo

**Chưa nghiệm thu migration/cleanup; không file legacy nào bị xóa.** Các số liệu trong mục 7–8 là lịch sử lần trước, không phải trạng thái DB mới nhất.

### Snapshot và bảo vệ dữ liệu

- Snapshot SQLite được tạo bằng `sqlite3.Connection.backup` từ kết nối read-only; không sao chép riêng file DB khi WAL có thể tồn tại.
- Backup local nằm tại `../gomoku_logging_backup_20261119T173723541431Z/`, ngoài `logs/`, `evidence/` và `data/`. Đây là bản sao trên cùng ổ đĩa, không bảo vệ khỏi hỏng ổ đĩa.
- `legacy.zip` chứa cả 1.244 file trong `logs/` và `evidence/`; ZIP test và đối chiếu SHA-256 từng file với nội dung đọc lại đều đạt. Không phát hiện file thay đổi trong lúc đọc. `manifest.json` ghi path, kích thước, SHA-256 và trạng thái đối chiếu với `source_imports`; SHA-256 của manifest: `0412fcc7ee3ff4b0c440f91f7e765b6e6af6a8292d92239df24420e460605738`.
- Hai bản restore-check được tạo riêng bằng SQLite backup API từ snapshot: `integrity_check=ok`, `foreign_key_check` không có dòng và số hàng từng bảng khớp snapshot. Chưa thử phục hồi toàn bộ ứng dụng từ checkpoint/replay của production.
- Kiểm tra process Windows chỉ thấy các Python runner phục vụ phiên audit, không thấy lệnh training/tournament trong process list tại thời điểm kiểm tra. Đây không phải bảo đảm không có writer bắt đầu sau đó.

| Bảng | Snapshot trước thao tác tiếp theo | Báo cáo lần trước |
|---|---:|---:|
| Self-play games | 49.270 | 25.220 |
| Self-play moves | 713.023 | 446.634 |
| Self-play training steps | 674 | 196 |
| Tournament games | 7.520 | 20 |
| Tournament matches | 3.760 | 10 |

Snapshot có 24.050 self-play game rows với run ID không bắt đầu `legacy:` và 7.500 tournament game rows với experiment ID không bắt đầu `legacy:`. Phân loại này không tự chứng minh lineage hoặc thời điểm của từng ván. Sau snapshot chỉ thêm ba quarantine events; số rows game không thay đổi.

### Reconciliation và blocker

- `reconciliation-manifest.json` phân loại 1.133 file có cùng hash với một source import, 3 file quarantined và 108 file ngoài parser scope chưa đối chiếu; SHA-256 manifest: `adaf508d791a6a526982624d97f56c04eee21c1fb441189b65a1ac6c4b0e682a`. `imported_same_hash` không chứng minh game-level deduplication hoặc khả năng tái tạo byte từ DB.
- Trong snapshot có 57 nhóm hash lặp trong `selfplay.source_imports` và 79 nhóm trong `tournament.source_imports`. Ví dụ hash `0182cfe4bce6f084f8d33c2f01d83bd839d8a7f7ddce40b1343155bd8205b6dc` xuất hiện ở chín source identities của các snapshot; một số đường dẫn thuộc tournament nhưng được importer phân loại game self-play. Cùng hash không đủ để kết luận mọi nguồn là cùng một sự kiện, nhưng importer hiện tạo identity theo path và có thể đếm bản sao snapshot như các ván riêng. Cần đối chiếu lineage/game identity trước khi sửa thống kê.
- Đối chiếu cụ thể nhóm hash trên: chín source identities tạo chín run trong selfplay DB, tổng 900 game rows; có 100 `local_game_id` và 100 payload phân biệt, mỗi local ID có chín payload giống hệt. Đây là bằng chứng các bản ghi snapshot lặp trong DB, không phải bằng chứng để tự động xóa một run hay thay đổi lịch sử rating.
- Audit toàn bộ nhóm source hash lặp: trong selfplay có 40 nhóm có normalized run, với 19.274 game rows bổ sung có cùng `(local_game_id, payload_json)` trong cùng nhóm hash nguồn; trong tournament, không nhóm hash lặp nào ánh xạ tới normalized experiment qua metadata identity. Đây là **ứng viên trùng**, không chứng minh được toàn bộ là cùng sự kiện và không được tự xóa. Chi tiết identity/count nằm trong `duplicate-candidate-audit.json` bên cạnh backup; SHA-256 `e1f7b7eb2d2bb774b11654cc4743caf0e189103c1cbe6a5414b25f21281324b0`.
- Hai recovery histories có cùng SHA-256 nhưng mỗi source có game ID bị lặp với payload xung đột; `logs/integration_merged_20261114/games.jsonl` có nhiều identity/ruleset. Ba nguồn được lưu nguyên byte trong `events` dưới event `legacy_quarantine`, payload zlib+base64, source identity, SHA-256 và lý do; không được đưa vào bảng games/Elo. Hai event thuộc selfplay DB, một thuộc tournament DB. Giải nén và đối chiếu byte/hash cả ba nguồn đã đạt; chạy lại không thêm event. File nguồn và `legacy.zip` vẫn giữ nguyên.
- Không chạy importer bình thường ghi vào production vì chưa có cơ chế phân biệt snapshot, resume bổ sung và thí nghiệm độc lập được kiểm chứng. Không sửa Elo hoặc xóa các rows đã nhập trước đó. Không đề xuất xóa cho đến khi reconciliation game-level, kiểm tra writer ngay trước cleanup và khôi phục ứng dụng đều đạt.

### Kiểm thử trong lượt kiểm tra

- `.venv/Scripts/python.exe -m pytest -q tests/test_train_resume_logging.py tests/test_migrate_logs_to_sqlite.py tests/test_logging_v2.py tests/test_elo.py tests/test_experiment_exchange.py tests/test_tournament.py`: **67 passed**.
- `.venv/Scripts/python.exe -m pytest -q tests/test_selfplay.py tests/test_native_mcts.py -k "not parallel_workers and not batching_and_sample_targets"`: **21 passed, 2 deselected**.
- Không chạy lại training/tournament nghiên cứu và không chứng minh full trajectory Python/C++ parity. Không có cleanup, normalized game import mới hoặc thay đổi checkpoint/replay trong lượt này.
- Quarantine regression cùng bộ test resume/import/integrity/Elo/exchange/tournament: **69 passed**. `scripts/quarantine_legacy.py` lưu additive và rollback nếu lỗi; giải nén kiểm tra SHA-256, lặp lại idempotent. `integrity_check=ok` và `foreign_key_check` không có dòng cho cả hai production DB sau khi ghi quarantine; so với snapshot chỉ tăng 2 selfplay events và 1 tournament event, các bảng còn lại giữ nguyên số hàng.
- Đọc trực tiếp `checkpoints/latest.pt` và `checkpoints/buffer_latest.npz` qua `Trainer.load`/`ReplayBuffer.load_npz` trên CPU: iteration=500, global_step=484000, replay=50000 samples, frozen eval batch hiện diện, RNG PCG64. Chỉ tải để kiểm tra; không chạy optimizer hoặc ghi checkpoint. `tests/test_trainer.py tests/test_replay_buffer.py`: **16 passed**. Không suy ra checkpoint và buffer được ghi nguyên tử cùng nhau hoặc chứng minh mọi khía cạnh của resume.

## 18. Đối soát tiếp 108 file và snapshot trùng

- Đã kiểm tra lại SHA-256 của cả 1.244 file nguồn so với backup; không có file thay đổi. `reconciliation-manifest-v2.json` trong thư mục backup lưu từng path, kích thước, hash, phân loại và archive members; SHA-256 `ec3f1eb582f1f5040efb912c6ddf77a2dfe8429d12c8b4e1c1bb4a7b02a26710`.
- Trong 108 file trước đây ngoài parser scope, hai file `matrix.csv.before_dedup_*` và `summary.json.before_dedup_*` là aggregate tournament lịch sử. Cả hai đã được quarantine lossless trong tournament DB, kiểm tra byte/hash khi đọc lại và thử lặp không tạo bản ghi mới; file gốc vẫn giữ.
- 106 file còn lại được phân loại và giữ nguyên: 9 checkpoint/trọng số `.pt`, 41 mã nguồn snapshot `.py`, 49 bytecode `.pyc`, 3 cấu hình `.yaml`, 2 hình kết quả `.png`, 1 notebook `.ipynb` và 1 research bundle `.zip`. Mã nguồn/config/weights/notebook/ảnh có thể là evidence hoặc phục vụ tái lập; không coi là rác.
- `t35_colab_bundle.zip` có 114 thành viên, ZIP CRC test đạt; 113 thành viên có file ngoài archive cùng SHA-256. Trong 70 thành viên JSON/JSONL/CSV, 69 trùng byte với source đã ghi nhận import; file `transfer_integrity.json` không trùng đã được quarantine lossless riêng trong tournament DB với identity `evidence/colab_transfer/t35_colab_bundle.zip!/transfer_integrity.json`. File archive và toàn bộ source được giữ nguyên. Trùng byte không chứng minh trùng sự kiện.
- Trong 40 nhóm hash nguồn lặp có normalized self-play rows, 37 nhóm có ít nhất một source dưới `logs/`; 30 nhóm có source đường dẫn tournament, tương ứng 13.136 trong 19.274 rows ứng viên lặp. Một ví dụ game log thiếu cả `run_id`, `seed` và `pair`, nên không thể tự suy ra run lineage chỉ từ payload/path. Không tự đổi rule set, run ID hoặc kết quả.
- Sau khi thêm ba quarantine events trong lượt này, selfplay DB có 49.270 games/2 quarantine events; tournament DB có 7.520 games/4 quarantine events/0 rating events. Cả hai qua `PRAGMA integrity_check` và `foreign_key_check`. Không thêm game, tính Elo, xóa source, hoặc sửa checkpoint/replay.
- **Cleanup chưa đủ điều kiện:** các file có byte trùng có thể vẫn là bằng chứng nghiên cứu hoặc nguồn để giải quyết provenance của 19.274 candidate rows; chưa có đối chiếu danh tính đầy đủ cho mọi nguồn/game, không có backup ngoài ổ đĩa hiện tại và chưa xác thực ứng dụng resume end-to-end từ backup. Không xóa file nào.

### Kiểm tra bổ sung trên snapshot

- So sánh toàn bộ hàng theo khóa chính giữa snapshot ban đầu và production read-only: selfplay `runs` 314, `games` 49.270, `moves` 713.023, `mcts_roots` 266.389, `training_steps` 674; tournament `experiments` 27, `agents` 35, `matches` 3.760, `games` 7.520, `moves` 154.911, `rating_systems` 0 và `rating_events` 0. Từng hàng trong các bảng này giống nhau; quarantine chỉ thêm events, không thay đổi research rows.
- Chạy lại `inventory`/`import_files` trên **hai bản clone tạm** tạo bằng SQLite backup API: 1.136 nguồn, 1.133 `already_imported`, ba `skipped`, không có import mới; số hàng `games`, `events`, `source_imports` không đổi và cả hai clone đạt integrity/foreign-key checks. Tổng bộ đếm `skipped_or_existing` là 1.137 vì một backend source phân vùng vào hai database. Ba file skipped vẫn được bảo vệ bằng quarantine trong production và backup.
- Khi hủy clone tạm, Windows từ chối xóa một file do connection SQLite còn mở; đã đóng connection và xóa riêng thư mục clone do phép thử tạo. Không xóa hoặc thay đổi source nghiên cứu.
