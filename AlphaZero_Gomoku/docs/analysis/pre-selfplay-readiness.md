# Chuẩn bị continuation từ checkpoint 25k

## Quyết định và phạm vi

Báo cáo này chuẩn bị một run mới, không chạy self-play, optimizer update hay tournament ngoài các kiểm thử tự động của suite. Không nhập replay legacy; không thay đổi checkpoint hay SQLite lịch sử. `checkpoints/latest.pt` có SHA-256 `f7f521c1cc37a5054ed854201b087e484e9fcef3d4d6b1877ab81dff2e51d7e4`, Normal, ResNet 6 block × 64 filters, iteration 500. Kiểm tra bootstrap trong bộ nhớ bằng CPU: replay rỗng, optimizer Adam không có state, global step 0; không ghi generation hoặc chạy update. CUDA và C++ MCTS 0.1.0 khả dụng tại thời điểm kiểm tra; ổ đĩa còn 374.983.610.368 bytes, `os.access` cho biết thư mục làm việc có quyền ghi. Đây không phải kiểm tra mất điện.

Pilot trước có 100/100 ván Đen thắng, 100/100 bắt đầu tâm và 92/100 dài 9–11 ply; tactical 15/24 so với parent 16/24, 1/20 trước Heuristic. Không có bằng chứng cấu hình mới tăng sức chơi: opening ngẫu nhiên tăng số prefix có thể sinh, không chứng minh chất lượng phân phối hay năng lực phòng thủ.

## Cấu hình run

`configs/normal_continuation.yaml`: Normal; ResNet không đổi; 200 simulations; `c_puct=1.5`, Dirichlet α=0,15 và ε=0,25; replay uniform 50.000 vị trí, symmetry hiện hành; Adam reset, LR hằng `1e-4`, batch 32; 100 updates/iteration, 50 game/iteration, 100 iteration = tối đa 5.000 game và 10.000 updates. LR không kế thừa cosine 200/200 đã hết của pilot; kiểm tra `_set_lr()` tại step 0 và 10.000 đều là `0.0001`. LR hằng trong horizon này là lựa chọn vận hành thận trọng, **không phải cấu hình đã được chứng minh tối ưu**. Run kết thúc sau gate đầu; không tự kéo dài sang 5.000 game tiếp.

`selfplay_opening_plies: 2` chọn hai nước hợp lệ theo RNG/seed từng game từ bàn trống. Chỉ có hai quân trên bàn nên không thể đã thắng hoặc có nước thắng ngay trong Normal. Không ép tỷ lệ kết quả theo màu. Opening được ghi trong `provenance.opening_actions`; các nước ép này không tạo target policy giả. `moves_count` và số sample chỉ tính nước do MCTS chọn; index telemetry là ply thật, vì vậy độ dài toàn ván bằng `moves_count + len(opening_actions)`. Baseline bắt đầu từ bàn trống vẫn dùng `selfplay_opening_plies: 0` trong config của **run mới khác**, không sửa config giữa lúc resume. Đây là hai opening ngẫu nhiên trên bàn, không bảo đảm cả hai quân ở gần nhau; chưa đo tác động đến strength.

## Recovery, lineage và gate

Mỗi iteration dùng checkpoint nguồn có SHA, worker tensor digest, game ID, seed, ruleset và opening trong provenance; replay mới lưu game ID. Outbox chứa game/target trước SQLite commit; generation lưu checkpoint, replay và manifest hash trước khi thay `latest.json`. Resume xác thực config, run ID, pointer, hash và outbox; rollback generation cũ cần `--generation` cùng run ID nhánh mới. Một thư mục run chỉ có một writer. Không coi legacy replay thiếu cursor/lineage là generation xác minh.

Gate v1.3 sau mỗi 5.000 game mới: đúng 100 game paired đổi màu với Heuristic, opening bốn nước, Python MCTS 200, không noise; ghi tournament SQLite, paired Davidson ΔElo/CI và event có tactical holdout theo màu/family cùng SHA candidate. Nếu chưa có best, chỉ promote khi cận dưới 95% CI so Heuristic > 0. Nếu đã có best, 100 game với Heuristic **không** chứng minh hơn best: giữ best, không promote, ghi inconclusive; cần đối kháng best riêng trước khi quyết định. Kết quả 100 game là screening, không phải kết luận Elo nghiên cứu; tactical holdout nhỏ có các biến thể tương quan và chưa biết parent 25k từng thấy các thế tương tự. Không có gate tự động chứng minh candidate tốt hơn checkpoint parent, và không tự động rollback khi thua Heuristic.

## Lệnh bắt đầu sau báo cáo

Chạy từ `AlphaZero_Gomoku` trong PowerShell. Các đường dẫn run, SQLite và log dưới đây chưa tồn tại tại thời điểm kiểm tra; nếu đã được tạo, chọn tên mới, không xóa/ghi đè. Dừng nếu SHA parent không còn khớp hoặc backend C++ không nạp được. Đây là lệnh cho người dùng chạy sau, **chưa chạy trong nhiệm vụ này**:

```powershell
Get-FileHash checkpoints/latest.pt -Algorithm SHA256
.\.venv\Scripts\python.exe -u scripts/train.py `
  --config configs/normal_continuation.yaml `
  --bootstrap checkpoints/latest.pt --bootstrap-optimizer reset `
  --out-dir checkpoints/az_25k_continuation_v1 `
  --db data/az_25k_continuation_v1.sqlite `
  --tournament-db data/az_25k_continuation_v1_tournament.sqlite `
  --run-id az-25k-continuation-v1 --seed 314159 --workers 1 `
  --backend cpp --mcts-backend cpp --patterns-backend python --logging-level standard `
  2>&1 | Tee-Object -FilePath data/az_25k_continuation_v1.console.log -Append
```

Sau gián đoạn, dùng **cùng config, backend, seed, run ID và các đường dẫn**, thay hai tham số bootstrap bằng `--resume`:

```powershell
.\.venv\Scripts\python.exe -u scripts/train.py `
  --config configs/normal_continuation.yaml --resume `
  --out-dir checkpoints/az_25k_continuation_v1 `
  --db data/az_25k_continuation_v1.sqlite `
  --tournament-db data/az_25k_continuation_v1_tournament.sqlite `
  --run-id az-25k-continuation-v1 --seed 314159 --workers 1 `
  --backend cpp --mcts-backend cpp --patterns-backend python --logging-level standard `
  2>&1 | Tee-Object -FilePath data/az_25k_continuation_v1.console.log -Append
```

Để rollback có chủ ý, dùng `--resume --generation <generation-id>` với **run ID mới** và giữ nguyên thư mục generation/SQLite; chỉ chọn generation nằm trong committed history. Không tự xóa pending/orphan artifacts. Nếu validation hash, lineage hoặc recovery thất bại, dừng và giữ nguyên tất cả artifact để điều tra; không bootstrap đè.

## Thay đổi và kiểm thử

- `az/selfplay.py`, `config.py`: opening 0/2 ply, seed/provenance, chỉ học từ nước MCTS; kiểm thử legality, tái lập và lineage.
- `scripts/train.py`: gate v1.3, tactical screening, bảo vệ promotion khi chưa so best; cho phép output directory mới dưới `checkpoints/` mà không ghi đè checkpoint parent. Regression tests kiểm tra gate, resume và đường dẫn bootstrap.
- `configs/normal_continuation.yaml`, `SPEC.md`: cấu hình run và cập nhật hợp đồng opening/gate. Không đổi luật, ResNet, MCTS budget hay thuật toán sampling.
- Full suite trên Windows: `OMP_NUM_THREADS=1 .venv/Scripts/python.exe -m pytest -q -p no:cacheprovider` — **375 passed, 2 skipped, 57,99 s**. Lần đầu có 2 lỗi fixture mới do mock truyền `str` vào helper yêu cầu `Path`; đã sửa fixture và chạy lại full suite. Hai skip không được tính là pass. Suite có kiểm thử mô phỏng self-play/training trong môi trường test; không có production run.
- `scripts/train.py --help`, nạp checkpoint, bootstrap trong bộ nhớ, config/LR và C++ backend đã được kiểm tra. `graphify update .` hoàn tất; CLI cảnh báo package graphify 0.9.97 cũ hơn skill 0.9.106.

**Sẵn sàng vận hành có điều kiện:** có thể bắt đầu run 5.000 ván bằng lệnh trên nếu fingerprint/đường dẫn còn khớp và người dùng chấp nhận rủi ro chất lượng chưa được chứng minh. Gate đầu nằm ở cuối run; không suy ra rằng model đã mạnh hơn từ việc suite pass hoặc có thêm dữ liệu.

## Theo dõi và giới hạn

JSON mỗi iteration và `training_steps` ghi loss, policy/value loss, số sample, thời gian và global step; fixed batch không phải holdout độc lập theo game. Bảng `games`/`moves` cho phép thống kê winner theo màu, opening, độ dài đầy đủ, legal moves và source checkpoint; gate event và tournament SQLite chứa tactical, W/D/L, ΔElo/CI. Theo dõi dung lượng, integrity, tỷ lệ hai màu và phân phối độ dài; không can thiệp để ép tỷ lệ 50/50. Nếu xảy ra mismatch lineage/hash, dữ liệu không hợp lệ, NaN hoặc hết dung lượng: dừng và giữ artifacts. Nếu tactical hoặc kết quả Heuristic suy giảm, dừng ở gate 5.000 để xem xét thủ công; không có ngưỡng suy giảm có ý nghĩa thống kê đã được xác nhận. Windows process-crash tests không chứng minh độ bền directory-entry khi mất điện, hay an toàn concurrent writers. Phải so strength trên opening/seed độc lập và nhiều seed trước khi tuyên bố cải thiện.

## Fingerprint để cố định cấu hình thực thi

SHA-256 tại thời điểm chuẩn bị (Python 3.10.6, PyTorch 2.14.1+cu126, CUDA runtime 12.6, NumPy 2.2.6, PyYAML 6.0.3). Đối chiếu trước khi chạy; đây là fingerprint, không ngăn filesystem sửa file và không thay thế snapshot mã nguồn/dependencies đầy đủ. Worktree có thay đổi từ trước và không được commit.

```text
da17ddd8ce9461f62b2b1dd07e6043c6c11d43db0f62b3d2c3cc6afb357d99f0  configs/normal_continuation.yaml
c1854825b78404dfb130014d205ff423ea4fdcfafeba486f736ee28962cbfdfb  scripts/train.py
5ee1a4f76b18c50f1185c023a3e20328452af5130929d7ab6299a49f76ee46ef  scripts/tactical_benchmark.py
8039fd9317d9f9e9db58335106ea2b6384040dbdb8a9f3db1efe4fce07c0b686  scripts/tournament.py
875a4cfce1fd0f00bd603862d6dde9d9023e77304ca969d2e14ee9ac470af077  config.py
2e334bd94016b3599e3dfa3d5d07e1508960b1064e86bd3ba82667d653f38ad1  az/selfplay.py
f0db762d4f2a7bcbd82008d0731e91a4d1091cbd64cc6d615ab88c3df195cf40  az/trainer.py
cd8fc66a9c026f67613a1913e27e43bbdc54f3e68241de27ee0b6bc66d6b429f  az/training_recovery.py
955cc6b9b47b6ec2664bea32a9c85e0e27b1752816195090f16f773a3a813515  az/replay_buffer.py
037a5d86340e6ba58a4d44609c193fabd1e58d9752c01abd72bba15cf10acc64  az/mcts.py
6f712d5d110a540cb81a6b7e71641c4194b5985632293872fbaf31d667c83781  az/model.py
06a4d90c72eb98ffe2cb3bd5ecff3644fe162c52ef9bb27198c5511d65bf107c  az/elo.py
e3608cf6c333b3ca170cfdd95de30e684cf6de7fb7ebc610321a34cf02b95295  gomoku/board.py
892339654b92ef9e7a95cc74e5e925a9862fca541c5266b876e3ba4c95827633  gomoku/rules/normal.py
70af8c6f43a9dd2013fcf1207a0ce9c374f9573fc9bc5333605d62592492a226  gomoku_mcts_native.cp310-win_amd64.pyd
```
