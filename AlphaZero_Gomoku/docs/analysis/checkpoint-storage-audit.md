# Kiểm toán dung lượng checkpoint `az_25k_continuation_v1`

## Kết luận

Ảnh chụp metadata ngày 20-11-2026 cho thấy phần lớn dung lượng đến từ **hai bản snapshot toàn bộ replay mỗi iteration**, không phải chỉ từ mẫu self-play mới. Generation ghi replay cùng checkpoint; outbox recovery cũng ghi replay đầy đủ bên cạnh samples mới. `np.savez` hiện không nén. Đây là cơ chế có chủ đích để phục hồi, không phải bằng chứng file thừa hoặc lỗi.

**Có thể xóa an toàn theo hợp đồng hiện tại: 0 byte đã được chứng minh.** Không xóa, di chuyển, sửa, nén lại hay deduplicate artifact trong kiểm toán này. Không có chính sách cho phép bỏ rollback lịch sử, chưa xác minh đầy đủ mọi hash, recovery và SQLite, và còn một tournament đang dùng hai checkpoint. Các con số dưới đây là **ứng viên giả định**, không phải lệnh xóa hay dung lượng đã được phê duyệt giải phóng.

Đơn vị dưới đây là GiB = 2³⁰ byte; con số GB trong thống kê ban đầu có thể dùng quy ước khác. Dữ liệu có thể thay đổi theo thời gian; cần kiểm kê lại sau khi mọi writer/reader liên quan đã dừng.

## Kiểm kê thực tế

Đếm bằng `os.walk`/`stat`, không đọc toàn bộ payload. Đây là kích thước logic tổng theo file, không đo dung lượng cụm đĩa hay loại trừ hard link.

| Nhóm | Số file | Byte | GiB | Vai trò |
|---|---:|---:|---:|---|
| `generations/*/replay.npz` | 506 | 22.636.405.051 | 21,0818 | Snapshot replay của từng generation; bao gồm fixed eval batch nếu có |
| `recovery/*/replay.npz` | 500 | 22.232.166.596 | 20,7053 | Snapshot toàn bộ replay sau self-play của outbox |
| `recovery/*/samples.npz` | 500 | 245.171.279 | 0,2283 | Mẫu mới của iteration, có hash và provenance |
| **Tổng NPZ** | **1.506** | **45.113.742.926** | **42,0154** | 1.006 replay snapshot và 500 sample, không có loại NPZ thứ tư độc lập |
| `generations/*/checkpoint.pt` | 506 | 2.979.166.968 | 2,7746 | Model, optimizer và metadata theo generation |
| JSON generation, recovery, pointer | 2.007 | 456.290.262 | 0,4250 | Manifest, records, metrics, `latest.json` |

JSON gồm 506 generation manifest, 500 recovery manifest, 500 `records.json`, 500 metrics và một pointer. Riêng records chiếm 455.615.988 byte. Không thấy `best.json` hay `.pending-*` trong ảnh chụp. `samples.npz` không phải generation snapshot; `recovery/*/replay.npz` mới là bản replay trong outbox. Theo manifest, 500 recovery replay hash đều khác nhau; không có hash nào trùng 501 hash replay generation phân biệt. Vì generation còn lưu eval arrays, không thể suy luận các file có cùng nội dung byte hoặc an toàn thay bằng hard link.

## Manifest, ancestry và khả năng khôi phục

- `latest.json` trỏ tới `fabf0022b6614983ac6001012ad81784`; `previous` là `f43be9e336474950a382a64f15f2ca80`. Tất cả 506 generation duyệt được từ latest qua commit ancestry; không thấy cycle, thiếu generation, checkpoint hay replay trong kiểm tra metadata. Không manifest nào có `commit_previous` trong ảnh chụp này. Không thấy `best.json`.
- Đã **thực sự tính SHA-256** latest checkpoint: `7550af00ccf7e0558cbb23a92c310e7ca8429ded14869e92374cb10a84001ebb`; replay: `b61c15d861b4a90f41b49f20cc5ee963b8b494a7120238b247b50c86b5d886b2`. Cả hai khớp manifest; pointer không đổi trong lúc tính hash. Đây không phải kiểm tra hash toàn bộ 506 generation hay 500 outbox. Hash không chứng minh dữ liệu khôi phục được hoặc bảo vệ nếu cả payload lẫn manifest bị sửa.
- `az/trainer.py:50-85` duyệt `commit_previous`, fallback `previous`, kiểm tra ancestry và chỉ hash **generation đích**. Vì vậy các manifest trung gian vẫn cần để đến rollback point cũ; thiếu payload của một generation thì rollback vào chính generation đó lỗi, kể cả ancestry còn đi xuyên manifest của nó được. `previous` biểu thị parent thực sự được load; `commit_previous` bảo toàn lịch sử commit sau branch (`az/trainer.py:234-285`).
- Resume STANDARD dùng checkpoint và replay generation đã chọn, cursor và hash pairing, run ID gốc, config và SQLite phù hợp. `Trainer.load` phục hồi optimizer/RNG nếu chúng có trong checkpoint; metadata/hashes không chứng minh việc khôi phục hoàn chỉnh đã thành công (`scripts/train.py:384-466`, `az/trainer.py:358-384`). Rollback tường minh chọn ancestor và dùng run ID mới theo quy tắc trong driver. Chưa chạy resume/rollback thực tế; chưa xác minh DB, optimizer, RNG, cursor hay tính đầy đủ games/metrics trong lần kiểm toán này.
- Outbox có identity `(run_id, iteration, source_checkpoint_sha256)` và manifest hash replay, samples, records; checkpoint còn bind manifest và metrics (`az/training_recovery.py:26-32,67-127`, `scripts/train.py:450-466,491-527,559-573`). **Ngay cả khi latest generation đầy đủ, STANDARD resume vẫn có thể bắt buộc outbox mà checkpoint bind**, kể cả để hoàn tất ghi metrics. Outbox đang dở hoặc có thể cần để retry phải giữ. Không suy từ SQLite có game rằng outbox đã thừa.
- `records.json`, samples, config, DB/log và checkpoint provenance có giá trị audit khác nhau: SQLite không tự chứng minh thay thế được toàn bộ bytes trong outbox. Giữ manifest đơn thuần không đủ để tái tạo dữ liệu hay xác minh hash payload đã mất. Chưa chứng minh khả năng tái lập toàn bộ run. SPEC §8.3 nêu single writer, không xóa tự động và chưa chứng minh durability mất điện trên Windows.
- Lúc kiểm tra process, tournament `eval50k-vs40k-400-seed508001` còn chạy và tham chiếu checkpoint generation `f43be9e336474950a382a64f15f2ca80` và `71e7aedb1f394ee782dc84832b130b6a`. Đây không chứng minh không còn consumer khác. Không chạm vào các file này; kiểm tra lại process và các tham chiếu trước mọi thao tác tương lai.

## Thiết kế retention / GC — đề xuất, chưa triển khai

Hợp đồng hiện tại giữ lịch sử generation và rollback tới mọi committed ancestor. Muốn thu hồi snapshot phải **thay đổi chính sách được chấp thuận**, xác định thời hạn rollback/audit, cập nhật SPEC và kiểm thử implementation; không gọi đây là tối ưu không thay đổi hành vi. Lựa chọn bảo thủ là lưu trữ offline toàn bộ artifact lịch sử đã kiểm hash và thử khôi phục trước khi cân nhắc thu hồi bản local; archive không thay thế đường dẫn rollback hiện tại nếu chưa có cơ chế restore và runbook được kiểm thử. Dung lượng chuyển sang archive không phải dung lượng tiết kiệm trên toàn hệ thống.

Thiết kế GC tương lai, mặc định **chỉ dry-run**, fail closed:

1. Chỉ chạy sau khi self-play **và mọi writer/reader liên quan** đã dừng, có cơ chế khóa single-writer được tất cả entrypoint tôn trọng. Nếu không thể chứng minh exclusivity, không apply. Chụp `latest.json`, `best.json` nếu có, identity, file metadata và hash; đọc lại khi kết thúc và ngay trước apply. Kiểm soát symlink/reparse point, traversal và path nằm ngoài root; lỗi, file đang ghi, thay đổi pointer, thiếu hoặc sai hash đều dừng, không tự sửa.
2. Xây đồ thị từ cả `latest` và `best`, cả `previous` lẫn `commit_previous`; bảo toàn tất cả manifest trên đường reachability, cả các nhánh, và mọi checkpoint được external consumer, benchmark, tournament hoặc audit tham chiếu. Đánh dấu generation được giữ theo chính sách: latest, best, các rollback point tường minh và các generation đang được sử dụng. **Đừng coi manifest-only là rollback point.**
3. Đọc recovery identity trong checkpoint của mọi generation được giữ; đánh dấu đầy đủ manifest, replay, samples, records và metrics của outbox được bind, outbox pending/in-flight, các retry/source có thể cần; bảo toàn DB và các bằng chứng audit đã xác định. Cần đối chiếu SQLite theo run/iteration/source và xác minh tính đầy đủ/hash; trường hợp mơ hồ giữ nguyên. `reject_other_source` quét manifest của mọi recovery directory, vì vậy không được để directory thiếu manifest. Nếu outbox cần phục hồi từ archive, phải có cơ chế restore nguyên vẹn trước khi resume.
4. Dry-run in từng path, loại, byte, hash, lý do giữ/ứng viên, generation/identity và tác động rollback/audit; tổng bytes theo nhóm, đồng thời phân biệt kích thước file và dung lượng thực tế có thể thu hồi. Không theo symlink, không wildcard xóa. Báo riêng mục không thể xác minh; các mục đó luôn giữ. Lưu kế hoạch cùng hash của inventory để người vận hành phê duyệt sau khi đối chiếu retention và backup.
5. Apply là tính năng **chưa có**: phải khóa độc quyền, tính lại toàn bộ kế hoạch, so sánh pointer/identity/hash/inventory với dry-run đã duyệt và dừng nếu khác; đưa ứng viên sang quarantine trên cùng volume để có thể hoàn nguyên, ghi nhật ký bền vững, sau đó thử resume và từng rollback point được giữ trên bản sao tách biệt cùng bản sao SQLite. Chỉ xóa quarantine sau thời gian giữ, backup độc lập đã khôi phục thử và phê duyệt riêng. Xử lý crash từng bước, không xóa metadata ancestry, không tự sửa manifest hoặc checkpoint. Phải bổ sung test crash, race, branch, best, dangling reference, symlink và failure-injection trước khi triển khai.

### Ước tính có điều kiện, không phải mức an toàn hiện tại

Ví dụ minh họa: giữ replay của latest, predecessor `f43...` và generation `71e...` đang được tournament tham chiếu; giữ **toàn bộ** checkpoint, manifest và outbox. Khi đó **503 replay generation khác** có tổng **22.487.413.327 byte ≈ 20,9430 GiB** là ứng viên chuyển archive/thu hồi local. Con số tính từ ảnh chụp, không phản ánh block được giải phóng, không xét các tham chiếu bên ngoài chưa phát hiện. Phần này **không an toàn để xóa theo hợp đồng hiện tại**: làm mất rollback trực tiếp tới 503 generation và có thể mất bằng chứng. Chỉ xét sau khi người dùng chấp thuận thu hẹp rollback, kiểm tra đầy đủ consumer, khôi phục thử archive và có GC đã kiểm thử. Bản replay outbox 20,7053 GiB không cộng vào con số trên vì chưa chứng minh dependencies và audit được thay thế; hiện không có mức giải phóng an toàn đã xác minh cho nhóm này.

## Đề xuất cho các lần self-play sau

- **Nén có đo đạc hoặc bit-pack planes.** Latest `replay.npz` là ZIP store không nén: `features.npy` 24.200.128 byte, `pi.npy` 12.100.128, `game_ids.npy` 12.800.128; 50.000 mẫu. Bốn plane nhị phân `uint8` chiếm 484 byte/mẫu; đóng bit có thể giảm lý thuyết còn khoảng 61 byte/mẫu, khoảng 21,15 MB cho 50.000 mẫu trước overhead. Benchmark CPU, RAM, tốc độ ghi/resume và tỉ lệ nén thực tế; giữ kiểm hash, cursor, cùng transform symmetry và compatibility. Không sửa NPZ lịch sử.
- **Game ID compact có version**: `U64` chiếm 256 byte/mẫu. Nếu xác minh mọi ID/contract cho phép biểu diễn 16 byte, tiết kiệm lý thuyết 12 MB/50.000 mẫu; không giả định mọi ID là UUID. Migration phải round-trip chính xác provenance, hỗ trợ đọc format cũ và phân biệt ID thiếu.
- **Giảm nhân snapshot**: thiết kế chunk/delta bất biến hoặc chia sẻ nội dung cho run mới thay vì ghi toàn bộ replay hai lần. Snapshot outbox trước train và snapshot generation có eval arrays/metadata khác nhau; không thay thế trực tiếp hay hard-link dựa trên tên. Cần chứng minh atomic commit, cursor/ring order, crash recovery, refcount/GC, hash, rollback, mixed-version và provenance; so sánh chi phí I/O với cách hiện tại. Không áp dụng lên lịch sử.
- `samples.npz` là mẫu mới, nhỏ hơn nhiều snapshot; không xóa chỉ vì loader hiện xác minh hash mà trả replay/records. Nếu định dựng replay từ samples + base, phải định nghĩa chính xác thứ tự/cursor, giữ base và chứng minh byte/semantic equivalence, recovery và rollback qua crash trước khi thay contract.

## Quy trình sau khi self-play dừng

1. Xác nhận self-play, tournament và mọi consumer liên quan đã dừng hoặc có bảo vệ file/lock được kiểm chứng; không giả định chỉ vì không thấy process training trong một lần chụp. Sao lưu nhất quán checkpoint, outbox, SQLite **cùng WAL/SHM khi cần** bằng phương pháp backup SQLite phù hợp, config và logs; xác minh bản sao trên nơi lưu độc lập.
2. Chạy inventory/dry-run của công cụ tương lai ở chế độ read-only, xác minh toàn bộ hash, ancestry hai cạnh, các pointer, bind checkpoint/outbox/metrics, SQLite và external references; ghi thời điểm và bản thân kết quả. Công cụ này **chưa tồn tại trong báo cáo này**.
3. Trên bản sao cô lập, thực hiện thử STANDARD resume latest và rollback từng điểm được cam kết, kiểm optimizer/RNG/cursor, DB, metrics và provenance; không chạy bằng thư mục hoặc SQLite sản xuất. Nếu lỗi hoặc không thể xác minh, dừng và giữ toàn bộ.
4. Chốt retention và mức mất rollback/audit, cập nhật SPEC, triển khai và kiểm thử GC; phê duyệt chính xác danh sách sau dry-run mới. Chỉ khi đủ các cổng ở trên mới quarantine rồi xóa theo quy trình được duyệt. Không có lệnh xóa nào được chạy hay khuyến nghị chạy ngay.

**Phạm vi bằng chứng:** chỉ đọc metadata toàn cây, manifest, payload latest để tính hash và ZIP metadata; tham khảo source/SPEC. Không hash toàn bộ artifact, không chạy restore, training, benchmark hay test; không kiểm tra mọi external reference và không đo hiệu quả nén thực tế. Các test crash có trong source không phải kết quả chạy của cuộc kiểm toán này.
