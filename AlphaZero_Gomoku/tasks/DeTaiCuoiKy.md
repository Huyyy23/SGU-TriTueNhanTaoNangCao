ĐỀ TÀI CUỐI KỲ MÔN TRÍ TUỆ NHÂN TẠO NÂNG CAO

CHỦ ĐỀ: XÂY DỰNG VÀ ĐÁNH GIÁ GAME AI ĐA CẤP ĐỘ

1. MỤC TIÊU
Đề tài giúp sinh viên vận dụng tổng hợp kiến thức về tác tử thông minh, tìm kiếm có đối thủ, học sâu
và học tăng cường thông qua việc phát triển một trò chơi có các tác tử AI khác nhau. Sau khi hoàn
thành, sinh viên có khả năng:
•  Mô hình hóa trò chơi thành môi trường ra quyết định: trạng thái, hành động hợp lệ, hàm chuyển

trạng thái, điều kiện kết thúc và phần thưởng.

•  Thiết  kế  tác  tử  Random,  chiến  lược  luật/heuristic,  Minimax/Alpha-Beta  và  MCTS;  giải  thích

nguyên lý và các giả định.

•  Xây dựng ít nhất một tác tử học được (Neural Network hoặc Reinforcement Learning), mô tả dữ

liệu huấn luyện hoặc quy trình tự chơi.

•  Đánh giá công bằng sức mạnh thi đấu và chi phí tính toán của từng tác tử; phân tích giới hạn và

các tình huống thắng/thua.

•  Phát triển giao diện chơi người–máy và máy–máy, tái lập thực nghiệm và trình bày kết quả có

căn cứ.

2. HÌNH THỨC THỰC HIỆN
•  Nhóm tối đa 04 sinh viên; mỗi nhóm chọn 01 game chính. Khối lượng thực nghiệm và phân công

phải tương xứng với số thành viên.

•  Đề tài phải có câu hỏi nghiên cứu cụ thể, ví dụ: MCTS cải thiện tỷ lệ thắng so với Alpha-Beta
trong cùng thời gian suy nghĩ hay không? Self-play có cải thiện tác tử học được so với mô hình
bắt chước luật không?

•  Khuyến khích tự xây dựng lõi game và tác tử. Nếu dùng engine, thư viện hoặc mã nguồn có sẵn

phải nêu rõ phần kế thừa và phần do nhóm thực hiện.

•  Bắt buộc có demo chơi được và thí nghiệm thực tế; không đánh giá cao việc liệt kê nhiều thuật

toán nhưng cài đặt không đúng hoặc thiếu phân tích.

•  Sinh viên được phép đề xuất đề tài khác ngoài chủ đề Game AI nếu phù hợp nội dung, mục tiêu
và chuẩn đầu ra của học phần Trí tuệ nhân tạo nâng cao. Trước khi triển khai, nhóm phải trao đổi
với giảng viên và được thống nhất về bài toán, phạm vi, phương pháp AI, sản phẩm và tiêu chí
đánh giá. Đề tài thay thế phải có thực nghiệm, baseline, so sánh và phân tích kết quả với mức độ
tương đương; không tự ý thay đổi đề tài sau khi đã được duyệt.

3. NỘI DUNG YÊU CẦU
3.1. Lựa chọn trò chơi và mô hình hóa bài toán
•  Có thể chọn: Connect Four (7×6), Tic-Tac-Toe mở rộng, Caro/Gomoku (quy định rõ kích thước
bàn và luật thắng), Othello/Reversi, cờ đam, mini-Shogi hoặc Shogi giới hạn; trò chơi khác phải
được giảng viên chấp thuận.

•  Mô tả luật chơi, vai trò người chơi, quan sát đầy đủ/không đầy đủ, tính xác định, trò chơi tổng

bằng không hay không, điều kiện hòa và độ phức tạp không gian trạng thái.

•  Định  nghĩa  API  game  thống  nhất:  reset(),  legal_actions(),  step(action),  is_terminal(),  result(),

render(), clone()/undo(); cấm nước đi bất hợp lệ; kiểm thử logic chiến thắng/thua/hòa.

•  Xác định input: trạng thái bàn cờ và người đến lượt; output: hành động hợp lệ, phân phối xác suất

hành động hoặc giá trị trạng thái; reward theo góc nhìn người chơi.

3.2. Các cấp độ tác tử AI
Cấp
1
2

Tác tử
Random
Rule-based / Heuristic

Yêu cầu kỹ thuật
Chọn đều từ các nước đi hợp lệ; làm baseline.
Ưu tiên thắng ngay, chặn đối thủ, kiểm soát vị trí; nêu thứ tự ưu tiên và xung đột luật.

Trang 1

3
4
5

Minimax / Alpha-Beta
MCTS (UCT)
Neural Network

Giới hạn độ sâu; hàm lượng giá trạng thái; so sánh số nút và thời gian có/không cắt tỉa.
Selection–Expansion–Simulation–Backpropagation; khảo sát số lần mô phỏng.
Học policy hoặc value bằng supervised learning từ ván tự sinh / expert; tách
train/validation/test theo ván.

Kết hợp policy/value dự đoán với MCTS; ghi rõ các thành phần đã/ chưa hiện thực.
Mạng policy-value, tìm kiếm PUCT, dữ liệu self-play, cập nhật lặp và đối đầu checkpoint.

Reinforcement Learning  Q-learning cho game nhỏ hoặc policy gradient / deep RL / self-play; đường cong học tập.
AlphaGo-inspired
AlphaZero-inspired

6
7
8
Lưu  ý  thuật  ngữ:  cấp  7–8  là  phiên  bản  mô  phỏng  cảm  hứng  từ  AlphaGo/AlphaZero,  không  được
khẳng định tái hiện đầy đủ hệ thống nguyên bản nếu chưa cài đặt đủ thành phần. Các cấp thể hiện
mức độ phức tạp phương pháp, không mặc nhiên tương ứng với sức mạnh thực tế.
3.3. Phạm vi bắt buộc và nâng cao
•  Bắt buộc: hoàn thiện engine, giao diện/demo; 04 tác tử cấp 1–4; tối thiểu 01 tác tử học được ở

cấp 5 hoặc 6; kiểm thử và giải đấu đối kháng.

•  Khuyến khích: thực hiện  cả cấp 5 và  6; tích hợp MCTS với mạng học được (cấp 7); vòng lặp

AlphaZero-inspired (cấp 8); tối ưu hóa tìm kiếm và khả năng giải thích nước đi.

•  Các nhóm chọn game phức tạp có thể đề xuất phạm vi bàn cờ/luật rút gọn nhưng phải ghi rõ và
được duyệt trước. Không dùng mạng chưa qua huấn luyện làm bằng chứng rằng đã triển khai cấp
Neural/RL.

3.4. Huấn luyện và dữ liệu
•  Dữ liệu có thể được sinh từ self-play, expert agent hoặc các ván game hợp lệ; mô tả số ván, cách

ghi nhãn, cách loại trùng và tỷ lệ phân chia.

•  Nếu huấn luyện policy từ chuyên gia, ghi rõ nguồn nhãn và nguy cơ học lại heuristic; nếu RL,

ghi phần thưởng, số episode, chiến lược exploration, mạng và siêu tham số.

•  Lưu checkpoint và seed; không sử dụng các ván đánh giá cuối cùng để chọn siêu tham số; khai

báo tài nguyên CPU/GPU và thời gian huấn luyện.

3.5. Thiết kế thí nghiệm đối kháng và độ đo
•  Tổ chức round-robin giữa các tác tử; thi đấu tối thiểu 100 ván/cặp (khuyến nghị 200), chia đều

quyền đi trước/đi sau; dùng nhiều seed khi phương pháp có tính ngẫu nhiên.

•  Cố định phiên bản game, luật, cấu hình phần cứng; báo cáo theo ít nhất hai chế độ: ngân sách
tính toán được quy định rõ (thời gian/nước hoặc số mô phỏng) và cấu hình mặc định minh bạch.
Không suy diễn công bằng tính toán khi chỉ áp cùng độ sâu tìm kiếm.

•  Các chỉ số cốt lõi: tỷ lệ thắng/hòa/thua, score = (thắng + 0,5 × hòa)/tổng số ván, thời gian trung
bình  một  nước,  độ  trễ  p95,  số  nút  duyệt/số  mô  phỏng,  RAM/VRAM  (nếu  phù  hợp),  thời  gian
huấn luyện.

•  Trình bày ma trận kết quả đối đầu và khoảng tin cậy (khuyến nghị Wilson/Bootstrap); kiểm tra

ảnh hưởng thứ tự đi trước, seed, search depth và simulation budget.

•  Bắt buộc có ít nhất 02 ablation/sensitivity analysis có ý nghĩa, ví dụ Alpha-Beta bật/tắt cắt tỉa;

MCTS số mô phỏng; heuristic bật/tắt; policy-only so với policy+MCTS.

3.6. Phân tích và diễn giải
•  Trả lời câu hỏi nghiên cứu dựa trên số liệu, không chỉ xếp hạng các cấp AI. Phân tích ván đấu

điển hình, vị trí chiến thuật, trường hợp agent mắc lỗi, tác động của first-player advantage.

•  So sánh chất lượng nước đi với độ phức tạp suy luận; phân biệt sức mạnh thi đấu thực nghiệm

với tính tối ưu đã được chứng minh.

•  Mô  tả  hạn  chế  (ngân  sách  huấn  luyện,  bản  đồ  trạng  thái,  game  nhỏ,  thiếu  dữ  liệu  expert)  và

hướng cải tiến phù hợp.

3.7. Sản phẩm nộp
•  Báo  cáo  hoàn  chỉnh;  slide  thuyết  trình;  poster  tóm  tắt  kết  quả;  demo  ứng  dụng  có  thể  chơi  và

chọn cấp độ AI.

Trang 2

•  Mã  nguồn  Python,  tệp  cấu  hình,  yêu  cầu  thư  viện,  test  tự  động,  checkpoint/weights  nếu  dung
lượng  cho  phép,  log  thi  đấu  thô,  notebook  tạo  bảng/hình,  README  hướng  dẫn  cài  đặt,  train,
evaluate, demo.

•  Một  tệp  ZIP  đặt  tên  NhomXX_TenGame_TTNTNC_HoTen_MaSV.zip;  nhóm  trưởng  nộp  và
cung cấp liên kết GitHub, lịch sử commit, phân công và tỷ lệ đóng góp của từng thành viên.

4. CẤU TRÚC BÁO CÁO
Chương 1. Giới thiệu
Bối cảnh; game, luật và ý nghĩa; mục tiêu, câu hỏi nghiên cứu; phạm vi; sản phẩm.
Chương 2. Cơ sở lý thuyết và công trình liên quan
Tác  tử  trò  chơi;  Minimax,  Alpha-Beta,  MCTS/UCT;  policy/value  networks;  RL  và  self-play;  phân
biệt AlphaGo-inspired, AlphaZero-inspired.
Chương 3. Phương pháp và thiết kế hệ thống
Biểu  diễn  game/state/action/reward;  kiến  trúc  engine,  agent,  UI;  thuật  toán  từng  cấp;  dữ  liệu,  quy
trình train và thiết kế giải đấu công bằng.
Chương 4. Thực nghiệm, kết quả và thảo luận
Môi trường chạy; ma trận đối đầu; win/draw/loss; độ trễ/chi phí; learning curves; ablations; minh họa
ván cờ, phân tích thất bại.
Chương 5. Kết luận và hướng phát triển
Câu trả lời cho câu hỏi nghiên cứu; thành tựu, hạn chế, khả năng tái lập và mở rộng.

5. QUY ĐỊNH VỀ THỰC NGHIỆM VÀ TRUNG THỰC HỌC THUẬT
•  Được phép tham khảo sách, kho mã nguồn mở, công cụ AI và thư viện chuyên dụng; bắt buộc

trích dẫn rõ, chỉ ra phần tự cài đặt và phần kế thừa.

•  Không báo cáo kết quả benchmark hoặc training chưa thực hiện; không chọn riêng các ván thắng

để minh họa kết luận tổng thể; lưu log đầy đủ và mã sinh biểu đồ.

•  Quy tắc game, kiểm thử, seeds, cấu hình, ngân sách tính toán và thiết lập thí nghiệm phải đủ để
chạy lại; kết quả ngoại suy sang game khác phải được nêu là giả thuyết, không phải kết luận.

6. RUBRIC ĐÁNH GIÁ
Điểm
Tiêu chí
10
1. Xác định bài toán/game
15
2. Xây dựng hệ thống
20
3. Tác tử tìm kiếm
15
4. Tác tử học được
15
5. Thiết kế đánh giá
10
6. Kết quả và thảo luận
10
7. Báo cáo, slide, demo
5
8. Tái lập và đạo đức
TỔNG
100
Điểm khuyến khích tối đa +10 (không vượt thang điểm quy định): AlphaGo-/AlphaZero-inspired có
thực nghiệm thuyết phục; thử nghiệm nhiều bàn chơi; giải thích chiến thuật; tối ưu search; nghiên
cứu sâu về tính tổng quát. Điểm cộng không thay thế phần bắt buộc.

Mô tả
Luật, state/action/reward, câu hỏi nghiên cứu, kiểm thử engine
Kiến trúc, API, tính đúng đắn, giao diện demo
Random, heuristic, Minimax/Alpha-Beta, MCTS đúng và có giải thích
NN hoặc RL thực sự huấn luyện; dữ liệu, loss/reward, checkpoint
Giải đấu cân bằng lượt, seed, độ đo, kiểm soát ngân sách và ablation
Ma trận đối đầu, chi phí, thống kê, nguyên nhân và hạn chế
Trình bày khoa học; trực quan hóa; trình diễn ổn định
Code, README, log, dẫn nguồn, đóng góp thành viên

7. GỢI Ý DANH SÁCH ĐỀ TÀI
Game / đề tài
Connect Four

Caro / Gomoku

Tic-Tac-Toe mở rộng

Othello / Reversi
Cờ đam / Checkers
Mini-Shogi / Shogi

Phạm vi khuyến nghị
Bàn 7×6; đầy đủ cấp 1–6; thuận lợi mở rộng
policy/value + MCTS
Bàn 9×9 hoặc 11×11; quy định chính xác thắng 5 và
luật chặn
Bàn 4×4, K=3/4; chuẩn đối chứng nhỏ, huấn luyện
nhanh
Bàn 6×6 hoặc 8×8; tìm kiếm và heuristic theo vị trí
Bàn rút gọn, làm rõ luật bắt quân và phong cấp
Mini-Shogi hoặc luật rút gọn; quản lý quân trong tay và
nước thả quân

Độ khó
Trung bình

Khá–cao

Dễ–trung bình

Khá
Cao
Rất cao

Trang 3

8. TÀI LIỆU GỢI Ý VÀ LƯU Ý TRIỂN KHAI
•  Mark  Liu,  AlphaGo  Simplified:  Rule-Based  AI  and  Deep  Learning  in  Everyday  Games,  CRC
Press (bản sách đính kèm): Ch. 1–8 về rule-based/Minimax/MCTS; Ch. 9–15 về neural networks
và reinforcement learning; Ch. 16–21 về AlphaGo/AlphaZero.

•  Russell  &  Norvig,  Artificial  Intelligence:  A  Modern  Approach  (2021):  tác  tử  và  tìm  kiếm  đối

kháng; Sutton & Barto, Reinforcement Learning: An Introduction (2018): MDP và RL.

•  Đề tài ưu tiên chứng minh tác dụng của mỗi thành phần hơn là triển khai đủ 8 cấp bằng mọi giá.

Giảng viên có thể duyệt biến thể phù hợp thời gian và tài nguyên nhóm.

---o0o---

Trang 4

