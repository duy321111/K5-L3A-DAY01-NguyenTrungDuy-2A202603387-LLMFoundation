# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603387
- **Họ và tên:** Nguyễn Trung Duy
- **SEED:** 7311

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook có output, báo cáo cuối ghi 15/15 kiểm tra đạt và đủ 10 dự đoán | Có |
| [`lab1_report.json`](lab1_report.json) — kết quả xuất từ notebook, SEED 7311 | Có |
| Ảnh tính tay: [`hand_calc1.jpg`](hand_calc1.jpg), [`hand_calc2.jpg`](hand_calc2.jpg), [`hand_calc3.jpg`](hand_calc3.jpg) | Có 3 tệp; chưa có tệp tên `hand_calc.jpg` theo mẫu nộp |

## 2. Kết quả chính

Số liệu dưới đây lấy từ `lab1_report.json`; F1 làm tròn 3 chữ số thập phân.

| Mô hình | F1 test | F1 SUB 40 | p95 (ms/tin) | USD/1.000 tin |
| --- | ---: | ---: | ---: | ---: |
| BoW + LogReg (tự code) | 0.846 | 0.974 | 0.128 | 0 |
| BoW + MLP | 0.910 | 1.000 | 0.896 | 0 |
| MiniLM + kNN | 0.872 | 1.000 | 55.983 | 0 |
| MiniLM + LogReg | 0.929 | 1.000 | 48.449 | 0 |
| MiniLM + MLP head | 0.939 | 1.000 | 37.562 | 0 |
| LLM nhỏ zero-shot | — | 0.000 | 2165.496 | 0.004903 |
| LLM lớn zero-shot | — | 0.000 | 1420.367 | 0.049033 |
| LLM nhỏ few-shot | — | 0.000 | 1625.499 | 0.017170 |

LLM chỉ được đánh giá trên SUB 40, không phải toàn bộ test. Trường `f1` của các hàng LLM trong JSON cũng là F1 trên SUB. Cả ba lần chạy đều ghi `invalid=40`; code đổi nhãn không hợp lệ thành ham, dẫn đến accuracy 0.5 và F1 0 trên SUB cân bằng. Vì vậy chưa thể kết luận LLM phân loại kém hơn từ các kết quả này. Chi phí là ước tính token theo số từ và bảng giá trong notebook, không phải hóa đơn thực tế; giá trị 0 của mô hình cục bộ chưa tính phần cứng và vận hành.

## 3. Dự đoán và thực tế

1. **`llm_models`:** Em dự đoán model lớn hơn model nhỏ 0.02–0.05 F1 và tốn gấp 3–5 lần. Thực tế cả hai có F1 SUB bằng 0, chi phí ước tính model lớn gấp 10 lần. Do cả hai đều có 40/40 nhãn không hợp lệ rồi bị đổi thành ham, phép so sánh chất lượng chưa có ý nghĩa. Cần kiểm tra response gốc, giới hạn token và xử lý nhãn trước khi chạy lại; output hiện có chưa xác định được nguyên nhân gốc.
2. **`llm_fewshot`:** Em dự đoán thêm 4 ví dụ giúp tăng F1 0.02–0.05 và tăng chi phí 1.5–2 lần. Thực tế F1 vẫn bằng 0, `invalid=40`, còn chi phí tăng khoảng 3.50 lần (0.017170/0.004903 USD mỗi 1.000 tin). Prompt dài hơn làm tăng chi phí, nhưng không giải quyết được nhãn không hợp lệ.
3. **`emb_tricky`:** Em dự đoán Encoder + LogReg đúng nhiều hơn BoW trên 6 tin dễ nhầm. Thực tế hai mô hình cùng đúng 4/6. Encoder sửa được tin “Exclusive: get a brand new phone at no cost...” mà BoW bỏ sót, nhưng lại bỏ sót “Your mobile number has been chosen for a complimentary holiday...” mà BoW đoán đúng. Embedding giúp biểu diễn ngữ nghĩa, nhưng bộ phân loại vẫn phụ thuộc dữ liệu huấn luyện và có thể sai trên cách diễn đạt mới.

## 4. Phân tích 50 case khó

| Mô hình | F1 SUB thường | F1 50 case khó | Mức giảm F1 | Accuracy tập khó |
| --- | ---: | ---: | ---: | ---: |
| BoW + LogReg | 0.974 | 0.333 | 0.641 | 0.52 |
| BoW + MLP | 1.000 | 0.368 | 0.632 | 0.52 |
| MiniLM + kNN | 1.000 | 0.759 | 0.241 | 0.72 |
| MiniLM + LogReg | 1.000 | 0.717 | 0.283 | 0.70 |
| MiniLM + MLP head | 1.000 | 0.596 | 0.404 | 0.62 |

**BoW + LogReg tụt nhiều nhất**, giảm khoảng 0.641 F1 so với SUB thường. BoW + MLP giảm khoảng 0.632. SUB chỉ gồm 40 tin nên điểm gần 1 không bảo đảm mô hình xử lý tốt cách diễn đạt khác phân phối dữ liệu huấn luyện.

Nhóm **spam giọng tự nhiên** bộc lộ điểm yếu chung: output liệt kê 5 case mà tất cả mô hình đều sai, đều thuộc nhóm này. Hai ví dụ:

- **Case #13:** “Hope you're well! Quick reminder that your photos from the event are ready, view & order prints...” là spam giả lời nhắc thân thiện để dẫn tới bán hàng không được yêu cầu. BoW thiếu tín hiệu từ khóa spam quen thuộc; embedding có thể đặt câu gần các lời nhắc hợp lệ nên bộ phân loại bỏ sót.
- **Case #14:** “Hi, I'm working from home and making about £300 a day, happy to show you how, no experience...” là lời dụ việc làm dưới dạng chia sẻ cá nhân. Cách kể chuyện khiến đặc trưng từ và vector nghĩa giống hội thoại bình thường, trong khi ý đồ lừa đảo cần thêm ngữ cảnh để nhận ra.

BoW cũng dễ gặp khó với né từ khóa như chèn khoảng trắng hoặc thay chữ bằng số. Có thể thử chuẩn hóa ký tự và bổ sung đặc trưng ký tự, nhưng phải kiểm tra nguy cơ chặn nhầm tin hợp lệ.

LLM zero-shot và prompt phòng thủ cùng đạt F1 0, accuracy 0.42 trên tập khó. Với lỗi nhãn không hợp lệ đã thấy ở SUB, chưa đủ bằng chứng để quy lỗi cho prompt injection hoặc kết luận prompt phòng thủ hiệu quả hay không. Cần ghi riêng response và tỷ lệ invalid trong stress test.

Các số liệu trên thuộc lần chạy 50 case đã lưu. Ô bonus thêm 5 case nằm sau ô đánh giá; output hiện có chưa chứng minh đã chạy lại trên 55 case. Bộ case viết tay dùng để phát hiện điểm yếu, không dùng thay cho tập đại diện lưu lượng thực tế.

## 5. Thử thách cá nhân

SEED 7311 cho `challenge = 3` trong JSON (chỉ số bắt đầu từ 0), tương ứng **đề #4: chạy kNN với k = 1, 5, 25**. Theo code và output đã có trong notebook, mỗi lần dùng embedding của tập test, lấy nhãn qua `knn_predict`, rồi tính F1:

| k | F1 test |
| ---: | ---: |
| 1 | 0.872 |
| 5 | 0.872 |
| 25 | 0.851 |

k = 1 chỉ dựa vào một hàng xóm nên nhạy với nhiễu. k = 25 lấy vùng lân cận rộng hơn, có thể đưa nhiều tin ham vào biểu quyết do dữ liệu mất cân bằng, làm bỏ sót spam. Trong lần chạy này k = 1 và k = 5 có F1 bằng nhau ở độ làm tròn đã in; k = 25 thấp hơn khoảng 0.021. Có thể giữ k = 5 làm cấu hình hiện tại, nhưng muốn chọn k đáng tin cậy cần dùng tập validation thay vì tối ưu theo test.

## 6. Phương án cho sản phẩm thật

Em chọn **BoW + LogReg làm tầng xử lý nhanh, kết hợp MiniLM + LogReg để kiểm tra tin chưa chắc chắn**, rồi đo lại toàn bộ hệ thống trước khi triển khai.

BoW + LogReg có p95 0.128 ms/tin và throughput khoảng 44.794 tin/giây trong phép đo cả lô, dư khả năng so với mức trung bình 10 triệu/86.400 ≈ 116 tin/giây. Đổi lại, recall test chỉ 0.733 và F1 tập khó 0.333 nên dùng riêng sẽ bỏ sót nhiều spam.

MiniLM + LogReg có F1 test 0.929, F1 tập khó 0.717 và p95 48.449 ms/tin, đạt ngưỡng trong phép đo hiện tại nhưng ít dư địa cho hàng đợi và mạng. MiniLM + MLP head đạt F1 test cao nhất 0.939, p95 37.562 ms/tin, nhưng F1 tập khó chỉ 0.596. MiniLM + kNN có F1 tập khó tốt nhất 0.759 nhưng p95 55.983 ms/tin vượt yêu cầu.

Router có thể chuyển các tin có xác suất BoW gần ngưỡng, chẳng hạn trong [0.3, 0.7], sang MiniLM + LogReg. Đây là thiết kế đề xuất, chưa được đo: cần kiểm tra cả trường hợp BoW tự tin nhưng sai, đo F1, tỷ lệ chuyển tầng và p95 đầu cuối. Throughput MiniLM + LogReg hiện khoảng 37.88 tin/giây; nếu xử lý toàn bộ lưu lượng thì cần ít nhất khoảng 4 worker tương đương theo phép tính lý tưởng, còn thực tế phải dự phòng giờ cao điểm và kiểm thử tải.

Em chấp nhận tăng chi phí tính toán và độ phức tạp vận hành để giảm bỏ sót, đồng thời chọn ngưỡng trên validation để hạn chế chặn nhầm OTP hoặc tin giao dịch. LLM hiện chưa phù hợp cho luồng đồng bộ vì p95 trên 1.4 giây và đầu ra chưa hợp lệ. Nếu ngoại suy chi phí ước tính trong notebook cho 30 ngày, 10 triệu tin/ngày tương ứng khoảng 1.471 USD/tháng với LLM nhỏ zero-shot, 14.710 USD với LLM lớn và 5.151 USD với few-shot; các mức này chưa tính retry, hạ tầng và thay đổi độ dài tin.

## 7. AI log và điều đã học

Các lượt hỗ trợ dưới đây được tổng hợp từ AI log đã ghi trong notebook; không có lịch sử chi tiết từng lượt ngoài nội dung đó.

| Phần | Nội dung hỏi AI | Hỗ trợ nhận được | Kiểm chứng đã ghi trong notebook |
| --- | --- | --- | --- |
| ML | Cách viết và giải thích NumPy, sigmoid, predict_proba | Giải thích và hướng dẫn viết hàm | Chạy code và kiểm tra kết quả |
| DL | Cách nhận biết overfitting | Giải thích train loss và validation loss | Xem biểu đồ và so sánh kết quả |
| Embedding | Cách hoạt động của embedding và kNN | Giải thích phân loại và tính F1 | Chạy thử và xem kết quả |
| LLM API | Viết code gọi API, xử lý response và phân loại | Hỗ trợ hàm gọi LLM, parse_label và xử lý lỗi | Chạy code, sửa lỗi, kiểm tra output |
| Zero-shot | Dự đoán F1 và chi phí model lớn/nhỏ | Gợi ý dự đoán và giải thích | Chạy thử, ghi nhận F1 = 0 |
| Few-shot | Viết dự đoán khi thêm 4 ví dụ | Gợi ý dự đoán F1 và chi phí | Chạy thử, ghi nhận invalid = 40 |
| Pareto | Nhận xét F1 và độ trễ | Gợi ý dự đoán và giải thích Pareto | So sánh với biểu đồ |
| Stress test | Phân tích 50 case khó | Giải thích điểm yếu của các mô hình | Xem accuracy và F1 |
| Báo cáo Markdown (lần này) | Đọc notebook và viết báo cáo cá nhân | Tổng hợp output và JSON; sửa nhận xét về mức giảm F1, độ trễ và giới hạn kết quả LLM | AI đối chiếu số liệu trong hai tệp; sinh viên cần tự đọc lại và xác nhận diễn giải trước khi nộp |

Em hiểu rằng F1 cao trên tập test thường chưa bảo đảm mô hình xử lý tốt spam có cách diễn đạt mới. Khi LLM có F1 bằng 0, cần kiểm tra đầu ra và cách xử lý nhãn trước khi kết luận về năng lực mô hình. Khi chọn phương án thực tế, cần xét đồng thời chất lượng, độ trễ và throughput, vì đạt p95 trên một lần đo chưa bảo đảm phục vụ đủ lưu lượng.
