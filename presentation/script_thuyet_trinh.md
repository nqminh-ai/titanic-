# Script thuyết trình — Titanic Survival Classification

Tổng thời lượng dự kiến: ~15 phút (17 slide). Gợi ý người nói: Điều phối (slide 1–8, 13–17), Thành viên 1–4 (slide 9–12, mỗi người ~1,5 phút).

---

## Slide 1 — Trang bìa (Điều phối, ~30s)
Xin chào thầy cô và các bạn. Nhóm chúng em xin trình bày dự án Mini Capstone với câu hỏi: ai sống sót trên Titanic? Chúng em xây dựng bốn mô hình phân loại — Logistic Regression, Extra Trees, SVM và Decision Tree — mỗi thành viên phụ trách một mô hình, và tất cả dùng chung một pipeline dữ liệu để so sánh công bằng.

## Slide 2 — Nội dung trình bày (~20s)
Bài trình bày gồm bảy phần: giới thiệu bài toán, kiểm tra dữ liệu và EDA, tiền xử lý và feature engineering, data contract dùng chung, bốn mô hình, so sánh kết quả, cuối cùng là kết luận cùng hướng phát triển.

## Slide 3 — Bài toán (~50s)
Dữ liệu gồm 891 hành khách có nhãn và 418 hành khách cần dự đoán trên Kaggle. Đây là bài toán phân loại nhị phân: Survived bằng 0 hoặc 1. Chỉ 38,4% hành khách sống sót — 549 người mất và 342 người sống — nên dữ liệu lệch lớp nhẹ. Mục tiêu của nhóm không chỉ là accuracy cao, mà là một quy trình tái lập được để bốn mô hình được so sánh trên cùng dữ liệu, cùng cách chia và cùng bộ metric.

## Slide 4 — Data audit (~50s)
Có ba cột bị thiếu dữ liệu. Cabin thiếu 77% nên chúng em loại bỏ. Age thiếu 20%, được điền bằng median theo nhóm giới tính và hạng vé, vì tuổi khác nhau rõ giữa các nhóm này. Embarked thiếu hai giá trị, điền bằng mode là S. Mọi thống kê dùng để điền chỉ học từ tập train, tránh rò rỉ dữ liệu. Không có bản ghi trùng, và PassengerId không được dùng làm feature.

## Slide 5 — EDA (~60s)
Hai yếu tố mạnh nhất là giới tính và hạng vé. Nữ có tỷ lệ sống 74%, nam chỉ 19%. Hạng 1 sống sót 63%, hạng 3 chỉ 24%. Cảng Cherbourg có tỷ lệ sống cao hơn, nhưng chủ yếu vì có nhiều khách hạng 1 lên tàu ở đó. Trẻ em và gia đình nhỏ có lợi thế, còn gia đình từ năm người trở lên có tỷ lệ sống rất thấp.

## Slide 6 — EDA đa biến (~40s)
Khi kết hợp giới tính và hạng vé, phụ nữ hạng 1 và 2 gần như đều sống sót, còn nam hạng 3 có cơ hội thấp nhất. Heatmap tương quan cho thấy Sex, Pclass và Fare liên quan mạnh với Survived. Chúng em lưu ý tương quan không đồng nghĩa với nhân quả.

## Slide 7 — Pipeline dữ liệu (~60s)
Pipeline chung gồm bảy bước, từ đọc dữ liệu đến xuất artifacts. Chúng em trích Title từ tên — Mr, Miss, Mrs, Master và Rare; tạo FamilySize và nhóm thành bốn loại gia đình; tạo AgeBand và FareBand theo ngưỡng phân vị học từ train. Kết quả là 28 feature sạch. Pipeline chung không scale dữ liệu; Logistic Regression và SVM tự scale bên trong Pipeline để scaler chỉ fit trên training fold.

## Slide 8 — Data contract (~50s)
Đây là luật chơi chung: random seed 42, chia 712 mẫu train và 179 mẫu validation có stratify, cross-validation 10-fold phân tầng, và accuracy là metric chính khi tuning. Mọi mô hình phải báo cáo thêm precision, recall, F1, ROC-AUC và confusion matrix. Mỗi thành viên theo cùng quy trình: baseline, cross-validation, GridSearch, đánh giá hold-out, learning curve, refit trên toàn bộ dữ liệu rồi tạo submission. Xin mời các thành viên trình bày phần mô hình của mình.

## Slide 9 — Logistic Regression (Thành viên 1, ~80s)
Em phụ trách Logistic Regression. Pipeline gồm StandardScaler và LogisticRegression, scaler đặt trong Pipeline nên chỉ fit trên training fold. Em thử 26 cấu hình với C, penalty L1/L2 và class_weight; tuning chỉ mất khoảng 2 giây. Cấu hình tốt nhất là penalty L1 với C bằng 0,1 — L1 đưa hơn mười hệ số về 0, tức mô hình tự chọn feature. CV tăng từ 0,822 lên 0,827 với độ lệch chuẩn thấp nhất nhóm, validation accuracy 0,838 và ROC-AUC 0,872 — cao nhất trong bốn mô hình. Nhìn vào hệ số: Sex có hệ số dương lớn nhất, Pclass âm — hạng vé càng thấp cơ hội càng giảm — còn gia đình lớn và danh xưng Mr đều làm giảm khả năng sống sót.

## Slide 10 — Extra Trees (Thành viên 2, ~80s)
Em phụ trách Extra Trees. Baseline overfit rõ: train accuracy 0,985 nhưng CV chỉ 0,784. Em tuning theo hướng coarse-to-fine, 144 rồi 27 cấu hình, khoảng 283 giây. Cấu hình tốt nhất là 300 cây, độ sâu tối đa 8, max_features log2, min_samples_leaf 4. CV tăng lên 0,837, validation 0,832, precision 0,820. Feature quan trọng nhất là Title_Mr, Sex và Pclass — khớp với EDA.

## Slide 11 — SVM (Thành viên 3, ~80s)
Em phụ trách SVM. Pipeline gồm StandardScaler và SVC. Grid dạng list-of-dicts để không tìm gamma cho kernel tuyến tính. Cấu hình tốt nhất là kernel RBF với C bằng 100 và gamma 0,001 — gamma nhỏ cho biên quyết định mượt. SVM đạt validation accuracy 0,844, precision 0,825 và F1 0,788, đều cao nhất nhóm. Mô hình dùng 287 support vector và tuning chỉ mất khoảng 41 giây.

## Slide 12 — Decision Tree (Thành viên 4, ~80s)
Em phụ trách Decision Tree. Em chạy grid lớn nhất: 2.880 cấu hình nhân 10 fold, tức 28.800 lần fit. Cây tốt nhất dùng gini với min_samples_leaf bằng 8, sâu 8 tầng và có 49 lá. CV đạt 0,838, cao nhất nhóm, và khoảng cách train–CV giảm từ 0,196 xuống 0,036, cho thấy overfitting được kiểm soát. Title_Mr chiếm gần một nửa độ quan trọng; câu hỏi đầu tiên của cây chính là hành khách có phải nam trưởng thành hay không.

## Slide 13 — So sánh (Điều phối, ~60s)
Đặt cạnh nhau trên cùng validation split: SVM dẫn đầu với accuracy 0,844 và F1 0,788, Logistic Regression theo sát với 0,838. Extra Trees và Decision Tree cùng đạt 0,832. Tất cả đều vượt xa baseline dummy luôn đoán "không sống sót", chỉ đạt 0,615. Decision Tree có CV cao nhất, Logistic Regression có ROC-AUC cao nhất.

## Slide 14 — Đọc kết quả (~60s)
Ba điểm cần nhấn mạnh. Thứ nhất, bốn mô hình chỉ cách nhau 0 đến 2 mẫu trên 179, nhỏ hơn độ lệch chuẩn cross-validation. Thứ hai, recall là điểm yếu chung: mọi mô hình bỏ sót 17 đến 19 người sống sót. Thứ ba, các feature quan trọng nhất nhất quán giữa các mô hình và khớp với nguyên tắc "phụ nữ và trẻ em trước".

## Slide 15 — Kết luận (~50s)
Nhóm chọn SVM kernel RBF làm mô hình cuối vì có accuracy, precision và F1 cao nhất cùng thời gian tuning ngắn. Logistic Regression đứng thứ hai với ROC-AUC cao nhất và kết quả ổn định nhất, phù hợp khi cần giải thích bằng hệ số; Decision Tree có CV tốt nhất và luật dễ đọc. Bài học lớn nhất: pipeline chung và data contract cho phép bốn người làm song song mà kết quả vẫn so sánh được, và feature engineering đóng góp nhiều hơn việc chọn thuật toán.

## Slide 16 — Hạn chế và hướng phát triển (~50s)
Hạn chế: tập validation chỉ 179 mẫu nên kết quả có độ bất định; tuning chỉ theo accuracy nên chưa tối ưu ngưỡng cho recall; feature importance dựa trên impurity có thể thiên lệch. Hướng phát triển: so sánh bằng repeated CV, thử ensemble voting hoặc stacking, dùng permutation importance hoặc SHAP, và khai thác thêm feature từ Ticket và Cabin.

## Slide 17 — Cảm ơn (~10s)
Phần trình bày của nhóm đến đây là kết thúc. Cảm ơn thầy cô và các bạn đã lắng nghe, nhóm xin nhận câu hỏi.

---
### Câu hỏi có thể gặp
- **Vì sao không chọn mô hình có CV cao nhất?** CV của Decision Tree (0,838) và SVM (0,830) chênh nhau ít hơn độ lệch chuẩn; SVM tốt hơn trên hold-out ở hầu hết metric. Cả hai đều là lựa chọn hợp lý.
- **Vì sao mô hình cây không cần scale?** Cây chia theo ngưỡng nên không phụ thuộc thang đo.
- **Có rò rỉ dữ liệu không?** Mọi thống kê điền thiếu và ngưỡng band học từ train; scaler nằm trong Pipeline nên chỉ fit trên training fold.
