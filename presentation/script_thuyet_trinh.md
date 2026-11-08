# Script thuyết trình — Titanic Survival Classification

**Thời lượng dự kiến:** khoảng 22–25 phút cho 32 slide.
**Phân công người nói:**
- Điều phối: slide 1–12 và 26–32
- Nguyễn Tấn Dũng: slide 13–15
- Nguyễn Trọng Hiệp: slide 16–18
- Nguyễn Quang Minh: slide 19–21
- Đỗ Trường Sơn: slide 22–25

**Điều khiển slide:** phím → hoặc Space để sang slide, ← để quay lại, F để toàn màn hình. Các hiệu ứng tự chạy khi vào slide; nên đợi khoảng 1–2 giây cho biểu đồ hiện hết rồi mới nói vào số liệu.

---

## Phần mở đầu

### Slide 1 — Trang bìa (Điều phối, ~30s)
Xin chào thầy cô và các bạn. Nhóm chúng em xin trình bày dự án Mini Capstone với câu hỏi: ai sống sót trên Titanic? Chúng em xây dựng bốn mô hình phân loại — Logistic Regression, Extra Trees, SVM và Decision Tree. Mỗi thành viên phụ trách một mô hình, và cả bốn dùng chung một pipeline dữ liệu để so sánh công bằng.

### Slide 2 — Lộ trình trình bày (~20s)
Bài trình bày có bốn phần. Đầu tiên là giới thiệu bài toán và cách nhóm tổ chức. Thứ hai là dữ liệu và EDA. Thứ ba — phần chính — là bốn mô hình; mỗi mô hình gồm lý thuyết, cách triển khai và kết quả. Cuối cùng là so sánh và kết luận.

## Phần 1 — Giới thiệu

### Slide 3 — Mở phần Giới thiệu (~5s)
Chúng ta bắt đầu với phần giới thiệu dự án.

### Slide 4 — Bài toán (~50s)
Dữ liệu gồm 891 hành khách có nhãn và 418 hành khách cần dự đoán trên Kaggle. Mỗi hành khách có 11 cột thông tin gốc như hạng vé, giới tính, tuổi, giá vé và họ tên. Chỉ 38,4% hành khách sống sót, nên dữ liệu lệch lớp nhẹ. Đây là bài toán phân loại nhị phân. Mục tiêu của nhóm không chỉ là đạt accuracy cao, mà là xây dựng một quy trình tái lập được, để bốn mô hình được so sánh trên cùng dữ liệu, cùng cách chia và cùng bộ metric.

### Slide 5 — Tổ chức nhóm (~45s)
Nhóm làm theo hai giai đoạn. Ở giai đoạn A, phần chung xử lý toàn bộ dữ liệu — audit, EDA, feature engineering, chia dữ liệu — rồi xuất ra thư mục artifacts/common. Ở giai đoạn B, bốn thành viên làm song song: mỗi người chỉ đọc từ artifacts/common và chỉ ghi vào thư mục của mình. Nhờ vậy không ai xung đột với ai, và khi tích hợp, kết quả so sánh được trực tiếp.

## Phần 2 — Dữ liệu và EDA

### Slide 6 — Mở phần Dữ liệu (~5s)
Tiếp theo là dữ liệu và phân tích khám phá.

### Slide 7 — Data audit (~50s)
Có ba cột bị thiếu dữ liệu. Cabin thiếu 77%, quá nhiều để điền tin cậy, nên chúng em loại bỏ. Age thiếu gần 20%, được điền bằng median theo nhóm giới tính và hạng vé, vì tuổi khác nhau rõ giữa các nhóm này. Embarked chỉ thiếu 2 dòng, điền bằng mode là S; Fare thiếu 1 dòng ở tập test, điền bằng median. Mọi thống kê dùng để điền chỉ học từ tập train, để tránh rò rỉ dữ liệu.

### Slide 8 — Giới tính, hạng vé và danh xưng (~60s)
Bảng bên trái cho thấy hai yếu tố mạnh nhất. Phụ nữ hạng 1 sống sót 97%, hạng 2 là 92%; trong khi nam hạng 2 và hạng 3 chỉ khoảng 15%. Bên phải là tỷ lệ sống theo danh xưng trích từ tên. Mrs và Miss sống trên 70%, Mr chỉ 16%. Điểm thú vị là Master — tức bé trai — sống sót 58%, gần gấp bốn lần Mr. Vì vậy Title mang thông tin cả về giới tính lẫn độ tuổi, điều mà riêng cột Sex không có.

### Slide 9 — Gia đình, cảng lên tàu và tương quan (~45s)
Quy mô gia đình có quan hệ hình chữ U ngược: đi một mình chỉ sống 30%, gia đình 3–4 người sống 61%, nhưng gia đình từ năm người trở lên chỉ còn 16%. Cảng Cherbourg có tỷ lệ sống cao hơn, nhưng chủ yếu vì có nhiều khách hạng 1 lên tàu ở đó. Heatmap bên phải tổng hợp tương quan sau khi mã hóa. Chúng em lưu ý tương quan không đồng nghĩa với nhân quả.

### Slide 10 — Feature engineering (~60s)
Từ những quan sát trên, chúng em tạo feature. Từ Name, trích Title và gom các danh xưng hiếm vào nhóm Rare. Từ SibSp và Parch, tạo FamilySize rồi chia thành bốn nhóm gia đình. Age và Fare được giữ giá trị gốc và thêm các band theo phân vị. Sex được mã hóa nhị phân, Embarked được one-hot. Ticket và Cabin bị loại. Kết quả là 28 feature, không còn giá trị thiếu, và train với test có cùng thứ tự cột. Phần chung không scale dữ liệu; Logistic Regression và SVM tự scale bên trong Pipeline của mình.

### Slide 11 — Data contract (~50s)
Đây là luật chơi chung cho cả bốn mô hình: random seed 42; chia 712 mẫu train và 179 mẫu validation có stratify; cross-validation 10-fold phân tầng; và accuracy là metric chính khi tuning. Mỗi thành viên phải đi qua đúng tám bước, từ đọc artifacts đến xuất submission. Tập test Kaggle không bao giờ được dùng để chọn mô hình.

## Phần 3 — Bốn mô hình

### Slide 12 — Bốn mô hình, bốn cách nhìn (~30s)
Bốn mô hình được chọn vì đại diện cho bốn cách tiếp cận khác nhau. Logistic Regression là mô hình tuyến tính cho ra xác suất. Extra Trees là ensemble gồm nhiều cây ngẫu nhiên. SVM tìm biên có lề rộng nhất trong không gian kernel. Decision Tree chia dữ liệu bằng chuỗi câu hỏi có hoặc không. Xin mời các thành viên trình bày.

### Slide 13 — Logistic Regression: lý thuyết (Nguyễn Tấn Dũng, ~70s)
Em xin trình bày Logistic Regression. Mô hình tính tổ hợp tuyến tính z bằng w nhân x cộng b, rồi đưa z qua hàm sigmoid để được xác suất từ 0 đến 1. Nếu xác suất lớn hơn 0,5 thì dự đoán là sống sót — chính là đường cong và ngưỡng trên hình. Mô hình học bằng cách tối thiểu log-loss. Em thêm điều chuẩn L1 với hệ số λ bằng 1 chia C; L1 có tính chất ép các hệ số không cần thiết về đúng 0. Ưu điểm lớn nhất là dễ diễn giải: mỗi hệ số là log-odds. Hạn chế là biên quyết định chỉ là tuyến tính, và bắt buộc phải scale dữ liệu.

### Slide 14 — Logistic Regression: triển khai (~60s)
Pipeline gồm StandardScaler rồi LogisticRegression. Scaler nằm bên trong Pipeline nên chỉ fit trên training fold của mỗi lần cross-validation, tránh rò rỉ thông tin. Em tìm trên lưới gồm C từ 0,001 đến 100, penalty L1 và L2, và class_weight — tổng cộng 26 cấu hình, 260 lần fit, chỉ mất khoảng 2 giây. Cấu hình tốt nhất là L1 với C bằng 0,1. Mô hình hội tụ sau 17 vòng lặp, và 14 trên 28 hệ số bằng 0, nghĩa là L1 đã tự loại một nửa số feature thừa.

### Slide 15 — Logistic Regression: kết quả (~60s)
So với baseline, mọi metric đều tăng. CV đạt 0,827 với độ lệch chuẩn chỉ 0,032 — ổn định nhất nhóm. Validation accuracy 0,838, và ROC-AUC 0,872, cao nhất trong bốn mô hình. Nhìn vào hệ số: giới tính nữ có hệ số dương lớn nhất, odds sống tăng 2,26 lần mỗi độ lệch chuẩn. Pclass, gia đình lớn và danh xưng Mr đều làm giảm khả năng sống. Learning curve cho thấy train và CV hội tụ quanh 0,83, khoảng cách cuối chỉ 0,005, nên mô hình không bị overfit.

### Slide 16 — Extra Trees: lý thuyết (Nguyễn Trọng Hiệp, ~70s)
Em xin trình bày Extra Trees, tức Extremely Randomized Trees. Đây là ensemble gồm nhiều cây quyết định; mỗi cây cho một xác suất, và kết quả cuối là trung bình của tất cả các cây — như trên hình, bốn cây cho 0,82, 0,71, 0,38 và 0,77, trung bình 0,67 nên dự đoán là sống. Khác với Random Forest, Extra Trees dùng toàn bộ dữ liệu cho mỗi cây thay vì bootstrap, và chọn ngưỡng chia ngẫu nhiên thay vì tìm ngưỡng tối ưu. Sự ngẫu nhiên này làm variance giảm hơn nữa và huấn luyện nhanh hơn. Hạn chế là khó diễn giải hơn một cây đơn, và cây sâu vẫn dễ overfit nếu không giới hạn.

### Slide 17 — Extra Trees: triển khai (~60s)
Lưới đầy đủ có 1.296 cấu hình, quá lớn để chạy với 10-fold. Vì vậy em tuning theo hướng coarse-to-fine: chạy 144 cấu hình thô trong 248 giây, rồi thu hẹp quanh vùng tốt nhất với 27 cấu hình trong 35 giây. Grid fine xác nhận lại cấu hình của grid coarse: 300 cây, độ sâu tối đa 8, max_features log2, mỗi lá ít nhất 4 mẫu, mỗi lần chia cần ít nhất 10 mẫu. Mô hình cây không cần scale. Ba tham số giới hạn độ phức tạp là chìa khóa chống overfit.

### Slide 18 — Extra Trees: kết quả (~60s)
Extra Trees là mô hình cải thiện nhiều nhất sau tuning: CV tăng từ 0,784 lên 0,837, precision tăng 0,077. Biểu đồ bên trái cho thấy lý do: baseline đạt 0,986 trên train nhưng chỉ 0,784 trên CV, tức overfit nặng. Sau khi giới hạn độ sâu, khoảng cách chỉ còn 0,02. Feature quan trọng nhất là Title_Mr, Sex và Pclass, khớp với EDA. Điểm yếu là recall 0,725, thấp nhất nhóm.

### Slide 19 — SVM: lý thuyết (Nguyễn Quang Minh, ~70s)
Em xin trình bày Support Vector Machine. SVM tìm siêu phẳng phân tách hai lớp sao cho lề — dải vàng trên hình — rộng nhất. Chỉ những điểm nằm trên mép lề, gọi là support vector, quyết định vị trí của biên. Bài toán tối ưu gồm hai phần: tối thiểu một phần hai bình phương chuẩn w để lề rộng, cộng C nhân tổng sai số để phạt các điểm vi phạm lề. C lớn thì biên cứng hơn. Để có biên phi tuyến, em dùng kernel RBF đo độ giống nhau giữa hai điểm; gamma nhỏ thì vùng ảnh hưởng rộng và biên mượt. Hạn chế: SVM bắt buộc scale và rất nhạy với C và gamma.

### Slide 20 — SVM: triển khai (~60s)
Pipeline gồm StandardScaler rồi SVC. Em dùng grid dạng list-of-dicts: kernel RBF được thử với C và gamma, còn kernel linear chỉ thử C, vì gamma không có ý nghĩa với kernel tuyến tính. Tổng cộng 66 cấu hình, mất 41 giây. Heatmap bên phải cho thấy đỉnh 0,830 nằm ở gamma bằng 0,001 và C bằng 100: kernel mượt nhưng phạt vi phạm nghiêm. Ngược lại, gamma lớn với C nhỏ làm accuracy tụt còn khoảng 0,63. Kernel linear tốt nhất đạt khoảng 0,826, nên RBF nhỉnh hơn.

### Slide 21 — SVM: kết quả (~60s)
SVM đạt validation accuracy 0,844, precision 0,825 và F1 0,788 — cả ba đều cao nhất nhóm. Số support vector giảm từ 346 xuống 287, nghĩa là biên đơn giản hơn và tổng quát tốt hơn. Learning curve cho thấy train khoảng 85–86%, CV khoảng 83%, khoảng cách chỉ 2,5%. Trên tập test, mô hình dự đoán 39% hành khách sống sót, rất sát tỷ lệ thực 38,4%.

### Slide 22 — Decision Tree: lý thuyết (Đỗ Trường Sơn, ~70s)
Em xin trình bày Decision Tree. Cây đưa ra một chuỗi câu hỏi có hoặc không, ví dụ: hành khách có danh xưng Mr không, hạng vé có phải hạng 3 không… cho đến khi đến một lá và đưa ra dự đoán. Sơ đồ trên hình chỉ là minh họa. Ở mỗi nút, cây chọn feature và ngưỡng làm giảm độ lẫn lộn Gini nhiều nhất. Gini bằng 0 khi nút chỉ còn một lớp. Ưu điểm là dễ đọc như một bộ luật, không cần scale. Nhược điểm là rất dễ overfit nếu để cây mọc tự do, và không ổn định khi dữ liệu thay đổi nhẹ.

### Slide 23 — Decision Tree: triển khai (~60s)
Vì một cây fit rất nhanh, em chạy grid lớn nhất nhóm: 2.880 cấu hình nhân 10 fold, tức 28.800 lần fit, mất khoảng 250 giây. Cấu hình tốt nhất không giới hạn độ sâu, nhưng mỗi lá phải có ít nhất 8 mẫu, nên cây tự dừng ở độ sâu 8 với 49 lá. Khi refit trên toàn bộ 891 mẫu, cây sâu 11 tầng và có 66 lá.

### Slide 24 — Decision Tree: kết quả (~60s)
Decision Tree có CV cao nhất nhóm, 0,838. Khoảng cách train–CV giảm từ 0,196 xuống 0,036, cho thấy overfit được kiểm soát. Riêng Title_Mr chiếm gần một nửa độ quan trọng, sau đó là Fare và Pclass. Sex có importance bằng 0 không phải vì giới tính không quan trọng, mà vì Title_Mr đã tách giới tính ngay từ nút gốc. Đây là hạn chế của importance dựa trên Gini. Hình bên phải là learning curve: train và CV tiến gần nhau khi tăng số mẫu.

### Slide 25 — Decision Tree: ba tầng đầu của cây (~40s)
Đây là ba tầng đầu của cây tối ưu, mỗi nút ghi điều kiện chia, Gini, số mẫu và lớp đa số. Màu xanh là nút nghiêng về "sống", màu cam là nghiêng về "không sống"; màu càng đậm thì nút càng thuần nhất. Nút gốc hỏi hành khách có danh xưng Mr hay không. Nhánh bên trái — phụ nữ và trẻ em, 300 mẫu — tiếp tục tách theo hạng vé: hạng 1–2 gần như đều sống (138 trên 145 ở một nút con), còn hạng 3 được tách theo gia đình lớn. Nhánh bên phải — nam trưởng thành, 412 mẫu — chủ yếu là không sống, và được tách tiếp theo giá vé và tuổi.

## Phần 4 — Kết quả và kết luận

### Slide 26 — Mở phần Kết quả (Điều phối, ~5s)
Bây giờ chúng ta đặt bốn mô hình cạnh nhau.

### Slide 27 — So sánh (~60s)
Trên cùng tập validation 179 hành khách, SVM dẫn đầu với accuracy 0,844, cùng precision và F1 cao nhất. Logistic Regression đứng thứ hai với 0,838 và ROC-AUC cao nhất. Decision Tree có CV cao nhất, còn Extra Trees giảm overfit mạnh nhất. Tất cả đều vượt xa mô hình dummy luôn đoán "không sống sót", chỉ đạt 0,615. Biểu đồ cột cắt trục từ 0,80 để thấy rõ khác biệt — thực tế khác biệt rất nhỏ.

### Slide 28 — Confusion matrix (~50s)
Bốn confusion matrix cho thấy các mô hình sai ở cùng một chỗ. Số người sống bị bỏ sót luôn từ 17 đến 19, nhiều hơn số báo động nhầm, từ 11 đến 13. Lớp sống sót là lớp thiểu số và có nhiều ngoại lệ, ví dụ nam hạng 1 vẫn sống sót, nên khó đoán bằng feature hiện có. SVM chỉ hơn các mô hình khác một đến hai hành khách.

### Slide 29 — Ba điều rút ra (~60s)
Chúng em rút ra ba điều. Thứ nhất, toàn bộ khác biệt chỉ khoảng hai mẫu, trong khi độ lệch chuẩn của cross-validation còn lớn hơn khoảng cách đó. Thứ hai, không mô hình nào vượt recall 0,754 — giới hạn nằm ở thông tin trong feature chứ không ở thuật toán. Thứ ba, mọi mô hình đều đồng thuận về các tín hiệu quan trọng nhất: giới tính, hạng vé và quy mô gia đình — khớp với nguyên tắc "phụ nữ và trẻ em trước".

### Slide 30 — Kết luận (~50s)
Nhóm chọn SVM kernel RBF với C bằng 100 và gamma 0,001 làm mô hình cuối, vì có accuracy, precision và F1 cao nhất cùng thời gian tuning ngắn. Logistic Regression là lựa chọn tốt khi cần giải thích bằng hệ số; Decision Tree khi cần bộ luật dễ đọc. Bài học lớn nhất: pipeline chung và data contract giúp bốn người làm song song mà kết quả vẫn so sánh công bằng, và feature engineering đóng góp nhiều hơn việc chọn thuật toán.

### Slide 31 — Hạn chế và hướng phát triển (~50s)
Về hạn chế: tập validation chỉ 179 mẫu, mỗi mẫu tương đương 0,56% accuracy, nên kết quả có độ bất định cao. Tuning chỉ theo accuracy nên chưa tối ưu cho recall. Importance dựa trên Gini bị thiên lệch, và chưa kiểm chứng trên leaderboard Kaggle. Hướng phát triển gồm repeated cross-validation kèm kiểm định thống kê, ensemble bốn mô hình, dùng SHAP để giải thích, tạo thêm feature từ Ticket và Cabin, và hạ ngưỡng quyết định để giảm bỏ sót người sống.

### Slide 32 — Cảm ơn (~10s)
Phần trình bày của nhóm đến đây là kết thúc. Cảm ơn thầy cô và các bạn đã lắng nghe; nhóm xin nhận câu hỏi.

---

## Câu hỏi có thể gặp

- **Vì sao không chọn Decision Tree, mô hình có CV cao nhất?**
  CV của Decision Tree (0,838) và SVM (0,830) chênh nhau ít hơn độ lệch chuẩn. SVM tốt hơn trên hold-out ở hầu hết metric và ổn định hơn: độ lệch chuẩn CV là 0,032 so với 0,040. Cả hai đều là lựa chọn hợp lý.

- **Vì sao Logistic Regression dùng L1 lại tốt hơn L2?**
  Một số feature trùng lặp thông tin, ví dụ FamilySize với SibSp và Parch, hay các band với giá trị gốc. L1 loại bớt chúng nên mô hình gọn và ổn định hơn. Thực tế L2 với C = 0,01 cũng chỉ kém 0,0014.

- **Vì sao mô hình cây không cần scale?**
  Cây chia dữ liệu theo ngưỡng của từng feature riêng lẻ, nên thứ tự giá trị mới quan trọng, không phải thang đo.

- **Có rò rỉ dữ liệu không?**
  Không. Mọi thống kê điền thiếu và ngưỡng band đều học từ train. Scaler nằm trong Pipeline nên chỉ fit trên training fold. Tập test Kaggle không được dùng để chọn mô hình.

- **Làm sao tăng recall?**
  Có hai cách: hạ ngưỡng quyết định xuống dưới 0,5, hoặc dùng `class_weight="balanced"`. Cả hai đổi một ít precision lấy recall. Grid hiện tại có thử `balanced`, nhưng vì tuning theo accuracy nên cấu hình này không được chọn.
