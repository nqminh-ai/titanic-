# Kế hoạch Mini Capstone — Titanic Classification

## 1. Mục tiêu

Xây dựng một quy trình machine learning hoàn chỉnh cho bài toán dự đoán `Survived` trên Titanic, tham khảo lộ trình của notebook `detailed-and-typical-solution-ensemble-modeling.ipynb`:

1. Hiểu bài toán và dữ liệu.
2. Làm sạch và tiền xử lý dữ liệu.
3. EDA (Exploratory Data Analysis).
4. FE (Feature Engineering).
5. Xây dựng bốn mô hình phân loại độc lập:
   - Logistic Regression.
   - Extra Trees.
   - Support Vector Machine (SVM/SVC).
   - Decision Tree.
6. Đánh giá bằng cùng một cách chia dữ liệu và cùng bộ metric.
7. Tối ưu siêu tham số bằng cross-validation.
8. Tổng hợp kết quả trong notebook chung và chọn mô hình cuối cùng.

Nguyên tắc quan trọng: notebook chung là nguồn chuẩn cho dữ liệu và đặc trưng; notebook riêng không tự ý định nghĩa lại tập đặc trưng hoặc cách chia train/validation.

## 2. Cấu trúc thư mục đích

```text
Mini_Capstone/
├── input/titanic/
│   ├── train.csv
│   ├── test.csv
│   └── gender_submission.csv
├── notebooks/
│   └── 00_common_pipeline.ipynb
├── members/
│   ├── logistic_regression/
│   │   └── 01_logistic_regression.ipynb
│   ├── extra_trees/
│   │   └── 02_extra_trees.ipynb
│   ├── svm/
│   │   └── 03_svm.ipynb
│   └── decision_tree/
│       └── 04_decision_tree.ipynb
├── artifacts/
│   ├── common/
│   ├── logistic_regression/
│   ├── extra_trees/
│   ├── svm/
│   └── decision_tree/
├── reports/
│   ├── figures/common/
│   ├── figures/logistic_regression/
│   ├── figures/extra_trees/
│   ├── figures/svm/
│   └── figures/decision_tree/
└── prompts/
```

`artifacts/common/` chứa dữ liệu và metadata dùng chung. Mỗi thư mục model chỉ chứa kết quả của đúng thành viên đó. Không commit file tạm, cache hoặc output notebook quá lớn.

## 3. Phân vai

| Vai trò | Phạm vi chính | File được phép sửa |
|---|---|---|
| Điều phối/nhiệm vụ chung | Data audit, preprocessing, EDA, FE, split, data contract, tích hợp cuối | `notebooks/00_common_pipeline.ipynb`, `artifacts/common/`, phần tổng hợp trong `PROJECT_PLAN.md` |
| Thành viên 1 | Logistic Regression | `members/logistic_regression/`, `artifacts/logistic_regression/`, `reports/figures/logistic_regression/` |
| Thành viên 2 | Extra Trees | `members/extra_trees/`, `artifacts/extra_trees/`, `reports/figures/extra_trees/` |
| Thành viên 3 | SVM | `members/svm/`, `artifacts/svm/`, `reports/figures/svm/` |
| Thành viên 4 | Decision Tree | `members/decision_tree/`, `artifacts/decision_tree/`, `reports/figures/decision_tree/` |

## 4. Giai đoạn A — Nhiệm vụ chung

### A1. Khởi tạo và khả năng tái lập

- Tạo `notebooks/00_common_pipeline.ipynb`.
- Khai báo `RANDOM_STATE = 42` và sử dụng thống nhất.
- Đọc dữ liệu bằng đường dẫn tương đối từ `input/titanic/`.
- Ghi phiên bản Python và các thư viện chính.
- Không phụ thuộc đường dẫn Kaggle hoặc đường dẫn tuyệt đối của một máy cá nhân.

### A2. Data audit

- Hiển thị shape, tên cột, kiểu dữ liệu, vài dòng đầu/cuối.
- Kiểm tra bản ghi trùng lặp, tính duy nhất của `PassengerId`.
- Lập bảng số lượng và tỷ lệ thiếu cho train/test.
- Kiểm tra target `Survived`: số lượng, tỷ lệ lớp và mức mất cân bằng.
- Kiểm tra các giá trị bất thường của `Age`, `Fare`, `SibSp`, `Parch`, `Pclass`.
- Ghi rõ ý nghĩa của từng biến và quyết định giữ, bỏ hoặc biến đổi.

### A3. EDA đơn biến

- Target: count plot và tỷ lệ sống sót.
- Numerical: phân phối `Age`, `Fare`, `SibSp`, `Parch` bằng histogram/boxplot.
- Categorical: phân phối `Sex`, `Pclass`, `Embarked`.
- Mỗi biểu đồ phải kèm 1–3 nhận xét, không chỉ hiển thị hình.

### A4. EDA hai biến và đa biến

- Tỷ lệ sống sót theo `Sex`, `Pclass`, `Embarked`, `SibSp`, `Parch`.
- Phân phối `Age` và `Fare` theo `Survived`.
- Phân tích `Sex × Pclass × Survived` và `Age × Pclass × Survived`.
- Heatmap tương quan sau khi mã hóa các biến phù hợp.
- Nêu rõ tương quan không đồng nghĩa với quan hệ nhân quả.
- Lưu hình dùng cho báo cáo vào `reports/figures/common/`.

### A5. Tiền xử lý và feature engineering

Thực hiện nhất quán cho train và test:

- Giữ `PassengerId` riêng để tạo submission, không dùng làm feature.
- `Embarked`: điền mode lấy từ train; mã hóa nhất quán.
- `Fare`: điền median lấy từ train.
- `Age`: điền median theo nhóm `Sex × Pclass`; nếu nhóm không có dữ liệu thì fallback về median toàn train.
- `Sex`: mã hóa nhị phân.
- `Title`: trích danh xưng từ `Name`, chuẩn hóa `Mlle/Ms/Mme`, gom danh xưng hiếm thành `Rare`, rồi mã hóa.
- `FamilySize = SibSp + Parch + 1`.
- Tạo `FamilyCategory`: `Single`, `Small` (2), `Medium` (3–4), `Large` (>=5), sau đó one-hot hoặc mã hóa có giải thích.
- Tạo `AgeBand` và `FareBand` với ngưỡng học từ train. Không để lỗi nhóm `Age > 64` như notebook tham khảo.
- Bỏ `Name`, `Ticket`, `Cabin` sau khi đã khai thác thông tin cần thiết; giải thích quyết định.
- Kiểm tra train/test có cùng tên và thứ tự feature.
- Kiểm tra không còn NaN, infinity hoặc cột object chưa mã hóa.

Không scale feature trong phần chung. Logistic Regression và SVM sẽ scale bên trong `Pipeline` riêng; mô hình cây không cần scale.

### A6. Chia dữ liệu và data contract

- Tách `X` và `y`.
- Dùng một hold-out split chung, đề xuất `test_size=0.20`, `stratify=y`, `random_state=42`.
- Mọi model phải dùng đúng split này để so sánh công bằng.
- Cross-validation dùng `StratifiedKFold(n_splits=10, shuffle=True, random_state=42)`.
- Metric chính: `accuracy` để bám notebook tham khảo.
- Metric bổ sung bắt buộc: precision, recall, F1, ROC-AUC và confusion matrix.
- Khi tuning, dùng `scoring='accuracy'`; đồng thời báo cáo các metric bổ sung trên hold-out.

Xuất vào `artifacts/common/`:

- `X_train.csv`, `X_valid.csv`, `y_train.csv`, `y_valid.csv`.
- `X_full.csv`, `y_full.csv`, `X_test.csv`.
- `test_passenger_id.csv`.
- `feature_manifest.json`: danh sách feature, kiểu dữ liệu, target, random seed, cách split, quy tắc FE.
- `baseline_metrics.json`: baseline đơn giản, ví dụ dự đoán lớp phổ biến và Logistic Regression mặc định chỉ để kiểm tra pipeline.

### A7. Tạo notebook riêng sau khi phần chung hoàn tất

- Chạy notebook chung từ đầu đến cuối và xác nhận artifacts hợp lệ.
- Tạo bốn notebook riêng trong bốn thư mục thành viên.
- Mỗi notebook riêng phải có phần tóm tắt hoặc bản sao các bước chung đã hoàn thành, nhưng phần thực thi model phải đọc trực tiếp từ `artifacts/common/` để đảm bảo dữ liệu giống nhau.
- Không để thành viên sửa notebook chung trong giai đoạn phát triển model.

### A8. Tiêu chí hoàn thành phần chung

- Notebook chạy lại từ đầu đến cuối không lỗi.
- Có kết luận EDA rõ ràng và các biểu đồ chính.
- Không có data leakage rõ ràng giữa validation và quá trình fit/tuning.
- Dữ liệu train/valid/test có cùng schema.
- Các artifact chung và manifest được tạo đầy đủ.
- Bốn notebook model mở được và đọc đúng artifact chung.

## 5. Giai đoạn B — Quy trình bắt buộc cho mọi model

Mỗi thành viên thực hiện cùng khung sau trong notebook riêng:

1. Đọc data contract và artifacts chung; kiểm tra schema/shape/NaN.
2. Xây dựng baseline model với tham số mặc định hoặc tối thiểu hợp lý.
3. Đánh giá baseline bằng 10-fold stratified cross-validation trên `X_train, y_train`.
4. Fit baseline trên `X_train`, đánh giá trên `X_valid`.
5. Báo cáo accuracy, precision, recall, F1, ROC-AUC, classification report và confusion matrix.
6. Vẽ ROC curve; vẽ learning curve nếu thời gian cho phép, khuyến nghị bắt buộc để bám notebook tham khảo.
7. Xây dựng lưới tham số có kích thước hợp lý.
8. Dùng `GridSearchCV` với cùng StratifiedKFold và `scoring='accuracy'`.
9. Báo cáo `best_params_`, `best_score_`, thời gian tuning và bảng các cấu hình tốt nhất.
10. Đánh giá best estimator trên hold-out validation bằng toàn bộ metric.
11. So sánh baseline với tuned model và giải thích overfit/underfit.
12. Refit best estimator trên `X_full, y_full`.
13. Dự đoán nhãn và xác suất/decision score cho `X_test`.
14. Lưu model, metrics, CV results, validation predictions, test predictions và submission vào thư mục artifact riêng.
15. Ghi kết luận, hạn chế và đề xuất cải tiến.

Quy ước file đầu ra của mỗi model:

```text
artifacts/<model>/
├── best_model.joblib
├── metrics.json
├── cv_results.csv
├── validation_predictions.csv
├── test_predictions.csv
└── submission.csv
```

`metrics.json` tối thiểu có các khóa: `model_name`, `random_state`, `cv_folds`, `baseline_cv_mean`, `baseline_cv_std`, `best_cv_score`, `best_params`, `validation_accuracy`, `validation_precision`, `validation_recall`, `validation_f1`, `validation_roc_auc`.

`validation_predictions.csv` gồm `row_id`, `y_true`, `y_pred`, `y_score`. `test_predictions.csv` gồm `PassengerId`, `Survived`, `score`. Nếu estimator không có `predict_proba`, dùng `decision_function` làm `score` và ghi rõ ý nghĩa.

## 6. Giai đoạn C — Nhiệm vụ riêng

### Thành viên 1 — Logistic Regression

- Dùng `Pipeline([StandardScaler(), LogisticRegression(...)])` để scaler chỉ được fit trong từng fold.
- Baseline: `max_iter` đủ lớn và `random_state=42` nếu phiên bản hỗ trợ.
- Grid đề xuất:
  - `C`: `[0.001, 0.01, 0.1, 1, 10, 100]`.
  - `penalty`: `l1`, `l2` với solver tương thích, ưu tiên `liblinear`; có thể thêm `none`/`None` nếu phiên bản hỗ trợ.
  - `class_weight`: `[None, 'balanced']`.
- Phân tích hệ số của best model theo feature; chú ý hệ số chỉ so sánh hợp lý sau khi scale.
- Kiểm tra convergence warning và tăng `max_iter` nếu cần.

### Thành viên 2 — Extra Trees

- Baseline: `ExtraTreesClassifier(random_state=42, n_jobs=-1)`.
- Grid đề xuất:
  - `n_estimators`: `[100, 300, 500]`.
  - `max_depth`: `[None, 4, 8, 12]`.
  - `max_features`: `['sqrt', 'log2', None]`.
  - `min_samples_split`: `[2, 5, 10]`.
  - `min_samples_leaf`: `[1, 2, 4]`.
  - `criterion`: `['gini', 'entropy']`.
  - `class_weight`: `[None, 'balanced']`.
- Nếu full grid quá lớn, thu nhỏ có lý do hoặc chạy hai giai đoạn coarse-to-fine.
- Báo cáo feature importance và lưu biểu đồ top features.

### Thành viên 3 — SVM

- Dùng `Pipeline([StandardScaler(), SVC(probability=True, random_state=42)])`.
- Baseline với kernel RBF.
- Grid đề xuất:
  - `C`: `[0.1, 1, 10, 50, 100]`.
  - `kernel`: `['rbf', 'linear']`.
  - `gamma`: `['scale', 'auto', 0.001, 0.01, 0.1, 1]` cho RBF; tách grid dạng list-of-dicts để không tìm `gamma` vô nghĩa cho linear.
  - `class_weight`: `[None, 'balanced']`.
- Báo cáo số support vectors.
- Ghi nhận chi phí thời gian của `probability=True`; nếu tuning chậm, có thể tune với `probability=False`, sau đó dựng lại best model với `probability=True` và cùng tham số để refit.

### Thành viên 4 — Decision Tree

- Baseline: `DecisionTreeClassifier(random_state=42)`.
- Grid đề xuất:
  - `criterion`: `['gini', 'entropy', 'log_loss']` nếu phiên bản hỗ trợ.
  - `max_depth`: `[None, 2, 3, 4, 5, 6, 8, 10]`.
  - `min_samples_split`: `[2, 5, 10, 20]`.
  - `min_samples_leaf`: `[1, 2, 4, 8, 12]`.
  - `max_features`: `[None, 'sqrt', 'log2']`.
  - `class_weight`: `[None, 'balanced']`.
  - Có thể khảo sát `ccp_alpha` để pruning.
- Vẽ cây đã tối ưu hoặc ba tầng đầu nếu cây lớn.
- Báo cáo feature importance, độ sâu và số lá; phân tích chênh lệch train/validation để chỉ ra overfitting.

## 7. Giai đoạn D — Tích hợp vào notebook chung

Chỉ bắt đầu khi cả bốn thư mục artifact đã đầy đủ.

- Notebook chung đọc bốn `metrics.json` và tạo bảng xếp hạng thống nhất.
- So sánh CV mean/std và các metric hold-out; không chọn model chỉ dựa trên một con số.
- Đọc `validation_predictions.csv` để:
  - Kiểm tra tất cả model dùng đúng các dòng validation.
  - Vẽ confusion matrix và ROC curve cạnh nhau.
  - Tính tương quan/độ đồng thuận giữa dự đoán của bốn model.
- Đọc các `submission.csv` riêng để kiểm tra schema.
- Chọn best single model theo tiêu chí đã công bố; tùy chọn tạo hard/soft voting nếu dữ liệu dự đoán cho phép.
- Lưu bảng tổng hợp cuối vào `artifacts/common/model_comparison.csv`.
- Viết kết luận về trade-off giữa hiệu năng, khả năng giải thích, thời gian huấn luyện và độ ổn định.

## 8. Chiến lược Git để giảm conflict

1. Điều phối hoàn thành và commit phần chung trước.
2. Tạo một tag hoặc ghi commit hash của baseline chung vào `feature_manifest.json` nếu repo Git đã được khởi tạo.
3. Mỗi thành viên tạo branch từ cùng commit baseline:
   - `model/logistic-regression`
   - `model/extra-trees`
   - `model/svm`
   - `model/decision-tree`
4. Thành viên chỉ sửa thư mục được phân công; không sửa notebook chung, prompt hoặc artifact của người khác.
5. Mỗi người tự commit notebook, artifact nhỏ và hình cần thiết của mình.
6. Merge lần lượt bốn branch. Do đường dẫn tách biệt, conflict dự kiến rất thấp.
7. Điều phối thực hiện một commit tích hợp cuối trên notebook chung sau khi bốn branch đã merge.

Gợi ý commit message:

```text
feat(common): add Titanic EDA and shared feature pipeline
feat(logreg): train and tune logistic regression
feat(extra-trees): train and tune extra trees classifier
feat(svm): train and tune support vector classifier
feat(decision-tree): train and tune decision tree classifier
feat(integration): compare models and select final estimator
```

Hiện thư mục Mini_Capstone chưa phải Git repository. Cần thống nhất repo root trước khi áp dụng branch/commit workflow.

## 9. Definition of Done toàn dự án

- Tất cả notebook chạy từ đầu đến cuối trong môi trường thống nhất.
- Bốn model dùng cùng feature manifest và cùng validation split.
- Mỗi model có baseline, cross-validation, tuning, hold-out evaluation và artifact đầy đủ.
- Không scale ngoài Pipeline đối với Logistic Regression và SVM.
- Không dùng test Kaggle để lựa chọn hoặc tối ưu model.
- Notebook chung đọc được đầu ra của cả bốn model mà không cần copy/paste code huấn luyện.
- Có bảng so sánh, lựa chọn model cuối và file submission đúng hai cột `PassengerId`, `Survived`.

