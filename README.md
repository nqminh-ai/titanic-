# Mini Capstone — Titanic Survival Classification

Dự án xây dựng một quy trình machine learning có khả năng tái lập để dự đoán hành khách sống sót trên Titanic. Trọng tâm của dự án là chuẩn hóa EDA, preprocessing, feature engineering và cách chia dữ liệu trước khi huấn luyện, đánh giá bốn mô hình phân loại độc lập.

## Mục tiêu

- Khám phá và kiểm tra chất lượng dữ liệu Titanic.
- Xây dựng một bộ feature dùng chung, không xử lý lại dữ liệu riêng cho từng mô hình.
- So sánh Logistic Regression, Extra Trees, Support Vector Machine và Decision Tree trên cùng hold-out split và cross-validation contract.
- Đánh giá bằng accuracy, precision, recall, F1, ROC-AUC và confusion matrix.
- Xuất artifacts có thể đọc lại độc lập để hỗ trợ làm việc nhóm và tái lập kết quả.

## Cấu trúc dự án

```text
Mini_Capstone/
├── input/titanic/                 # Dữ liệu train/test Titanic
├── notebooks/
│   └── 00_common_pipeline.ipynb   # EDA và pipeline dữ liệu dùng chung
├── members/
│   ├── logistic_regression/
│   ├── extra_trees/
│   ├── svm/
│   └── decision_tree/             # Notebook và PLAN.md riêng của từng mô hình
├── artifacts/common/              # Dataset đã xử lý và metadata dùng chung
├── reports/figures/common/        # Biểu đồ EDA
├── PROJECT_PLAN.md                # Kế hoạch và data contract của dự án
└── README.md
```

## Common pipeline

Notebook `notebooks/00_common_pipeline.ipynb` thực hiện:

1. Đọc `train.csv` và `test.csv` bằng đường dẫn tương đối.
2. Audit shape, schema, dữ liệu trùng, tính duy nhất của `PassengerId`, missing values, phân bố target và outliers.
3. EDA đơn biến, hai biến và đa biến; lưu biểu đồ vào `reports/figures/common/`.
4. Điền dữ liệu thiếu và học ngưỡng feature chỉ từ tập train Titanic.
5. Tạo các feature `Title`, `FamilySize`, `FamilyCategory`, `AgeBand` và `FareBand`.
6. Mã hóa categorical nhất quán giữa train và test.
7. Tạo một hold-out split dùng chung với stratification.
8. Chạy baseline kiểm tra pipeline và xuất artifacts.
9. Đọc lại artifacts từ ổ đĩa để xác nhận khả năng tái sử dụng.

`PassengerId` không được dùng làm feature. Các cột `Name`, `Ticket` và `Cabin` bị loại sau khi khai thác thông tin cần thiết. Common pipeline không scale dữ liệu; Logistic Regression và SVM phải đặt `StandardScaler` bên trong pipeline riêng của mô hình.

## Data contract

- Random seed: `42`.
- Hold-out split: `test_size=0.20`, `stratify=y`, `random_state=42`.
- Cross-validation: `StratifiedKFold(n_splits=10, shuffle=True, random_state=42)`.
- Metric chính khi lựa chọn mô hình: `accuracy`.
- Metric bổ sung: precision, recall, F1, ROC-AUC và confusion matrix.
- Tập test Kaggle không được dùng để lựa chọn hoặc tối ưu mô hình.

Pipeline hiện xuất 28 feature và các tập dữ liệu có kích thước:

| Artifact | Shape |
|---|---:|
| `X_train.csv` | 712 × 28 |
| `X_valid.csv` | 179 × 28 |
| `X_full.csv` | 891 × 28 |
| `X_test.csv` | 418 × 28 |

## Yêu cầu môi trường

- Python 3.10 trở lên.
- Jupyter Notebook hoặc JupyterLab.
- Các thư viện chính: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `nbformat` và `nbclient`.

Có thể tạo môi trường ảo và cài thư viện như sau:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install numpy pandas matplotlib seaborn scikit-learn jupyter nbformat nbclient
```

## Cách chạy

Khởi động Jupyter từ thư mục gốc của dự án:

```powershell
jupyter lab
```

Mở và chạy toàn bộ cell của:

```text
notebooks/00_common_pipeline.ipynb
```

Chỉ bắt đầu notebook mô hình sau khi common pipeline chạy thành công. Mỗi notebook trong `members/` đọc trực tiếp dữ liệu từ `artifacts/common/` thay vì xử lý lại dữ liệu thô.

## Artifacts dùng chung

Common pipeline tạo các file sau trong `artifacts/common/`:

```text
X_train.csv
X_valid.csv
y_train.csv
y_valid.csv
X_full.csv
y_full.csv
X_test.csv
test_passenger_id.csv
feature_manifest.json
baseline_metrics.json
```

`feature_manifest.json` ghi lại danh sách và kiểu dữ liệu feature, seed, quy tắc split, CV contract, các thống kê học từ train, quy tắc feature engineering và phiên bản thư viện.

## Baseline hiện tại

Logistic Regression baseline sử dụng `StandardScaler` bên trong `Pipeline`:

| Metric | Kết quả |
|---|---:|
| 10-fold CV accuracy | 0.8216 ± 0.0442 |
| Validation accuracy | 0.8324 |
| Validation precision | 0.8095 |
| Validation recall | 0.7391 |
| Validation F1 | 0.7727 |
| Validation ROC-AUC | 0.8681 |

Các số liệu này dùng để kiểm tra tính hợp lệ của pipeline, không phải kết luận mô hình tốt nhất của dự án.

## Quy trình phát triển mô hình

Mỗi mô hình cần thực hiện cùng một quy trình:

1. Đọc shared artifacts và kiểm tra schema.
2. Xây dựng baseline.
3. Chạy 10-fold stratified cross-validation.
4. Tuning bằng `GridSearchCV` với `scoring="accuracy"`.
5. Đánh giá best estimator trên shared validation split.
6. Vẽ learning curve và các biểu đồ đánh giá phù hợp.
7. Refit trên `X_full`, `y_full`.
8. Dự đoán `X_test` và xuất submission.

Mỗi thư mục thành viên có `PLAN.md` mô tả pipeline, không gian tuning, đầu ra và tiêu chí hoàn thành dành riêng cho mô hình đó.

Không commit credentials, môi trường ảo, cache, file tạm hoặc model binary lớn. Xem `.gitignore` để biết danh sách loại trừ hiện tại.

## Trạng thái

- Common EDA và feature pipeline: hoàn thành.
- Shared artifacts và manifest: hoàn thành, đã kiểm tra đọc lại độc lập.
- Bốn notebook mô hình: đã tạo khung, chờ huấn luyện và tuning chi tiết.
- So sánh và lựa chọn mô hình cuối: chưa thực hiện.

