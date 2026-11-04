# Kế hoạch — Logistic Regression

## Phạm vi

Triển khai, đánh giá và tối ưu Logistic Regression trong `01_logistic_regression.ipynb`. Notebook chỉ đọc dữ liệu từ `artifacts/common/`; không đọc lại CSV thô và không định nghĩa lại preprocessing hoặc hold-out split.

## Data contract bắt buộc

- `RANDOM_STATE = 42`.
- Dùng nguyên trạng `X_train`, `X_valid`, `y_train`, `y_valid`, `X_full`, `y_full`, `X_test` và `test_passenger_id` từ shared artifacts.
- Cross-validation: `StratifiedKFold(n_splits=10, shuffle=True, random_state=42)`.
- Metric tuning chính: `accuracy`.
- Metric hold-out: accuracy, precision, recall, F1, ROC-AUC, classification report và confusion matrix.
- Không dùng validation hoặc test Kaggle trong quá trình fit scaler và tuning.

## Pipeline mô hình

Dùng scaler bên trong pipeline để scaler chỉ được fit trên training fold:

```python
Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=5000, random_state=42)),
])
```

Không scale shared artifacts bên ngoài pipeline.

## Các bước thực hiện

1. Đọc manifest và shared artifacts; xác nhận schema, shape, NaN và infinity.
2. Fit baseline pipeline và chạy 10-fold CV trên `X_train`, `y_train`.
3. Đánh giá baseline trên shared validation split.
4. Chạy `GridSearchCV` với lưới tham số hợp lý.
5. Báo cáo cấu hình tốt nhất, CV mean/std, thời gian tuning và các cấu hình đứng đầu.
6. Đánh giá best estimator trên validation bằng toàn bộ metric bắt buộc.
7. Vẽ confusion matrix, ROC curve và learning curve.
8. Phân tích hệ số sau scaling; trình bày các feature có ảnh hưởng dương/âm lớn nhất.
9. Kiểm tra convergence warning; tăng `max_iter` nếu cần.
10. Refit best pipeline trên `X_full`, `y_full` và dự đoán `X_test`.
11. Xuất model, metrics, CV results, validation predictions, test predictions và submission.

## Không gian tuning đề xuất

```python
{
    "model__C": [0.001, 0.01, 0.1, 1, 10, 100],
    "model__penalty": ["l1", "l2"],
    "model__solver": ["liblinear"],
    "model__class_weight": [None, "balanced"],
}
```

Có thể thử penalty `None` bằng một grid riêng nếu phiên bản scikit-learn hỗ trợ. Không kết hợp solver và penalty không tương thích.

## Đầu ra

Lưu trong `artifacts/logistic_regression/`:

- `best_model.joblib`
- `metrics.json`
- `cv_results.csv`
- `validation_predictions.csv`
- `test_predictions.csv`
- `submission.csv`

Lưu biểu đồ trong `reports/figures/logistic_regression/`.

## Tiêu chí hoàn thành

- Notebook chạy từ đầu đến cuối không lỗi.
- Không có leakage do scaling.
- Validation predictions giữ đúng thứ tự các dòng shared validation.
- Submission gồm đúng hai cột `PassengerId`, `Survived`.
- Có so sánh baseline với tuned model và nhận xét về khả năng tổng quát hóa.

