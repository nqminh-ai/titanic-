# Kế hoạch — Support Vector Machine

## Phạm vi

Triển khai, đánh giá và tối ưu SVM/SVC trong `03_svm.ipynb`. Notebook chỉ dùng shared artifacts, không tự tạo lại feature hoặc split.

## Data contract bắt buộc

- `RANDOM_STATE = 42`.
- Cross-validation: `StratifiedKFold(n_splits=10, shuffle=True, random_state=42)`.
- Metric tuning chính: `accuracy`.
- Metric hold-out: accuracy, precision, recall, F1, ROC-AUC, classification report và confusion matrix.
- Scaling phải nằm bên trong pipeline.
- Không dùng validation hoặc test Kaggle trong tuning.

## Pipeline baseline

```python
Pipeline([
    ("scaler", StandardScaler()),
    ("model", SVC(kernel="rbf", probability=True, random_state=42)),
])
```

## Các bước thực hiện

1. Đọc manifest và shared artifacts; xác nhận schema, shape, NaN và infinity.
2. Chạy baseline 10-fold CV trên `X_train`, `y_train`.
3. Fit baseline và đánh giá shared validation split.
4. Tạo list-of-dicts grid để không tìm `gamma` cho linear kernel.
5. Chạy `GridSearchCV` với shared CV và `scoring="accuracy"`.
6. Báo cáo best params, CV mean/std, thời gian tuning và top configurations.
7. Đánh giá best estimator trên validation bằng toàn bộ metric.
8. Vẽ confusion matrix, ROC curve và learning curve.
9. Báo cáo số support vectors theo lớp.
10. So sánh RBF và linear kernel; nhận xét độ phức tạp và khả năng tổng quát hóa.
11. Refit best pipeline trên `X_full`, `y_full`, dự đoán test và xuất artifacts.

## Không gian tuning đề xuất

```python
[
    {
        "model__kernel": ["rbf"],
        "model__C": [0.1, 1, 10, 50, 100],
        "model__gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1],
        "model__class_weight": [None, "balanced"],
    },
    {
        "model__kernel": ["linear"],
        "model__C": [0.1, 1, 10, 50, 100],
        "model__class_weight": [None, "balanced"],
    },
]
```

`probability=True` làm tăng thời gian fit. Nếu tuning quá chậm, có thể tune với `probability=False`, sau đó dựng lại estimator tốt nhất với `probability=True` và cùng tham số để refit. Phải ghi rõ cách làm trong notebook.

## Đầu ra

Lưu trong `artifacts/svm/`:

- `best_model.joblib`
- `metrics.json`
- `cv_results.csv`
- `validation_predictions.csv`
- `test_predictions.csv`
- `submission.csv`

Lưu biểu đồ trong `reports/figures/svm/`.

## Tiêu chí hoàn thành

- Notebook chạy từ đầu đến cuối không lỗi.
- Scaler không bị fit ngoài CV folds.
- Có báo cáo support vectors và chi phí thời gian.
- Validation predictions khớp shared validation rows.
- Submission gồm đúng `PassengerId`, `Survived`.

