# Kế hoạch — Extra Trees

## Phạm vi

Triển khai, đánh giá và tối ưu `ExtraTreesClassifier` trong `02_extra_trees.ipynb`. Notebook phải sử dụng shared artifacts và shared split; không xử lý lại dữ liệu Titanic thô.

## Data contract bắt buộc

- `RANDOM_STATE = 42`.
- Cross-validation: `StratifiedKFold(n_splits=10, shuffle=True, random_state=42)`.
- Metric tuning chính: `accuracy`.
- Metric hold-out: accuracy, precision, recall, F1, ROC-AUC, classification report và confusion matrix.
- Không scale dữ liệu cho Extra Trees.
- Không dùng validation hoặc test Kaggle để lựa chọn tham số.

## Baseline

```python
ExtraTreesClassifier(
    random_state=42,
    n_jobs=-1,
)
```

## Các bước thực hiện

1. Đọc manifest và shared artifacts; kiểm tra schema, shape, NaN và infinity.
2. Chạy baseline 10-fold CV trên `X_train`, `y_train`.
3. Fit baseline và đánh giá trên shared validation split.
4. Thiết kế grid có kích thước phù hợp; dùng coarse-to-fine nếu full grid quá lớn.
5. Chạy `GridSearchCV` với shared CV và `scoring="accuracy"`.
6. Báo cáo best params, best CV score, độ lệch chuẩn, thời gian và top configurations.
7. Đánh giá best estimator trên validation bằng toàn bộ metric.
8. Vẽ confusion matrix, ROC curve và learning curve.
9. Báo cáo feature importance và lưu biểu đồ top features.
10. So sánh train/validation/CV để nhận diện overfitting.
11. Refit best estimator trên `X_full`, `y_full`, dự đoán test và xuất artifacts.

## Không gian tuning đề xuất

```python
{
    "n_estimators": [100, 300, 500],
    "max_depth": [None, 4, 8, 12],
    "max_features": ["sqrt", "log2", None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "criterion": ["gini", "entropy"],
    "class_weight": [None, "balanced"],
}
```

Full grid khá lớn. Nên chạy một grid coarse nhỏ trước, sau đó thu hẹp quanh vùng tốt nhất và ghi rõ lý do thay đổi.

## Đầu ra

Lưu trong `artifacts/extra_trees/`:

- `best_model.joblib`
- `metrics.json`
- `cv_results.csv`
- `validation_predictions.csv`
- `test_predictions.csv`
- `submission.csv`

Lưu biểu đồ trong `reports/figures/extra_trees/`.

## Tiêu chí hoàn thành

- Notebook chạy từ đầu đến cuối không lỗi.
- Kết quả có thể tái lập với seed 42.
- Có phân tích feature importance và overfitting.
- Validation predictions khớp shared validation rows.
- Submission gồm đúng `PassengerId`, `Survived`.

