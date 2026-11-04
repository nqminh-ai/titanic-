# Kế hoạch — Decision Tree

## Phạm vi

Triển khai, đánh giá và tối ưu `DecisionTreeClassifier` trong `04_decision_tree.ipynb`. Notebook phải đọc shared artifacts và không tự thay đổi feature engineering hoặc split.

## Data contract bắt buộc

- `RANDOM_STATE = 42`.
- Cross-validation: `StratifiedKFold(n_splits=10, shuffle=True, random_state=42)`.
- Metric tuning chính: `accuracy`.
- Metric hold-out: accuracy, precision, recall, F1, ROC-AUC, classification report và confusion matrix.
- Không scale dữ liệu cho Decision Tree.
- Không dùng validation hoặc test Kaggle để chọn tham số.

## Baseline

```python
DecisionTreeClassifier(random_state=42)
```

## Các bước thực hiện

1. Đọc manifest và shared artifacts; kiểm tra schema, shape, NaN và infinity.
2. Chạy baseline 10-fold CV trên `X_train`, `y_train`.
3. Fit baseline và đánh giá trên shared validation split.
4. Chạy `GridSearchCV` với shared CV và `scoring="accuracy"`.
5. Khảo sát kiểm soát độ phức tạp qua depth, minimum samples và tùy chọn pruning.
6. Báo cáo best params, CV mean/std, thời gian tuning và top configurations.
7. Đánh giá best estimator trên validation bằng toàn bộ metric.
8. Vẽ confusion matrix, ROC curve và learning curve.
9. Báo cáo feature importance, độ sâu thực tế và số lá.
10. Vẽ cây tối ưu; nếu cây lớn chỉ hiển thị ba tầng đầu.
11. So sánh train/validation score để phân tích overfitting.
12. Refit trên `X_full`, `y_full`, dự đoán test và xuất artifacts.

## Không gian tuning đề xuất

```python
{
    "criterion": ["gini", "entropy", "log_loss"],
    "max_depth": [None, 2, 3, 4, 5, 6, 8, 10],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 4, 8, 12],
    "max_features": [None, "sqrt", "log2"],
    "class_weight": [None, "balanced"],
}
```

Chỉ dùng `criterion="log_loss"` nếu phiên bản scikit-learn hỗ trợ. Có thể khảo sát `ccp_alpha` trong một bước pruning riêng để giữ grid chính ở kích thước hợp lý.

## Đầu ra

Lưu trong `artifacts/decision_tree/`:

- `best_model.joblib`
- `metrics.json`
- `cv_results.csv`
- `validation_predictions.csv`
- `test_predictions.csv`
- `submission.csv`

Lưu biểu đồ trong `reports/figures/decision_tree/`.

## Tiêu chí hoàn thành

- Notebook chạy từ đầu đến cuối không lỗi.
- Có so sánh rõ baseline và cây đã regularize/prune.
- Báo cáo độ sâu, số lá và feature importance.
- Validation predictions khớp shared validation rows.
- Submission gồm đúng `PassengerId`, `Survived`.

