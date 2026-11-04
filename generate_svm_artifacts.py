import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import json, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV, learning_curve
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve,
    average_precision_score
)
from sklearn.decomposition import PCA

# Setup paths
ROOT = Path.cwd()
ARTIFACT_COMMON = ROOT / 'artifacts' / 'common'
ARTIFACT_SVM = ROOT / 'artifacts' / 'svm'
FIGURE_SVM = ROOT / 'reports' / 'figures' / 'svm'

ARTIFACT_SVM.mkdir(parents=True, exist_ok=True)
FIGURE_SVM.mkdir(parents=True, exist_ok=True)

sns.set_theme(style='whitegrid', font='sans-serif')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 12

print("Step 1: Loading common artifacts...")
X_train = pd.read_csv(ARTIFACT_COMMON / 'X_train.csv')
y_train = pd.read_csv(ARTIFACT_COMMON / 'y_train.csv').squeeze('columns')
X_valid = pd.read_csv(ARTIFACT_COMMON / 'X_valid.csv')
y_valid = pd.read_csv(ARTIFACT_COMMON / 'y_valid.csv').squeeze('columns')
X_full = pd.read_csv(ARTIFACT_COMMON / 'X_full.csv')
y_full = pd.read_csv(ARTIFACT_COMMON / 'y_full.csv').squeeze('columns')
X_test = pd.read_csv(ARTIFACT_COMMON / 'X_test.csv')
test_passenger_id = pd.read_csv(ARTIFACT_COMMON / 'test_passenger_id.csv')

RANDOM_STATE = 42
CV = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

print("Step 2: Baseline Model...")
baseline_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE))
])

baseline_cv_scores = cross_val_score(baseline_pipe, X_train, y_train, cv=CV, scoring='accuracy', n_jobs=1)
baseline_cv_mean = float(baseline_cv_scores.mean())
baseline_cv_std = float(baseline_cv_scores.std())
print(f"Baseline CV Accuracy: {baseline_cv_mean:.4f} +/- {baseline_cv_std:.4f}")

baseline_pipe.fit(X_train, y_train)
y_pred_base = baseline_pipe.predict(X_valid)
y_score_base = baseline_pipe.predict_proba(X_valid)[:, 1]

base_acc = float(accuracy_score(y_valid, y_pred_base))
base_prec = float(precision_score(y_valid, y_pred_base))
base_rec = float(recall_score(y_valid, y_pred_base))
base_f1 = float(f1_score(y_valid, y_pred_base))
base_auc = float(roc_auc_score(y_valid, y_score_base))

print(f"Baseline Valid: Acc={base_acc:.4f}, Prec={base_prec:.4f}, Rec={base_rec:.4f}, F1={base_f1:.4f}, AUC={base_auc:.4f}")

print("Step 3: Hyperparameter Tuning via GridSearchCV...")
param_grid = [
    {
        'model__kernel': ['rbf'],
        'model__C': [0.1, 1, 10, 50, 100],
        'model__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        'model__class_weight': [None, 'balanced'],
    },
    {
        'model__kernel': ['linear'],
        'model__C': [0.1, 1, 10],
        'model__class_weight': [None, 'balanced'],
    },
]

tune_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', SVC(random_state=RANDOM_STATE, probability=False))
])

t0 = time.time()
grid = GridSearchCV(tune_pipe, param_grid, cv=CV, scoring='accuracy', n_jobs=1, return_train_score=True)
grid.fit(X_train, y_train)
tune_time = time.time() - t0
print(f"Tuning finished in {tune_time:.2f}s")
print(f"Best CV Score: {grid.best_score_:.4f}")
print(f"Best Params: {grid.best_params_}")

# Export cv_results
cv_results_df = pd.DataFrame(grid.cv_results_)
cv_results_df.to_csv(ARTIFACT_SVM / 'cv_results.csv', index=False)

print("Step 4: Refit Best Estimator with probability=True on X_train...")
best_params = {k.replace('model__', ''): v for k, v in grid.best_params_.items()}
best_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', SVC(**best_params, probability=True, random_state=RANDOM_STATE))
])
best_pipe.fit(X_train, y_train)

y_pred_tuned = best_pipe.predict(X_valid)
y_score_tuned = best_pipe.predict_proba(X_valid)[:, 1]

tuned_acc = float(accuracy_score(y_valid, y_pred_tuned))
tuned_prec = float(precision_score(y_valid, y_pred_tuned))
tuned_rec = float(recall_score(y_valid, y_pred_tuned))
tuned_f1 = float(f1_score(y_valid, y_pred_tuned))
tuned_auc = float(roc_auc_score(y_valid, y_score_tuned))
n_sv = best_pipe.named_steps['model'].n_support_.tolist()

print(f"Tuned Valid: Acc={tuned_acc:.4f}, Prec={tuned_prec:.4f}, Rec={tuned_rec:.4f}, F1={tuned_f1:.4f}, AUC={tuned_auc:.4f}")
print(f"Support Vectors per class: {n_sv} (Total: {sum(n_sv)})")

print("Step 5: Visualizations...")
# 1. Confusion Matrix Comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
cm_base = confusion_matrix(y_valid, y_pred_base)
cm_tuned = confusion_matrix(y_valid, y_pred_tuned)

sns.heatmap(cm_base, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False,
            xticklabels=['Perished (0)', 'Survived (1)'], yticklabels=['Perished (0)', 'Survived (1)'])
axes[0].set_title(f"Baseline SVM (RBF default)\nValidation Accuracy: {base_acc:.2%}", fontweight='bold')
axes[0].set_ylabel('True Label')
axes[0].set_xlabel('Predicted Label')

sns.heatmap(cm_tuned, annot=True, fmt='d', cmap='Greens', ax=axes[1], cbar=False,
            xticklabels=['Perished (0)', 'Survived (1)'], yticklabels=['Perished (0)', 'Survived (1)'])
axes[1].set_title(f"Tuned SVM (C=100, $\\gamma=0.001$)\nValidation Accuracy: {tuned_acc:.2%}", fontweight='bold')
axes[1].set_ylabel('True Label')
axes[1].set_xlabel('Predicted Label')

plt.suptitle("Confusion Matrix Comparison on Hold-out Validation Set (N=179)", fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(FIGURE_SVM / '01_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. ROC & Precision-Recall Curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fpr_b, tpr_b, _ = roc_curve(y_valid, y_score_base)
fpr_t, tpr_t, _ = roc_curve(y_valid, y_score_tuned)

axes[0].plot(fpr_b, tpr_b, label=f"Baseline RBF (AUC = {base_auc:.3f})", color='#1f77b4', lw=2)
axes[0].plot(fpr_t, tpr_t, label=f"Tuned RBF (AUC = {tuned_auc:.3f})", color='#2ca02c', lw=2.5)
axes[0].plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.7, label='Random Chance (AUC = 0.500)')
axes[0].set_xlabel('False Positive Rate (1 - Specificity)')
axes[0].set_ylabel('True Positive Rate (Recall / Sensitivity)')
axes[0].set_title('Receiver Operating Characteristic (ROC) Curve', fontweight='bold')
axes[0].legend(loc='lower right', frameon=True)
axes[0].grid(True, alpha=0.3)

prec_b, rec_b, _ = precision_recall_curve(y_valid, y_score_base)
prec_t, rec_t, _ = precision_recall_curve(y_valid, y_score_tuned)
ap_b = average_precision_score(y_valid, y_score_base)
ap_t = average_precision_score(y_valid, y_score_tuned)

axes[1].plot(rec_b, prec_b, label=f"Baseline RBF (AP = {ap_b:.3f})", color='#1f77b4', lw=2)
axes[1].plot(rec_t, prec_t, label=f"Tuned RBF (AP = {ap_t:.3f})", color='#2ca02c', lw=2.5)
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].set_title('Precision-Recall (PR) Curve', fontweight='bold')
axes[1].legend(loc='lower left', frameon=True)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURE_SVM / '02_roc_pr_curves.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Learning Curve
train_sizes, train_scores, val_scores = learning_curve(
    best_pipe, X_train, y_train, cv=CV, scoring='accuracy',
    train_sizes=np.linspace(0.1, 1.0, 8), n_jobs=1, random_state=RANDOM_STATE
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

plt.figure(figsize=(9, 5.5))
plt.plot(train_sizes, train_mean, 'o-', color='#1f77b4', lw=2, label='Training score')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='#1f77b4')
plt.plot(train_sizes, val_mean, 'o-', color='#2ca02c', lw=2, label='10-Fold CV Validation score')
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color='#2ca02c')
plt.title(f"Learning Curve for Tuned SVM (C=100, $\\gamma=0.001$, RBF Kernel)", fontweight='bold')
plt.xlabel("Training Set Size (samples)")
plt.ylabel("Accuracy Score")
plt.ylim(0.70, 0.90)
plt.legend(loc='lower right', frameon=True)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURE_SVM / '03_learning_curve.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Hyperparameter Score Heatmap for RBF (C vs gamma)
rbf_results = cv_results_df[(cv_results_df['param_model__kernel'] == 'rbf') & 
                            (cv_results_df['param_model__class_weight'].isna())].copy()
rbf_results['gamma_str'] = rbf_results['param_model__gamma'].astype(str)
pivot = rbf_results.pivot(index='param_model__C', columns='gamma_str', values='mean_test_score')
cols_order = [c for c in ['0.001', '0.01', '0.1', '1', '1.0', 'scale', 'auto'] if c in pivot.columns]
pivot = pivot[cols_order]

plt.figure(figsize=(9, 6))
sns.heatmap(pivot, annot=True, fmt='.3f', cmap='viridis', cbar_kws={'label': 'Mean 10-Fold CV Accuracy'})
plt.title("GridSearchCV Performance Surface: RBF Kernel ($C$ vs $\\gamma$)", fontweight='bold')
plt.xlabel("Gamma ($\\gamma$) Bandwidth Parameter")
plt.ylabel("Cost / Penalty Parameter ($C$)")
plt.tight_layout()
plt.savefig(FIGURE_SVM / '04_hyperparameter_surface.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. Support Vector & Decision Function Distribution
dec_func = best_pipe.decision_function(X_valid)
plt.figure(figsize=(9, 5.5))
sns.kdeplot(dec_func[y_valid == 0], label='Perished (Actual 0)', color='#d62728', fill=True, alpha=0.4, lw=2)
sns.kdeplot(dec_func[y_valid == 1], label='Survived (Actual 1)', color='#2ca02c', fill=True, alpha=0.4, lw=2)
plt.axvline(0, color='black', linestyle='--', lw=2, label='Optimal Decision Boundary ($f(x)=0$)')
plt.axvline(-1, color='gray', linestyle=':', lw=1.5, label='Negative Margin Boundary ($f(x)=-1$)')
plt.axvline(1, color='gray', linestyle=':', lw=1.5, label='Positive Margin Boundary ($f(x)=+1$)')
plt.title("Distribution of SVM Decision Function Values on Validation Set", fontweight='bold')
plt.xlabel("Signed Distance to Hyperplane: $f(x) = w^T \\phi(x) + b$")
plt.ylabel("Density")
plt.legend(frameon=True)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURE_SVM / '05_decision_function_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# 6. 2D Decision Boundary via PCA projection
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_train_scaled = best_pipe.named_steps['scaler'].transform(X_train)
X_train_pca = pca.fit_transform(X_train_scaled)

svm_pca = SVC(C=100, gamma=0.001, kernel='rbf', random_state=RANDOM_STATE)
svm_pca.fit(X_train_pca, y_train)

svm_pca_lin = SVC(C=0.1, kernel='linear', random_state=RANDOM_STATE)
svm_pca_lin.fit(X_train_pca, y_train)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

# Linear Boundary
Z_lin = svm_pca_lin.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z_lin = Z_lin.reshape(xx.shape)
axes[0].contourf(xx, yy, Z_lin > 0, alpha=0.25, cmap='RdYlGn')
axes[0].contour(xx, yy, Z_lin, colors=['k', 'k', 'k'], linestyles=[':', '-', ':'], levels=[-1, 0, 1])
axes[0].scatter(X_train_pca[y_train == 0, 0], X_train_pca[y_train == 0, 1], c='#d62728', label='Perished (0)', alpha=0.6, s=30)
axes[0].scatter(X_train_pca[y_train == 1, 0], X_train_pca[y_train == 1, 1], c='#2ca02c', label='Survived (1)', alpha=0.6, s=30)
axes[0].set_title("Linear Kernel Margin & Boundary (PCA 2D)", fontweight='bold')
axes[0].set_xlabel("Principal Component 1")
axes[0].set_ylabel("Principal Component 2")
axes[0].legend(loc='upper right', frameon=True)

# RBF Boundary
Z_rbf = svm_pca.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z_rbf = Z_rbf.reshape(xx.shape)
axes[1].contourf(xx, yy, Z_rbf > 0, alpha=0.25, cmap='RdYlGn')
axes[1].contour(xx, yy, Z_rbf, colors=['k', 'k', 'k'], linestyles=[':', '-', ':'], levels=[-1, 0, 1])
axes[1].scatter(X_train_pca[y_train == 0, 0], X_train_pca[y_train == 0, 1], c='#d62728', label='Perished (0)', alpha=0.6, s=30)
axes[1].scatter(X_train_pca[y_train == 1, 0], X_train_pca[y_train == 1, 1], c='#2ca02c', label='Survived (1)', alpha=0.6, s=30)
axes[1].set_title("RBF Kernel Non-linear Warping & Margin (PCA 2D)", fontweight='bold')
axes[1].set_xlabel("Principal Component 1")
axes[1].set_ylabel("Principal Component 2")
axes[1].legend(loc='upper right', frameon=True)

plt.suptitle("Geometric Comparison of SVM Decision Boundaries: Linear vs RBF Kernel", fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(FIGURE_SVM / '06_decision_boundary_2d.png', dpi=300, bbox_inches='tight')
plt.close()

print("Step 6: Refit Full Data & Export Artifacts...")
full_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', SVC(**best_params, probability=True, random_state=RANDOM_STATE))
])
full_pipe.fit(X_full, y_full)

joblib.dump(full_pipe, ARTIFACT_SVM / 'best_model.joblib')

# Validation predictions
val_preds_df = pd.DataFrame({
    'row_id': X_valid.index,
    'y_true': y_valid.values,
    'y_pred': y_pred_tuned,
    'y_score': y_score_tuned
})
val_preds_df.to_csv(ARTIFACT_SVM / 'validation_predictions.csv', index=False)

# Test predictions
test_preds = full_pipe.predict(X_test)
test_scores = full_pipe.predict_proba(X_test)[:, 1]

test_preds_df = pd.DataFrame({
    'PassengerId': test_passenger_id['PassengerId'],
    'Survived': test_preds,
    'score': test_scores
})
test_preds_df.to_csv(ARTIFACT_SVM / 'test_predictions.csv', index=False)

# Submission
sub_df = pd.DataFrame({
    'PassengerId': test_passenger_id['PassengerId'],
    'Survived': test_preds
})
sub_df.to_csv(ARTIFACT_SVM / 'submission.csv', index=False)

# Metrics JSON
metrics_data = {
    'model_name': 'Support Vector Machine (SVC)',
    'random_state': RANDOM_STATE,
    'cv_folds': 10,
    'baseline_cv_mean': baseline_cv_mean,
    'baseline_cv_std': baseline_cv_std,
    'best_cv_score': float(grid.best_score_),
    'best_params': best_params,
    'validation_accuracy': tuned_acc,
    'validation_precision': tuned_prec,
    'validation_recall': tuned_rec,
    'validation_f1': tuned_f1,
    'validation_roc_auc': tuned_auc,
    'support_vectors_per_class': n_sv,
    'total_support_vectors': sum(n_sv),
    'full_model_support_vectors': full_pipe.named_steps['model'].n_support_.tolist(),
    'tuning_time_seconds': round(tune_time, 2)
}

with open(ARTIFACT_SVM / 'metrics.json', 'w', encoding='utf-8') as f:
    json.dump(metrics_data, f, indent=2)

print("ALL STEPS COMPLETED SUCCESSFULLY!")
