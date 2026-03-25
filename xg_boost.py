import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

df = pd.read_csv("osteoporosis.csv")
df = df.drop(columns=["Id"])

for col in df.columns:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mode()[0])

X = df.drop("Osteoporosis", axis=1)
y = df["Osteoporosis"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

label_encoders = {}
cat_cols = X_train.select_dtypes(include=["object"]).columns

for col in cat_cols:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col].astype(str))
    X_test[col] = X_test[col].astype(str).map(
        lambda val, le=le: le.transform([val])[0] if val in le.classes_ else -1
    )
    label_encoders[col] = le

X_train["Age_x_FamilyHistory"] = X_train["Age"] * X_train["Family History"]
X_test["Age_x_FamilyHistory"]  = X_test["Age"]  * X_test["Family History"]

drop_cols = [
    "Alcohol Consumption",
    "Medications",
    "Hormonal Changes",
    "Body Weight",
    "Gender",
    "Vitamin D Intake",
]
X_train = X_train.drop(columns=[c for c in drop_cols if c in X_train.columns])
X_test  = X_test.drop(columns=[c for c in drop_cols if c in X_test.columns])

feature_names = list(X_train.columns)
print(f"Features ({len(feature_names)}): {feature_names}\n")

neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
spw = neg / pos
print(f"scale_pos_weight: {spw:.2f}\n")

xgb = XGBClassifier(
    eval_metric="logloss",
    scale_pos_weight=spw,
    random_state=42
)

param_grid = {
    "n_estimators":     [300, 500],
    "max_depth":        [3, 4],
    "learning_rate":    [0.03, 0.05, 0.1],
    "reg_lambda":       [0, 1, 5],
    "reg_alpha":        [0, 1],
    "min_child_weight": [1, 3],
    "subsample":        [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}

print("Running GridSearchCV (this may take a few minutes)...")
gs = GridSearchCV(xgb, param_grid, cv=5, scoring="roc_auc", n_jobs=-1, verbose=1)
gs.fit(X_train, y_train)

best_xgb = gs.best_estimator_
print(f"\nBest params: {gs.best_params_}")
print(f"CV ROC-AUC : {gs.best_score_:.4f}\n")

cal_xgb = CalibratedClassifierCV(best_xgb, method="isotonic", cv=5)
cal_xgb.fit(X_train, y_train)

y_prob_raw = best_xgb.predict_proba(X_test)[:, 1]
y_prob_cal = cal_xgb.predict_proba(X_test)[:, 1]
y_pred = (y_prob_cal > 0.50).astype(int)
print("=" * 55)
print("XGBoost TEXT BRANCH — EVALUATION")
print("=" * 55)
print(f"Test Accuracy  : {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(f"ROC-AUC (raw)  : {roc_auc_score(y_test, y_prob_raw):.4f}")
print(f"ROC-AUC (cal)  : {roc_auc_score(y_test, y_prob_cal):.4f}")
print(f"\nClassification Report (threshold=0.40):\n")
print(classification_report(y_test, y_pred, target_names=["No Disease", "Disease"]))

fusion_output = pd.DataFrame({
    "prob_text_negative": cal_xgb.predict_proba(X_test)[:, 0],
    "prob_text_positive": y_prob_cal,
    "pred_text":          y_pred,
    "true_label":         y_test.values,
}, index=X_test.index)

print("\n" + "=" * 55)
print("FUSION LAYER — TEXT BRANCH SCORES (first 10 rows)")
print("=" * 55)
print(fusion_output.head(10).to_string())

print("\n" + "=" * 55)
print("FUSION LAYER — SCORE DISTRIBUTION SUMMARY")
print("=" * 55)
print(fusion_output[["prob_text_negative", "prob_text_positive"]].describe().round(4))

fusion_output.to_csv("xgb_text_branch_scores.csv")
print("\nSaved → xgb_text_branch_scores.csv")
print("Merge this with your CNN image branch scores on the same index before fusion.\n")

import joblib
joblib.dump(cal_xgb, "xgb_model.joblib")
joblib.dump(label_encoders, "label_encoders.joblib")

