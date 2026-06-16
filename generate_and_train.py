"""
Student Academic Performance Prediction
Dataset generation + Model training script
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, f1_score)
import joblib
import json
import os

np.random.seed(42)
N = 800

# ── Feature generation ──────────────────────────────────────────────
study_hours      = np.random.uniform(0.5, 10.0, N)
attendance       = np.random.uniform(40, 100, N)
assignments      = np.random.randint(0, 11, N)          # 0-10
prev_marks       = np.random.uniform(30, 100, N)
participation    = np.random.randint(1, 6, N)            # 1-5 scale
sleep_hours      = np.random.uniform(4, 10, N)
extracurricular  = np.random.randint(0, 4, N)            # 0-3 activities

# ── Label generation (rule-based with noise) ─────────────────────────
score = (
    study_hours     * 3.5 +
    attendance      * 0.25 +
    assignments     * 4.0 +
    prev_marks      * 0.30 +
    participation   * 2.5 +
    sleep_hours     * 1.0 +
    extracurricular * 1.5 +
    np.random.normal(0, 5, N)
)

# Map score to grade buckets
def score_to_grade(s):
    p = np.percentile(score, [25, 55, 80])
    if s < p[0]:   return "Poor"
    elif s < p[1]: return "Average"
    elif s < p[2]: return "Good"
    else:          return "Excellent"

grades = np.array([score_to_grade(s) for s in score])

df = pd.DataFrame({
    "study_hours_per_day":      np.round(study_hours, 2),
    "attendance_percentage":    np.round(attendance, 1),
    "assignments_completed":    assignments,
    "previous_semester_marks":  np.round(prev_marks, 1),
    "class_participation":      participation,
    "sleep_hours":              np.round(sleep_hours, 1),
    "extracurricular_activities": extracurricular,
    "performance_grade":        grades
})

os.makedirs("data",  exist_ok=True)
os.makedirs("model", exist_ok=True)
df.to_csv("data/student_performance.csv", index=False)
print(f"Dataset saved  →  {len(df)} records")
print(df["performance_grade"].value_counts())

# ── Preprocessing ────────────────────────────────────────────────────
X = df.drop("performance_grade", axis=1)
y = df["performance_grade"]

le = LabelEncoder()
y_enc = le.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

# ── Train models & pick best ─────────────────────────────────────────
models = {
    "RandomForest":        RandomForestClassifier(n_estimators=200, max_depth=10,
                                                   random_state=42, n_jobs=-1),
    "GradientBoosting":    GradientBoostingClassifier(n_estimators=150,
                                                       learning_rate=0.1,
                                                       max_depth=5, random_state=42),
    "LogisticRegression":  LogisticRegression(max_iter=1000, random_state=42),
}

results = {}
for name, mdl in models.items():
    cv = cross_val_score(mdl, X_scaled, y_enc, cv=5, scoring="accuracy")
    results[name] = cv.mean()
    print(f"{name:25s}  CV Acc = {cv.mean():.4f} ± {cv.std():.4f}")

best_name = max(results, key=results.get)
best_model = models[best_name]
best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)
acc    = accuracy_score(y_test, y_pred)
f1     = f1_score(y_test, y_pred, average="weighted")
report = classification_report(y_test, y_pred,
                                target_names=le.classes_, output_dict=True)
cm     = confusion_matrix(y_test, y_pred).tolist()

print(f"\nBest model : {best_name}")
print(f"Test Acc   : {acc:.4f}")
print(f"F1 (wtd)   : {f1:.4f}")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ── Feature importances (RF / GB have them; LR uses coef_) ──────────
if hasattr(best_model, "feature_importances_"):
    importances = best_model.feature_importances_.tolist()
else:
    importances = np.abs(best_model.coef_).mean(axis=0).tolist()

feature_names = X.columns.tolist()

# ── Save artefacts ────────────────────────────────────────────────────
joblib.dump(best_model, "model/model.pkl")
joblib.dump(scaler,     "model/scaler.pkl")
joblib.dump(le,         "model/label_encoder.pkl")

meta = {
    "best_model":      best_name,
    "test_accuracy":   round(acc, 4),
    "f1_score":        round(f1, 4),
    "classes":         le.classes_.tolist(),
    "feature_names":   feature_names,
    "feature_importances": dict(zip(feature_names, [round(x, 4) for x in importances])),
    "confusion_matrix": cm,
    "classification_report": report,
    "model_comparison": {k: round(v, 4) for k, v in results.items()},
}
with open("model/meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("\nModel artefacts saved to model/")
