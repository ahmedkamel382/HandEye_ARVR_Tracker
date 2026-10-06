"""
Trains the gesture Random Forest from datasets/hand_dataset_v2*.csv and saves
<repo_root>/assets/hand_gesture_rf.pkl (loaded automatically by hand_tracker.py).

Run from ai_pipeline/hand_kinematics/:   python model_trainer.py
Needs: pip install scikit-learn pandas joblib
"""
import glob
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, cross_val_predict

from hand_features import FEATURE_NAMES

script_dir = os.path.dirname(os.path.abspath(__file__))
files = glob.glob(os.path.join(script_dir, 'datasets', 'hand_dataset_v2*.csv'))
if not files:
    raise SystemExit("No hand_dataset_v2*.csv found. Run hand_data_collector.py first.")

df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
X, y, groups = df[FEATURE_NAMES].values, df['Label'].values, df['Session'].values
print(f"Loaded {len(df)} frames from {len(files)} file(s), {len(set(groups))} session(s)")
print("Per class:", {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))})
if len(np.unique(y)) < 3:
    raise SystemExit("Need data for all 3 classes (0 Neutral, 1 Pinch, 2 Fist).")


def make_model():
    return RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=3,
                                  class_weight='balanced', random_state=42, n_jobs=-1)


# Honest evaluation. Neighbouring video frames are near-duplicates, so a random
# split would give a fake ~100%.
all_classes = set(np.unique(y))
sessions_complete = len(set(groups)) >= 2 and all(
    set(np.unique(y[groups == g])) == all_classes for g in set(groups))

if sessions_complete:
    print("\nEvaluation: leave-one-session-out cross-validation")
    cv, cv_groups = LeaveOneGroupOut(), groups
else:
    # Typical case: each run recorded a single gesture, so a held-out session
    # would contain a class the model never saw (-> meaningless 0%).
    # Instead: cut each class's frames (in recording order) into ~5 s chunks
    # and hold out whole chunks, so train/test never share neighbouring frames.
    print("\nEvaluation: chunked cross-validation (each session holds only some gestures)")
    CHUNK = 150
    order = np.argsort(df['Timestamp'].values, kind='stable')
    cv_groups = np.zeros(len(df), dtype=int)
    for c in np.unique(y):
        idx = order[y[order] == c]
        cv_groups[idx] = int(c) * 10000 + np.arange(len(idx)) // CHUNK
    cv = GroupKFold(n_splits=min(5, len(np.unique(cv_groups))))

pred = cross_val_predict(make_model(), X, y, groups=cv_groups, cv=cv)
eval_y = y

print(classification_report(eval_y, pred, target_names=['Neutral', 'Pinch', 'Fist'], digits=3))
print("Confusion matrix (rows=true, cols=pred):\n", confusion_matrix(eval_y, pred))

final = make_model().fit(X, y)
top = sorted(zip(final.feature_importances_, FEATURE_NAMES), reverse=True)[:8]
print("\nTop features:", ", ".join(f"{n} ({v:.2f})" for v, n in top))

out = os.path.abspath(os.path.join(script_dir, '..', '..', 'assets', 'hand_gesture_rf.pkl'))
os.makedirs(os.path.dirname(out), exist_ok=True)
joblib.dump({'model': final, 'feature_names': FEATURE_NAMES}, out, compress=3)
print(f"\nSaved model -> {out}")
