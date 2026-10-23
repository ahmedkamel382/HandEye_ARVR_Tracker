"""TEST-ONLY copy of the feature maths in ai_pipeline/hand_kinematics/hand_features.py, used to train the
stand-in hand model of the end-to-end test. The real model/feature code is exercised by the parity test."""
import numpy as np

ANGLE_TRIPLETS = {
    "thumb_cmc": (0, 1, 2), "thumb_mcp": (1, 2, 3), "thumb_ip": (2, 3, 4),
    "index_mcp": (0, 5, 6), "index_pip": (5, 6, 7), "index_dip": (6, 7, 8),
    "middle_mcp": (0, 9, 10), "middle_pip": (9, 10, 11), "middle_dip": (10, 11, 12),
    "ring_mcp": (0, 13, 14), "ring_pip": (13, 14, 15), "ring_dip": (14, 15, 16),
    "pinky_mcp": (0, 17, 18), "pinky_pip": (17, 18, 19), "pinky_dip": (18, 19, 20),
}
FINGER_TIPS = {"index": 8, "middle": 12, "ring": 16, "pinky": 20}
FEATURE_NAMES = ([f"ang_{n}" for n in ANGLE_TRIPLETS] + [f"thumb_to_{f}_tip" for f in FINGER_TIPS]
                 + [f"wrist_to_{f}_tip" for f in ("thumb", *FINGER_TIPS)])


def _angle(a, b, c):
    v1, v2 = a - b, c - b
    d = np.linalg.norm(v1) * np.linalg.norm(v2)
    return 0.0 if d < 1e-9 else float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / d, -1.0, 1.0))))


def extract_features(p):
    palm = np.linalg.norm(p[9] - p[0])
    palm = 1.0 if palm < 1e-9 else palm
    ang = [_angle(p[a], p[b], p[c]) for a, b, c in ANGLE_TRIPLETS.values()]
    td = [np.linalg.norm(p[4] - p[t]) / palm for t in FINGER_TIPS.values()]
    wd = [np.linalg.norm(p[t] - p[0]) / palm for t in (4, *FINGER_TIPS.values())]
    return np.array(ang + td + wd, dtype=np.float64)


def train_standin_model(path, seed=0):
    """Random Forest on synthetic neutral(0)/pinch(1)/fist(2) hands, saved as {"model", "feature_names"}."""
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    import synthetic_landmarks as syn
    rng = np.random.default_rng(seed)
    X, y = [], []
    for label in (0, 1, 2):
        for _ in range(250):
            tip = (float(rng.uniform(0.2, 0.8)), float(rng.uniform(0.2, 0.6)))
            scale = float(rng.uniform(0.7, 1.4))
            kw = dict(tip=tip, scale=scale, rng=rng, noise=0.002)
            if label == 1:
                kw["pinch"] = True
            if label == 2:
                kw["curled"] = (True, True, True, True)
            img = syn.make_hand(**kw)
            X.append(extract_features(syn.make_hand_world(img, rng, 0.002).astype(np.float64)))
            y.append(label)
    model = RandomForestClassifier(n_estimators=40, min_samples_leaf=2, random_state=seed).fit(np.array(X), np.array(y))
    joblib.dump({"model": model, "feature_names": FEATURE_NAMES}, path)
    return model
