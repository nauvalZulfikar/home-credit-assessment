import os
import sys
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
import platform
import sklearn
import optbinning
from scipy.stats import norm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REP = os.environ.get("HC_REP", os.path.join(ROOT, "reports"))
MODELS = os.environ.get("HC_MODELS", os.path.join(ROOT, "models"))

os.makedirs(REP, exist_ok=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_lgbm import load_data, downcast_floats

def lgbm_predict(X, feats):
    boosters = [lgb.Booster(model_file=os.path.join(MODELS, f"lgbm_fold{k}.txt")) for k in range(5)]
    return np.mean([b.predict(X[feats]) for b in boosters], axis=0)

def sc_predict(X, sc, meta):
    Xs = X[meta["features"]].copy()
    for c in meta["cat_cols"]:
        Xs[c] = Xs[c].astype(str)
    return sc.predict_proba(Xs)[:, 1]

def shift(X, cols, delta):
    X = X.copy()
    for c in cols:
        X[c] = X[c].astype(float) + delta
    return X

def check(name, base, new, expect):
    d = new - base
    if expect == "same":
        ok = bool(np.max(np.abs(d)) == 0)
    elif expect == "down":
        ok = bool(d.mean() < 0 and (d <= 1e-9).mean() > 0.9)
    else:
        ok = bool(d.mean() > 0 and (d >= -1e-9).mean() > 0.9)
    return {"test": name, "expect": expect, "mean_change": float(d.mean()), "share_moved_right_way": float((d <= 1e-9).mean() if expect == "down" else (d >= -1e-9).mean()), "max_abs_change": float(np.max(np.abs(d))), "passed": ok}

def ab_sample_size(p1, p2, alpha=0.05, power=0.8):
    za, zb = norm.ppf(1 - alpha / 2), norm.ppf(power)
    pbar = (p1 + p2) / 2
    n = (za * np.sqrt(2 * pbar * (1 - pbar)) + zb * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2 / (p1 - p2) ** 2
    return int(np.ceil(n))

def main():
    df = downcast_floats(load_data())
    X = df[df.IS_TRAIN == 1].sample(n=3000, random_state=42).reset_index(drop=True)
    feats = json.load(open(os.path.join(MODELS, "lgbm_features.json")))
    sc = joblib.load(os.path.join(MODELS, "scorecard.pkl"))
    meta = json.load(open(os.path.join(MODELS, "scorecard_features.json")))
    
    bl = lgbm_predict(X, feats)
    bs = sc_predict(X, sc, meta)
    
    ext = [c for c in ["APP_EXT_SOURCE_1", "APP_EXT_SOURCE_2", "APP_EXT_SOURCE_3", "APP_EXT_MEAN", "APP_EXT_MIN", "APP_EXT_MAX"] if c in X.columns]
    late = [c for c in ["INS_LATE_MEAN", "INS_RECENT_LATE_RATE", "POS_LATE_MEAN", "CC_LATE_MEAN", "BUR_BB_LATE_MEAN"] if c in X.columns]
    dpd = [c for c in ["INS_DPD_MAX", "INS_DPD_MEAN", "POS_DPD_MAX", "CC_DPD_MAX", "BUR_BB_DPD_MAX"] if c in X.columns]
    
    tests = []
    
    Xg = X.copy()
    Xg["APP_CODE_GENDER"] = X["APP_CODE_GENDER"].astype(str).map({"M": "F", "F": "M"}).fillna("F").astype(X["APP_CODE_GENDER"].dtype)
    tests.append(check("LightGBM: gender flipped (not a model input)", bl, lgbm_predict(Xg, feats), "same"))
    tests.append(check("Scorecard: gender flipped (not a model input)", bs, sc_predict(Xg, sc, meta), "same"))
    
    Xi = X.copy()
    Xi["SK_ID_CURR"] = X["SK_ID_CURR"] + 10_000_000
    tests.append(check("LightGBM: client ID changed", bl, lgbm_predict(Xi, feats), "same"))
    tests.append(check("Scorecard: client ID changed", bs, sc_predict(Xi, sc, meta), "same"))
    
    Xs = shift(X, ext, 0.1)
    tests.append(check("LightGBM: external scores +0.1", bl, lgbm_predict(Xs, feats), "down"))
    tests.append(check("Scorecard: external scores +0.1", bs, sc_predict(Xs, sc, meta), "down"))
    
    Xs = shift(X, late, 0.2)
    tests.append(check("LightGBM: late-payment share +0.2", bl, lgbm_predict(Xs, feats), "up"))
    tests.append(check("Scorecard: late-payment share +0.2", bs, sc_predict(Xs, sc, meta), "up"))
    
    Xs = shift(X, dpd, 30)
    tests.append(check("LightGBM: days past due +30", bl, lgbm_predict(Xs, feats), "up"))
    tests.append(check("Scorecard: days past due +30", bs, sc_predict(Xs, sc, meta), "up"))
    
    ev = json.load(open(os.path.join(REP, "evaluate_summary.json")))
    p1 = ev["default_rate"]
    p2 = ev["at_10"]["LightGBM"]["approved_bad_rate"]
    
    ab = {
        "bad_rate_now": p1,
        "bad_rate_expected": p2,
        "alpha": 0.05,
        "power": 0.8,
        "clients_per_group": ab_sample_size(p1, p2),
        "clients_per_group_half_effect": ab_sample_size(p1, p1 - (p1 - p2) / 2),
        "note": "Bad rate is only known after the loans mature (months), so the test length = time to enrol this many clients + the observation window."
    }
    
    versions = {
        "python": platform.python_version(),
        "lightgbm": lgb.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "optbinning": optbinning.__version__,
        "seed": 42,
        "folds": 5
    }
    
    out = {
        "behaviour_tests": tests,
        "all_passed": all(t["passed"] for t in tests),
        "ab_test": ab,
        "versions": versions
    }
    
    json.dump(out, open(os.path.join(REP, "model_checks.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
