import os
import sys
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT = os.environ.get("HC_FEAT", "/Volumes/SongSJ/data/home-credit/feat")
PRED = os.environ.get("HC_PRED", FEAT)
REP = os.environ.get("HC_REP", os.path.join(ROOT, "reports"))
MODELS = os.environ.get("HC_MODELS", os.path.join(ROOT, "models"))

os.makedirs(REP, exist_ok=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_lgbm import load_data, downcast_floats

def main():
    SMOKE = int(os.environ.get("HC_SMOKE", "0"))
    df = downcast_floats(load_data())
    test = df[df.IS_TRAIN == 0].reset_index(drop=True)
    if SMOKE > 0:
        test = test.head(5000).reset_index(drop=True)

    feats = json.load(open(os.path.join(MODELS, "lgbm_features.json")))
    boosters = [lgb.Booster(model_file=os.path.join(MODELS, f"lgbm_fold{k}.txt")) for k in range(5)]
    p_lgbm = np.mean([b.predict(test[feats]) for b in boosters], axis=0)
    ref = pd.read_parquet(os.path.join(PRED, "lgbm_test.parquet"))
    assert (ref.SK_ID_CURR.values == test.SK_ID_CURR.values).all()
    lgbm_diff = float(np.max(np.abs(p_lgbm - ref.PRED.values)))

    p_sc = None
    points = None
    sc_diff = None
    points_diff = None
    if os.path.exists(os.path.join(MODELS, "scorecard.pkl")):
        sc = joblib.load(os.path.join(MODELS, "scorecard.pkl"))
        meta = json.load(open(os.path.join(MODELS, "scorecard_features.json")))
        Xs = test[meta["features"]].copy()
        for c in meta["cat_cols"]:
            Xs[c] = Xs[c].astype(str)
        p_sc = sc.predict_proba(Xs)[:, 1]
        points = sc.score(Xs)
        ref2 = pd.read_parquet(os.path.join(PRED, "logreg_test.parquet"))
        sc_diff = float(np.max(np.abs(p_sc - ref2.PRED.values)))
        points_diff = float(np.max(np.abs(points - ref2.SCORE.values)))

    scored = test[["SK_ID_CURR"]]
    scored["PD_LGBM"] = p_lgbm
    if p_sc is not None:
        scored["PD_SCORECARD"] = p_sc
        scored["SCORE_POINTS"] = points
    scored.to_csv(os.path.join(REP, "scored_test.csv"), index=False)

    score_check = {
        "n_test": len(test),
        "lgbm_max_abs_diff": lgbm_diff,
        "scorecard_max_abs_diff": sc_diff,
        "points_max_abs_diff": points_diff,
        "match": bool(lgbm_diff < 1e-6 and (sc_diff is None or sc_diff < 1e-6))
    }

    with open(os.path.join(REP, "score_check.json"), "w") as f:
        json.dump(score_check, f, default=float, indent=2)

    print(json.dumps(score_check, default=float, indent=2))
    if not score_check["match"]:
        raise SystemExit("saved models do not reproduce stored predictions")

if __name__ == "__main__":
    main()
