import json
import os
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, roc_curve

RAW = os.environ.get("HC_RAW", "/Volumes/SongSJ/data/home-credit")
FEAT = os.environ.get("HC_FEAT", "/Volumes/SongSJ/data/home-credit/feat")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REP = os.environ.get("HC_REP", os.path.join(ROOT, "reports"))
PRED = os.environ.get("HC_PRED", FEAT)
MODELS = os.environ.get("HC_MODELS", os.path.join(ROOT, "models"))
NO_EXT = os.environ.get("HC_NO_EXT", "0") == "1"
SMOKE = int(os.environ.get("HC_SMOKE", "0"))
TAG = "lgbm_noext" if NO_EXT else "lgbm"

def downcast_floats(df):
    for c in df.select_dtypes(include=['float64']).columns:
        df[c] = pd.to_numeric(df[c], downcast='float')
    return df

def load_data():
    df = pd.read_parquet(FEAT + "/app.parquet")
    for name in ["bureau", "prev", "behav"]:
        df = df.merge(pd.read_parquet(os.path.join(FEAT, name + ".parquet")), on="SK_ID_CURR", how="left")
    return df

DROP = ["SK_ID_CURR", "TARGET", "IS_TRAIN", "APP_CODE_GENDER"]   # gender excluded on purpose (fair lending)

def ks_stat(y, p):
    fpr, tpr, _ = roc_curve(y, p); return float(np.max(tpr - fpr))

def psi(expected, actual, bins=10):
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e = np.clip(e, 1e-6, None); a = np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))

def main():
    df = load_data()
    df = downcast_floats(df)
    train = df[df.IS_TRAIN == 1].reset_index(drop=True)
    test = df[df.IS_TRAIN == 0].reset_index(drop=True)
    if SMOKE > 0:
        train = train.sample(n=SMOKE, random_state=42).reset_index(drop=True)
        test = test.head(5000).reset_index(drop=True)
    feats = [c for c in df.columns if c not in DROP]
    if NO_EXT:
        feats = [c for c in feats if "EXT_" not in c]
    X = train[feats]
    y = train["TARGET"].astype(int)
    Xt = test[feats]
    
    os.makedirs(REP, exist_ok=True)
    os.makedirs(PRED, exist_ok=True)
    os.makedirs(MODELS, exist_ok=True)
    
    prev_path = os.path.join(REP, TAG + "_metrics.json")
    previous_oof_auc = json.load(open(prev_path))["oof_auc"] if os.path.exists(prev_path) else None
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(train))
    tpred = np.zeros(len(test))
    imp = np.zeros(len(feats))
    fold_auc = []
    best_iters = []
    
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        fit_idx, es_idx = train_test_split(tr, test_size=0.1, stratify=y.iloc[tr], random_state=42 + fold)
        model = lgb.LGBMClassifier(n_estimators=5000, learning_rate=0.03, num_leaves=31, min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0, n_jobs=8, random_state=42, verbose=-1)
        model.fit(X.iloc[fit_idx], y.iloc[fit_idx], eval_set=[(X.iloc[es_idx], y.iloc[es_idx])], eval_metric="auc", callbacks=[lgb.early_stopping(100, verbose=False)])
        model.booster_.save_model(os.path.join(MODELS, f"{TAG}_fold{fold}.txt"))
        oof[va] = model.predict_proba(X.iloc[va])[:, 1]
        tpred += model.predict_proba(Xt)[:, 1] / 5
        imp += model.booster_.feature_importance(importance_type="gain") / 5
        fold_auc.append(float(roc_auc_score(y.iloc[va], oof[va])))
        best_iters.append(int(model.best_iteration_))
        print(f"fold {fold} auc {fold_auc[-1]:.4f} iters {best_iters[-1]}", flush=True)
    
    train_auc_last = roc_auc_score(y.iloc[fit_idx], model.predict_proba(X.iloc[fit_idx])[:, 1])
    
    oof_auc = roc_auc_score(y, oof)
    metrics = {
        "model": "LightGBM (no external scores)" if NO_EXT else "LightGBM",
        "tag": TAG,
        "n_features": len(feats),
        "fold_auc": fold_auc,
        "auc_mean": float(np.mean(fold_auc)),
        "auc_std": float(np.std(fold_auc)),
        "oof_auc": float(oof_auc),
        "oof_gini": float(2 * oof_auc - 1),
        "oof_ks": ks_stat(y, oof),
        "train_auc_last_fold": float(train_auc_last),
        "best_iters": best_iters,
        "psi_train_vs_test": psi(oof, tpred),
        "default_rate": float(y.mean()),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "early_stopping": "inner 10% split of each training fold",
        "previous_oof_auc": previous_oof_auc
    }
    
    dec = pd.qcut(oof, 10, labels=False, duplicates="drop")
    table = pd.DataFrame({"decile": dec, "y": y}).groupby("decile")["y"].agg(["count", "mean"]).reset_index()
    metrics["decile_bad_rate"] = table["mean"].round(4).tolist()
    
    json.dump(metrics, open(os.path.join(REP, f"{TAG}_metrics.json"), "w"), indent=2)
    pd.DataFrame({"feature": feats, "gain": imp}).sort_values("gain", ascending=False).to_csv(os.path.join(REP, f"{TAG}_importance.csv"), index=False)
    pd.DataFrame({"SK_ID_CURR": train.SK_ID_CURR, "TARGET": y, "PRED": oof}).to_parquet(os.path.join(PRED, f"{TAG}_oof.parquet"), index=False)
    pd.DataFrame({"SK_ID_CURR": test.SK_ID_CURR, "PRED": tpred}).to_parquet(os.path.join(PRED, f"{TAG}_test.parquet"), index=False)
    json.dump(feats, open(os.path.join(MODELS, f"{TAG}_features.json"), "w"))
    
    if TAG == "lgbm" and SMOKE == 0:
        pd.DataFrame({"SK_ID_CURR": test.SK_ID_CURR, "TARGET": tpred}).to_csv(os.path.join(REP, "submission_lgbm.csv"), index=False)
    
    print(metrics)

if __name__ == "__main__":
    main()
