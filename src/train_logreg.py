import json
import joblib
import os
import numpy as np
import pandas as pd
from optbinning import BinningProcess, Scorecard
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve

RAW = os.environ.get("HC_RAW", "/Volumes/SongSJ/data/home-credit")
FEAT = os.environ.get("HC_FEAT", "/Volumes/SongSJ/data/home-credit/feat")
REP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")
MODELS = os.environ.get("HC_MODELS", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"))

def div(a, b):
    return a / b.replace(0, np.nan)

def load_data():
    app = pd.read_parquet(os.path.join(FEAT, "app.parquet"))
    bureau = pd.read_parquet(os.path.join(FEAT, "bureau.parquet"))
    prev = pd.read_parquet(os.path.join(FEAT, "prev.parquet"))
    behav = pd.read_parquet(os.path.join(FEAT, "behav.parquet"))
    df = app.merge(bureau, on="SK_ID_CURR", how="left") \
             .merge(prev, on="SK_ID_CURR", how="left") \
             .merge(behav, on="SK_ID_CURR", how="left")
    return df

def ks_stat(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    return max(tpr - fpr)

def psi(expected, actual, bins=10):
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.clip(np.histogram(expected, edges)[0] / len(expected), 1e-6, None)
    a = np.clip(np.histogram(actual, edges)[0] / len(actual), 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))

CANDIDATES = [
    "APP_EXT_SOURCE_1","APP_EXT_SOURCE_2","APP_EXT_SOURCE_3","APP_EXT_MEAN","APP_EXT_MIN","APP_EXT_NULL_CNT",
    "APP_AGE_YEARS","APP_EMPLOYED_YEARS","APP_EMPLOYED_TO_AGE","APP_DAYS_EMPLOYED_ANOM",
    "APP_CREDIT_TO_INCOME","APP_ANNUITY_TO_INCOME","APP_CREDIT_TO_ANNUITY","APP_CREDIT_TO_GOODS","APP_GOODS_MINUS_CREDIT",
    "APP_AMT_INCOME_TOTAL","APP_INCOME_PER_PERSON","APP_CNT_CHILDREN","APP_CHILDREN_RATIO",
    "APP_DAYS_ID_PUBLISH","APP_DAYS_REGISTRATION","APP_DAYS_LAST_PHONE_CHANGE","APP_OWN_CAR_AGE",
    "APP_REGION_RATING_CLIENT_W_CITY","APP_REGION_MISMATCH_CNT","APP_DOC_CNT","APP_REQ_BUREAU_TOTAL","APP_SOCIAL_DEF_RATIO_30",
    "APP_FLAG_OWN_CAR","APP_FLAG_OWN_REALTY","APP_NAME_CONTRACT_TYPE","APP_NAME_EDUCATION_TYPE","APP_NAME_INCOME_TYPE",
    "APP_NAME_FAMILY_STATUS","APP_NAME_HOUSING_TYPE","APP_OCCUPATION_TYPE","APP_ORGANIZATION_TYPE",
    "BUR_CNT","BUR_ACTIVE_CNT","BUR_ACTIVE_MEAN","BUR_DAYS_CREDIT_MAX","BUR_DAYS_CREDIT_MEAN","BUR_TOTAL_DEBT_RATIO","BUR_ACT_DEBT_RATIO",
    "BUR_OVERDUE_CNT","BUR_BB_DPD_MAX","BUR_BB_LATE_MEAN","BUR_CREDIT_CARD_CNT",
    "PREV_CNT","PREV_APPROVED_MEAN","PREV_REFUSED_MEAN","PREV_RECENT_REFUSED_CNT","PREV_LAST_REFUSED","PREV_APP_CREDIT_RATIO_MEAN",
    "PREV_CNT_PAYMENT_MEAN","PREV_DAYS_DECISION_MAX","PREV_CASH_MEAN",
    "INS_DPD_MAX","INS_DPD_MEAN","INS_LATE_MEAN","INS_RECENT_LATE_RATE","INS_PAY_RATIO_MIN","INS_UNDERPAID_MEAN","INS_PAID_TO_DUE","INS_DBD_MEAN",
    "POS_LATE_MEAN","POS_REMAINING_INST_SUM","POS_COMPLETED_MEAN",
    "CC_UTIL_MEAN","CC_UTIL_MAX","CC_LATE_MEAN","CC_PAY_TO_MIN_MEAN","CC_ATM_SHARE_MEAN"
]

def signed_vars(W, y):
    """Drop WoE variables whose coefficient has the wrong (positive) sign, one at a time, until all agree."""
    cols = list(W.columns)
    while True:
        coef = LogisticRegression(C=0.1, max_iter=2000).fit(W[cols], y).coef_[0]
        if coef.max() <= 0:
            return cols
        cols.pop(int(coef.argmax()))

def make_bp(cat_cols, variables=None, select=True):
    variables = variables or CANDIDATES
    cat_cols = [c for c in cat_cols if c in variables]
    return BinningProcess(variable_names=variables, categorical_variables=cat_cols, max_n_prebins=20, min_prebin_size=0.02,
        binning_fit_params={v: {"monotonic_trend": "auto_asc_desc"} for v in variables if v not in cat_cols},
        selection_criteria={"iv": {"min": 0.02, "max": 1.0, "strategy": "highest", "top": 30}} if select else None, n_jobs=4)

def main():
    global CANDIDATES
    os.makedirs(REP, exist_ok=True)
    df = load_data()
    missing = [c for c in CANDIDATES if c not in df.columns]
    if missing:
        print(f"Missing candidates: {missing}")
        CANDIDATES = [c for c in CANDIDATES if c in df.columns]
    cat_cols = [c for c in CANDIDATES if str(df[c].dtype) in ("category", "object")]
    for c in cat_cols:
        df[c] = df[c].astype(str)
    train = df[df.IS_TRAIN == 1].reset_index(drop=True)
    test = df[df.IS_TRAIN == 0].reset_index(drop=True)
    X = train[CANDIDATES]
    y = train["TARGET"].astype(int).values
    Xt = test[CANDIDATES]
    fold_auc = []
    oof = np.zeros(len(train))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for tr, va in skf.split(X, y):
        bp = make_bp(cat_cols)
        bp.fit(X.iloc[tr], y[tr])
        Wtr = bp.transform(X.iloc[tr], metric="woe")
        Wva = bp.transform(X.iloc[va], metric="woe")
        cols = signed_vars(Wtr, y[tr])
        lr = LogisticRegression(C=0.1, max_iter=2000)
        lr.fit(Wtr[cols], y[tr])
        oof[va] = lr.predict_proba(Wva[cols])[:, 1]
        fold_auc.append(roc_auc_score(y[va], oof[va]))
        print(f"Fold AUC: {fold_auc[-1]:.4f}", flush=True)
    bp0 = make_bp(cat_cols)
    bp0.fit(X, y)
    bp0.summary()[["name", "dtype", "status", "selected", "n_bins", "iv", "gini"]].sort_values("iv", ascending=False).to_csv(os.path.join(REP, "logreg_iv_all.csv"), index=False)
    W = bp0.transform(X, metric="woe")
    kept = signed_vars(W, y)
    dropped = [c for c in W.columns if c not in kept]
    print("dropped (wrong sign):", dropped, flush=True)
    X, Xt = X[kept], Xt[kept]
    sc = Scorecard(binning_process=make_bp(cat_cols, kept, select=False), estimator=LogisticRegression(C=0.1, max_iter=2000), scaling_method="pdo_odds", scaling_method_params={"pdo": 20, "odds": 50, "scorecard_points": 600})
    sc.fit(X, y)
    tpred = sc.predict_proba(Xt)[:, 1]
    train_pred_full = sc.predict_proba(X)[:, 1]
    bp_final = sc.binning_process_
    selected = list(bp_final.get_support(names=True))
    summ = bp_final.summary()
    summ = summ[["name", "dtype", "status", "selected", "n_bins", "iv", "gini"]]
    summ = summ.sort_values("iv", ascending=False)
    summ.to_csv(os.path.join(REP, "logreg_iv.csv"), index=False)
    coef_df = pd.DataFrame({"feature": selected, "coef": sc.estimator_.coef_[0]})
    coef_df = coef_df.sort_values("coef")
    coef_df.to_csv(os.path.join(REP, "logreg_coef.csv"), index=False)
    sc.table(style="detailed").to_csv(os.path.join(REP, "scorecard.csv"), index=False)
    metrics = {
        "model": "Logistic Regression (WoE scorecard)",
        "n_candidates": len(CANDIDATES),
        "n_selected": len(selected),
        "selected": selected,
        "fold_auc": [float(a) for a in fold_auc],
        "auc_mean": float(np.mean(fold_auc)),
        "auc_std": float(np.std(fold_auc)),
        "oof_auc": float(roc_auc_score(y, oof)),
        "oof_gini": float(2 * roc_auc_score(y, oof) - 1),
        "oof_ks": float(ks_stat(y, oof)),
        "train_auc_full": float(roc_auc_score(y, train_pred_full)),
        "psi_train_vs_test": psi(oof, tpred),
        "dropped_wrong_sign": dropped,
        "n_positive_coef": int((sc.estimator_.coef_[0] > 0).sum())
    }
    dec = pd.qcut(oof, 10, labels=False, duplicates="drop")
    metrics["decile_bad_rate"] = pd.Series(y).groupby(dec).mean().round(4).tolist()
    scores = sc.score(Xt)
    metrics["score_min"], metrics["score_median"], metrics["score_max"] = float(scores.min()), float(np.median(scores)), float(scores.max())
    metrics_json = json.dumps(metrics, indent=2)
    with open(os.path.join(REP, "logreg_metrics.json"), "w") as f:
        f.write(metrics_json)
    oof_df = pd.DataFrame({
        "SK_ID_CURR": train["SK_ID_CURR"],
        "TARGET": train["TARGET"],
        "PRED": oof
    })
    oof_df.to_parquet(os.path.join(FEAT, "logreg_oof.parquet"))
    test_df = pd.DataFrame({
        "SK_ID_CURR": test["SK_ID_CURR"],
        "PRED": tpred,
        "SCORE": scores
    })
    test_df.to_parquet(os.path.join(FEAT, "logreg_test.parquet"))
    submission_df = pd.DataFrame({
        "SK_ID_CURR": test["SK_ID_CURR"],
        "TARGET": tpred
    })
    submission_df.to_csv(os.path.join(REP, "submission_logreg.csv"), index=False)
    os.makedirs(MODELS, exist_ok=True)
    joblib.dump(sc, os.path.join(MODELS, "scorecard.pkl"))
    json.dump({"features": kept, "cat_cols": [c for c in cat_cols if c in kept]}, open(os.path.join(MODELS, "scorecard_features.json"), "w"), indent=2)
    print(f"Final metrics: {metrics_json}")
    print(f"Final oof shape: {oof_df.shape}")
    print(f"Final test shape: {test_df.shape}")
    print(f"Final submission shape: {submission_df.shape}")

if __name__ == "__main__":
    main()
