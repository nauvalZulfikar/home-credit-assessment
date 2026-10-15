import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT = os.environ.get("HC_FEAT", "/Volumes/SongSJ/data/home-credit/feat")
PRED = os.environ.get("HC_PRED", FEAT)
REP = os.environ.get("HC_REP", os.path.join(ROOT, "reports"))
os.makedirs(REP, exist_ok=True)

def ks(y, s):
    fpr, tpr, _ = roc_curve(y, s); return float(np.max(tpr - fpr))

def at_reject(y, s, r):
    # reject the riskiest share r of clients
    y = np.asarray(y); n_rej = int(round(r * len(y)))
    rej = np.zeros(len(y), bool); rej[np.argsort(-np.asarray(s), kind="stable")[:n_rej]] = True
    return {"caught_pct": float(y[rej].sum() / y.sum()), "approved_bad_rate": float(y[~rej].mean()), "good_rejected_pct": float(((y == 0) & rej).sum() / (y == 0).sum())}

def safe_auc(y, s):
    y = np.asarray(y)
    return float(roc_auc_score(y, s)) if 10 <= y.sum() < len(y) else float("nan")

def bootstrap_auc(y, scores, n=200, seed=42):
    # 95% range of each AUC and of the LightGBM - scorecard difference, resampling clients
    rng = np.random.default_rng(seed); out = {k: [] for k in scores}
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        for k, s in scores.items():
            out[k].append(roc_auc_score(y[i], s[i]))
    out["diff"] = np.array(out["LightGBM"]) - np.array(out["Scorecard (WoE logistic)"])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in out.items()}

def main():
    app = pd.read_parquet(os.path.join(FEAT, "app.parquet"), columns=["SK_ID_CURR", "IS_TRAIN", "APP_EXT_SOURCE_1", "APP_EXT_SOURCE_2", "APP_EXT_SOURCE_3", "APP_ANNUITY_TO_INCOME", "APP_NAME_CONTRACT_TYPE", "APP_CODE_GENDER", "APP_AGE_YEARS", "APP_NAME_INCOME_TYPE", "APP_DAYS_EMPLOYED_ANOM"])
    bur_ids = set(pd.read_parquet(os.path.join(FEAT, "bureau.parquet"), columns=["SK_ID_CURR"]).SK_ID_CURR)
    prev_ids = set(pd.read_parquet(os.path.join(FEAT, "prev.parquet"), columns=["SK_ID_CURR"]).SK_ID_CURR)

    mix = pd.crosstab(app.APP_NAME_CONTRACT_TYPE.astype(str), app.IS_TRAIN, normalize="columns")
    mix = pd.DataFrame({"contract_type": mix.index, "train_share": mix[1].values, "test_share": mix[0].values}).round(4)
    mix.to_csv(os.path.join(REP, "test_mix.csv"), index=False)

    d = pd.read_parquet(os.path.join(PRED, "lgbm_oof.parquet")).rename(columns={"PRED": "LGBM"})
    d = d.merge(pd.read_parquet(os.path.join(PRED, "logreg_oof.parquet"))[["SK_ID_CURR", "PRED"]].rename(columns={"PRED": "SCORECARD"}), on="SK_ID_CURR")
    noext = os.path.join(PRED, "lgbm_noext_oof.parquet")
    if os.path.exists(noext):
        d = d.merge(pd.read_parquet(noext)[["SK_ID_CURR", "PRED"]].rename(columns={"PRED": "LGBM_NOEXT"}), on="SK_ID_CURR")
    d = d.merge(app, on="SK_ID_CURR")
    y = d.TARGET.astype(int).values

    ext = d[["APP_EXT_SOURCE_1", "APP_EXT_SOURCE_2", "APP_EXT_SOURCE_3"]]
    raw = {"B1 EXT_SOURCE_2 only": -d.APP_EXT_SOURCE_2, "B2 average of EXT_SOURCE_1-3": -ext.mean(axis=1, skipna=True),
           "B3 annuity-to-income rule": d.APP_ANNUITY_TO_INCOME, "Scorecard (WoE logistic)": d.SCORECARD, "LightGBM": d.LGBM}
    if "LGBM_NOEXT" in d:
        raw["LightGBM without external scores"] = d.LGBM_NOEXT
    models = {k: v.fillna(v.median()).values for k, v in raw.items()}

    rows = [{"model": "B0 approve everyone", "auc": 0.5, "gini": 0.0, "ks": 0.0, "caught_pct": 0.0, "approved_bad_rate": float(y.mean()), "good_rejected_pct": 0.0}]
    for k, s in models.items():
        auc = roc_auc_score(y, s)
        rows.append({"model": k, "auc": auc, "gini": 2 * auc - 1, "ks": ks(y, s), **at_reject(y, s, 0.10)})
    base = pd.DataFrame(rows).round(4)
    base.to_csv(os.path.join(REP, "baselines.csv"), index=False)

    curve = [{"model": k, "reject_pct": r, **{x: v for x, v in at_reject(y, models[k], r).items() if x != "good_rejected_pct"}}
             for k in ["B2 average of EXT_SOURCE_1-3", "Scorecard (WoE logistic)", "LightGBM"] for r in np.round(np.arange(0.05, 0.501, 0.05), 2)]
    curve = pd.DataFrame(curve)
    curve.to_csv(os.path.join(REP, "reject_curve.csv"), index=False)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    for k, g in curve.groupby("model", sort=False):
        ax.plot(g.reject_pct * 100, g.caught_pct * 100, marker="o", label=k)
    ax.plot([0, 50], [0, 50], "--", color="grey", label="random")
    ax.set_xlabel("Clients rejected (%)"); ax.set_ylabel("Defaults caught (%)"); ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(REP, "fig_reject_curve.png")); plt.close(fig)

    inc = d.APP_NAME_INCOME_TYPE.astype(str); top4 = inc.value_counts().index[:4]
    segs = {"Contract type": d.APP_NAME_CONTRACT_TYPE.astype(str), "Gender": d.APP_CODE_GENDER.astype(str),
            "Age band": pd.cut(d.APP_AGE_YEARS, [0, 30, 45, 60, 200], right=False, labels=["<30", "30-45", "45-60", "60+"]).astype(str),
            "Income type": inc.where(inc.isin(top4), "Other"),
            "Has bureau record": pd.Series(np.where(d.SK_ID_CURR.isin(bur_ids), "Yes", "No")),
            "Employment code 365243": pd.Series(np.where(d.APP_DAYS_EMPLOYED_ANOM == 1, "Yes", "No")),
            "First-time client": pd.Series(np.where(d.SK_ID_CURR.isin(prev_ids), "No", "Yes"))}
    overall = roc_auc_score(y, d.LGBM)
    rows = []
    for seg, g in segs.items():
        for grp in sorted(g.unique()):
            m = (g == grp).values; yy = y[m]
            rows.append({"segment": seg, "group": grp, "n": int(m.sum()), "n_bad": int(yy.sum()), "default_rate": yy.mean(),
                         "auc_lgbm": safe_auc(yy, d.LGBM.values[m]), "auc_scorecard": safe_auc(yy, d.SCORECARD.values[m]),
                         "mean_pred_lgbm": d.LGBM.values[m].mean()})
    seg = pd.DataFrame(rows)
    seg["small"] = (seg.n < 1000) | (seg.n_bad < 100)
    seg["auc_drop_lgbm"] = overall - seg.auc_lgbm
    seg["flag_drop"] = seg.auc_drop_lgbm > 0.03
    seg = seg.round(4)
    seg.to_csv(os.path.join(REP, "segments.csv"), index=False)

    rows = []
    for name, col in [("LightGBM", "LGBM"), ("Scorecard (WoE logistic)", "SCORECARD")]:
        rej = (d[col] > np.quantile(d[col], 0.90)).values
        for s in ["Gender", "Age band"]:
            for grp in sorted(segs[s].unique()):
                m = (segs[s] == grp).values
                rows.append({"model": name, "segment": s, "group": grp, "n": int(m.sum()), "default_rate": y[m].mean(),
                             "rejection_rate": rej[m].mean(), "good_rejected_pct": rej[m & (y == 0)].mean(), "defaults_caught_pct": rej[m & (y == 1)].mean()})
    fair = pd.DataFrame(rows).round(4)
    fair.to_csv(os.path.join(REP, "fairness.csv"), index=False)

    rows = []
    for name, col in [("LightGBM", "LGBM"), ("Scorecard (WoE logistic)", "SCORECARD")]:
        dec = pd.qcut(d[col], 10, labels=False, duplicates="drop") + 1
        t = pd.DataFrame({"decile": dec, "p": d[col], "y": y}).groupby("decile").agg(mean_pred=("p", "mean"), actual_rate=("y", "mean"), n=("y", "size")).reset_index()
        t.insert(0, "model", name); rows.append(t)
    cal = pd.concat(rows).round(4)
    cal.to_csv(os.path.join(REP, "calibration.csv"), index=False)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
    for k, g in cal.groupby("model", sort=False):
        ax.plot(g.mean_pred, g.actual_rate, marker="o", label=k)
    hi = float(cal[["mean_pred", "actual_rate"]].max().max()) * 1.05
    ax.plot([0, hi], [0, hi], "--", color="grey")
    ax.set_xlabel("Predicted default rate"); ax.set_ylabel("Actual default rate"); ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(REP, "fig_calibration.png")); plt.close(fig)

    ci = bootstrap_auc(y, {"LightGBM": models["LightGBM"], "Scorecard (WoE logistic)": models["Scorecard (WoE logistic)"]})
    auc = dict(zip(base.model, base.auc.astype(float)))
    bl = base[base.model.str.match(r"B\d")].sort_values("auc").iloc[-1]
    row = lambda s, g, c: float(seg[(seg.segment == s) & (seg.group == g)][c].iloc[0])
    gfair = fair[(fair.model == "LightGBM") & (fair.segment == "Gender")]
    summary = {
        "default_rate": float(y.mean()), "n_train": int(len(y)), "auc": auc,
        "auc_ci95": ci,
        "best_baseline": bl.model, "best_baseline_auc": float(bl.auc),
        "models_beat_baseline": bool(auc["LightGBM"] > bl.auc and auc["Scorecard (WoE logistic)"] > bl.auc),
        "at_10": {k: at_reject(y, models[k], 0.10) for k in ["Scorecard (WoE logistic)", "LightGBM", "LightGBM without external scores"] if k in models},
        "train_cash_share": float(mix[mix.contract_type == "Cash loans"].train_share.iloc[0]),
        "test_cash_share": float(mix[mix.contract_type == "Cash loans"].test_share.iloc[0]),
        "lgbm_auc_cash": row("Contract type", "Cash loans", "auc_lgbm"), "lgbm_auc_revolving": row("Contract type", "Revolving loans", "auc_lgbm"),
        "flagged_segments": [f"{r.segment}: {r.group}" for r in seg.itertuples() if r.flag_drop and not r.small],
        "small_segments": [f"{r.segment}: {r.group}" for r in seg.itertuples() if r.small],
        "fairness_max_gap_lgbm": float(gfair.rejection_rate.max() - gfair.rejection_rate.min()),
        "default_rate_gap_gender": float(gfair.default_rate.max() - gfair.default_rate.min()),
    }
    json.dump(summary, open(os.path.join(REP, "evaluate_summary.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
