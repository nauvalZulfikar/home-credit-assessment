import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT = os.environ.get("HC_FEAT", "/Volumes/SongSJ/data/home-credit/feat")
PRED = os.environ.get("HC_PRED", FEAT)
REP = os.environ.get("HC_REP", os.path.join(ROOT, "reports"))
os.makedirs(REP, exist_ok=True)

def psi(expected, actual, bins=10):
    expected = np.asarray(expected, float); actual = np.asarray(actual, float)
    expected = expected[~np.isnan(expected)]; actual = actual[~np.isnan(actual)]
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.clip(np.histogram(expected, edges)[0] / len(expected), 1e-6, None)
    a = np.clip(np.histogram(actual, edges)[0] / len(actual), 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))

def psi_cat(expected, actual):
    e = expected.astype(str).value_counts(normalize=True); a = actual.astype(str).value_counts(normalize=True)
    idx = e.index.union(a.index)
    e = np.clip(e.reindex(idx).fillna(0).values, 1e-6, None); a = np.clip(a.reindex(idx).fillna(0).values, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))

def status(v):
    return "ok" if v < 0.1 else ("watch" if v < 0.25 else "act")

def py(v):
    return v.item() if isinstance(v, np.generic) else v

def main():
    rd = lambda n: pd.read_parquet(os.path.join(PRED, n))
    app = pd.read_parquet(os.path.join(FEAT, "app.parquet"))
    lo, so, lt, st = rd("lgbm_oof.parquet"), rd("logreg_oof.parquet"), rd("lgbm_test.parquet"), rd("logreg_test.parquet")

    d = lo.rename(columns={"PRED": "LGBM"}).merge(so[["SK_ID_CURR", "PRED"]].rename(columns={"PRED": "SCORECARD"}), on="SK_ID_CURR")
    d = d.merge(app[["SK_ID_CURR", "APP_AMT_CREDIT"]], on="SK_ID_CURR")
    y = d.TARGET.astype(int).values; credit = d.APP_AMT_CREDIT.values
    rows = []
    for model, col in [("LightGBM", "LGBM"), ("Scorecard (WoE logistic)", "SCORECARD")]:
        order = np.argsort(-d[col].values, kind="stable")
        for r in np.round(np.arange(0.05, 0.501, 0.05), 2):
            n_rej = int(round(r * len(d))); rej = np.zeros(len(d), bool); rej[order[:n_rej]] = True
            bad, good = rej & (y == 1), rej & (y == 0)
            cb, cg = float(credit[bad].sum()), float(credit[good].sum())
            rows.append({"model": model, "reject_pct": float(r), "n_rejected": n_rej, "defaults_avoided": int(bad.sum()),
                         "good_clients_rejected": int(good.sum()), "credit_defaults_avoided": cb, "credit_good_rejected": cg,
                         "break_even_ratio": round(cg / cb, 3)})
    val = pd.DataFrame(rows)
    val.to_csv(os.path.join(REP, "value.csv"), index=False)

    g = val[val.model == "LightGBM"]
    fig, ax = plt.subplots(figsize=(9, 5), dpi=120)
    ax.plot(g.reject_pct * 100, g.credit_defaults_avoided / 1e9, marker="o", label="Credit of defaulters turned away")
    ax.plot(g.reject_pct * 100, g.credit_good_rejected / 1e9, marker="o", label="Credit of good clients turned away")
    ax.set_xlabel("Clients rejected (%)"); ax.set_ylabel("Credit amount (billions, data currency)")
    ax2 = ax.twinx()
    ax2.plot(g.reject_pct * 100, g.break_even_ratio, "--", color="grey", label="Good credit lost per 1 unit of bad credit avoided")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left"); fig.tight_layout()
    fig.savefig(os.path.join(REP, "fig_value.png")); plt.close(fig)

    df = app
    for t in ["bureau", "prev", "behav"]:
        df = df.merge(pd.read_parquet(os.path.join(FEAT, f"{t}.parquet")), on="SK_ID_CURR", how="left")
    top = pd.read_csv(os.path.join(REP, "lgbm_importance.csv")).feature.head(10)
    rows = []
    for f in top:
        tr, te = df.loc[df.IS_TRAIN == 1, f], df.loc[df.IS_TRAIN == 0, f]
        v = psi_cat(tr, te) if isinstance(tr.dtype, pd.CategoricalDtype) or tr.dtype == object else psi(tr, te)
        rows.append({"feature": f, "psi": round(v, 4), "status": status(v)})
    feats = pd.DataFrame(rows)
    sp_l, sp_s = psi(lo.PRED, lt.PRED), psi(so.PRED, st.PRED)
    fp = pd.concat([feats, pd.DataFrame([{"feature": "Model score (LightGBM)", "psi": round(sp_l, 4), "status": status(sp_l)},
                                         {"feature": "Model score (Scorecard)", "psi": round(sp_s, 4), "status": status(sp_s)}])])
    fp.to_csv(os.path.join(REP, "feature_psi.csv"), index=False)

    at = lambda m: {k: py(v) for k, v in val[(val.model == m) & (val.reject_pct == 0.10)].iloc[0].items()}
    summary = {"lgbm_at_10": at("LightGBM"), "scorecard_at_10": at("Scorecard (WoE logistic)"),
               "break_even_ratio_range_lgbm": [float(g.break_even_ratio.min()), float(g.break_even_ratio.max())],
               "feature_psi_max": float(feats.psi.max()), "feature_psi_max_name": feats.loc[feats.psi.idxmax(), "feature"],
               "n_features_watch_or_act": int((feats.status != "ok").sum()), "score_psi_lgbm": sp_l, "score_psi_scorecard": sp_s}
    json.dump(summary, open(os.path.join(REP, "value_summary.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
