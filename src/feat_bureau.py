import os
import pandas as pd
import numpy as np

def div(a, b):
    return a / b.replace(0, np.nan)

def main():
    RAW = os.environ.get("HC_RAW", "/Volumes/SongSJ/data/home-credit")
    FEAT = os.environ.get("HC_FEAT", "/Volumes/SongSJ/data/home-credit/feat")
    os.makedirs(FEAT, exist_ok=True)
    
    bb = pd.read_csv(os.path.join(RAW, "bureau_balance.csv"))
    bb = bb[["SK_ID_BUREAU", "MONTHS_BALANCE", "STATUS"]]
    bb["BB_DPD"] = pd.to_numeric(bb["STATUS"], errors="coerce")
    bb["BB_LATE"] = (bb["BB_DPD"] >= 1).astype(int)
    bb_agg = bb.groupby("SK_ID_BUREAU").agg(
        BB_MONTHS=("MONTHS_BALANCE", "count"),
        BB_DPD_MAX=("BB_DPD", "max"),
        BB_LATE_MEAN=("BB_LATE", "mean")
    ).reset_index()
    
    b = pd.read_csv(os.path.join(RAW, "bureau.csv"))
    # cleaning: impossible values
    for c in ["DAYS_CREDIT_ENDDATE", "DAYS_ENDDATE_FACT", "DAYS_CREDIT_UPDATE"]:
        bad = b[c].abs() > 36500
        print(f"{c}: {int(bad.sum())} values beyond 100 years -> NaN")
        b.loc[bad, c] = np.nan
    neg = b["AMT_CREDIT_SUM_DEBT"] < 0
    print(f"AMT_CREDIT_SUM_DEBT: {int(neg.sum())} negative -> 0")
    b.loc[neg, "AMT_CREDIT_SUM_DEBT"] = 0
    neg = b["AMT_CREDIT_SUM_LIMIT"] < 0
    print(f"AMT_CREDIT_SUM_LIMIT: {int(neg.sum())} negative -> NaN")
    b.loc[neg, "AMT_CREDIT_SUM_LIMIT"] = np.nan
    
    b = b.merge(bb_agg, on="SK_ID_BUREAU", how="left")
    
    b["IS_ACTIVE"] = (b["CREDIT_ACTIVE"] == "Active").astype(int)
    b["IS_BAD_DEBT"] = (b["CREDIT_ACTIVE"] == "Bad debt").astype(int)
    b["HAS_OVERDUE"] = (b["AMT_CREDIT_SUM_OVERDUE"].fillna(0) > 0).astype(int)
    b["DEBT_RATIO"] = div(b["AMT_CREDIT_SUM_DEBT"], b["AMT_CREDIT_SUM"])
    b["IS_CREDIT_CARD"] = (b["CREDIT_TYPE"] == "Credit card").astype(int)
    
    agg = b.groupby("SK_ID_CURR").agg(
        CNT=("SK_ID_BUREAU", "count"),
        ACTIVE_CNT=("IS_ACTIVE", "sum"),
        ACTIVE_MEAN=("IS_ACTIVE", "mean"),
        BAD_DEBT_CNT=("IS_BAD_DEBT", "sum"),
        OVERDUE_CNT=("HAS_OVERDUE", "sum"),
        DAYS_CREDIT_MIN=("DAYS_CREDIT", "min"),
        DAYS_CREDIT_MAX=("DAYS_CREDIT", "max"),
        DAYS_CREDIT_MEAN=("DAYS_CREDIT", "mean"),
        DAYS_ENDDATE_MAX=("DAYS_CREDIT_ENDDATE", "max"),
        CREDIT_DAY_OVERDUE_MAX=("CREDIT_DAY_OVERDUE", "max"),
        AMT_CREDIT_SUM=("AMT_CREDIT_SUM", "sum"),
        AMT_DEBT_SUM=("AMT_CREDIT_SUM_DEBT", "sum"),
        AMT_OVERDUE_SUM=("AMT_CREDIT_SUM_OVERDUE", "sum"),
        AMT_MAX_OVERDUE_MAX=("AMT_CREDIT_MAX_OVERDUE", "max"),
        DEBT_RATIO_MEAN=("DEBT_RATIO", "mean"),
        DEBT_RATIO_MAX=("DEBT_RATIO", "max"),
        PROLONG_SUM=("CNT_CREDIT_PROLONG", "sum"),
        CREDIT_CARD_CNT=("IS_CREDIT_CARD", "sum"),
        BB_DPD_MAX=("BB_DPD_MAX", "max"),
        BB_LATE_MEAN=("BB_LATE_MEAN", "mean"),
        BB_MONTHS_MEAN=("BB_MONTHS", "mean")
    ).reset_index()
    
    a = b[b["IS_ACTIVE"] == 1].groupby("SK_ID_CURR").agg(
        ACT_DEBT_SUM=("AMT_CREDIT_SUM_DEBT", "sum"),
        ACT_CREDIT_SUM=("AMT_CREDIT_SUM", "sum")
    ).reset_index()
    agg = pd.merge(agg, a, on="SK_ID_CURR", how="left")
    agg["ACT_DEBT_RATIO"] = div(agg["ACT_DEBT_SUM"], agg["ACT_CREDIT_SUM"])
    
    agg["TOTAL_DEBT_RATIO"] = div(agg["AMT_DEBT_SUM"], agg["AMT_CREDIT_SUM"])
    
    for col in agg.columns:
        if col != "SK_ID_CURR":
            agg.rename(columns={col: "BUR_" + col}, inplace=True)
    
    agg.replace([np.inf, -np.inf], np.nan, inplace=True)
    for col in agg.select_dtypes(include=np.number).columns:
        if agg[col].dtype == np.float64:
            agg[col] = agg[col].astype(np.float32)
    
    agg.to_parquet(os.path.join(FEAT, "bureau.parquet"), index=False)
    print(agg.shape)

if __name__ == "__main__":
    main()
