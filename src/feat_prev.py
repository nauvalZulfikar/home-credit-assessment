import os
import numpy as np
import pandas as pd

def div(a, b):
    return a / b.replace(0, np.nan)

def main():
    RAW = os.environ.get("HC_RAW", "/Volumes/SongSJ/data/home-credit")
    FEAT = os.environ.get("HC_FEAT", "/Volumes/SongSJ/data/home-credit/feat")
    os.makedirs(FEAT, exist_ok=True)
    
    INPUT = os.path.join(RAW, "previous_application.csv")
    OUTPUT = os.path.join(FEAT, "prev.parquet")
    
    p = pd.read_csv(INPUT, low_memory=False)
    
    cols = ["DAYS_FIRST_DRAWING", "DAYS_FIRST_DUE", "DAYS_LAST_DUE_1ST_VERSION", "DAYS_LAST_DUE", "DAYS_TERMINATION"]
    for col in cols:
        p[col] = p[col].replace(365243, np.nan)
    
    p["APPROVED"] = (p["NAME_CONTRACT_STATUS"] == "Approved").astype(int)
    p["REFUSED"] = (p["NAME_CONTRACT_STATUS"] == "Refused").astype(int)
    p["APP_CREDIT_RATIO"] = div(p["AMT_APPLICATION"], p["AMT_CREDIT"])
    p["CREDIT_GOODS_RATIO"] = div(p["AMT_CREDIT"], p["AMT_GOODS_PRICE"])
    p["DOWN_PAY_RATIO"] = div(p["AMT_DOWN_PAYMENT"], p["AMT_CREDIT"])
    p["IS_CASH"] = (p["NAME_CONTRACT_TYPE"] == "Cash loans").astype(int)
    p["DAYS_DECISION"] = p["DAYS_DECISION"].replace([365243], np.nan)
    p["IS_RECENT"] = (p["DAYS_DECISION"] >= -365).astype(int)
    p["RECENT_REFUSED"] = p["REFUSED"] * p["IS_RECENT"]
    p["TERM_DIFF"] = p["DAYS_LAST_DUE_1ST_VERSION"] - p["DAYS_LAST_DUE"]
    
    agg = p.groupby("SK_ID_CURR").agg(
        CNT=("SK_ID_PREV","count"), APPROVED_CNT=("APPROVED","sum"), APPROVED_MEAN=("APPROVED","mean"),
        REFUSED_CNT=("REFUSED","sum"), REFUSED_MEAN=("REFUSED","mean"), RECENT_REFUSED_CNT=("RECENT_REFUSED","sum"),
        CASH_MEAN=("IS_CASH","mean"),
        AMT_CREDIT_MEAN=("AMT_CREDIT","mean"), AMT_CREDIT_MAX=("AMT_CREDIT","max"),
        AMT_ANNUITY_MEAN=("AMT_ANNUITY","mean"),
        APP_CREDIT_RATIO_MEAN=("APP_CREDIT_RATIO","mean"), APP_CREDIT_RATIO_MIN=("APP_CREDIT_RATIO","min"),
        CREDIT_GOODS_RATIO_MEAN=("CREDIT_GOODS_RATIO","mean"), DOWN_PAY_RATIO_MEAN=("DOWN_PAY_RATIO","mean"),
        RATE_DOWN_PAYMENT_MEAN=("RATE_DOWN_PAYMENT","mean"),
        CNT_PAYMENT_MEAN=("CNT_PAYMENT","mean"),
        DAYS_DECISION_MAX=("DAYS_DECISION","max"), DAYS_DECISION_MEAN=("DAYS_DECISION","mean"),
        TERM_DIFF_MEAN=("TERM_DIFF","mean")
    ).reset_index()
    
    latest = p.sort_values(["DAYS_DECISION", "SK_ID_PREV"], ascending=False, kind="stable").drop_duplicates("SK_ID_CURR", keep="first")
    latest = latest[["SK_ID_CURR", "REFUSED", "AMT_CREDIT", "DAYS_DECISION"]]
    latest.columns = ["SK_ID_CURR", "LAST_REFUSED", "LAST_AMT_CREDIT", "LAST_DAYS_DECISION"]
    agg = pd.merge(agg, latest, on="SK_ID_CURR", how="left")
    
    for col in agg.columns:
        if col != "SK_ID_CURR":
            agg[col] = agg[col].replace([np.inf, -np.inf], np.nan)
    
    for col in agg.select_dtypes(include=np.number).columns:
        if agg[col].dtype == np.float64:
            agg[col] = agg[col].astype(np.float32)
    
    agg.rename(columns={col: "PREV_" + col for col in agg.columns if col != "SK_ID_CURR"}, inplace=True)
    
    agg.to_parquet(OUTPUT, index=False)
    print(f"Saved {OUTPUT} with shape {agg.shape}")

if __name__ == "__main__":
    main()
