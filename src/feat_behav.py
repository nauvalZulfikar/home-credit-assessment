import os
import pandas as pd
import numpy as np
from os import makedirs
from os.path import exists, join, abspath, dirname

HC_RAW = os.environ.get("HC_RAW", "/Volumes/SongSJ/data/home-credit")
HC_FEAT = os.environ.get("HC_FEAT", "/Volumes/SongSJ/data/home-credit/feat")
REP = join(dirname(dirname(abspath(__file__))), "reports")

def div(a, b):
    return a / b.replace(0, np.nan)

def read_csv(path, usecols=None):
    df = pd.read_csv(path, usecols=usecols)
    for col in df.select_dtypes(include=np.number).columns:
        if df[col].dtype == np.float64:
            df[col] = df[col].astype(np.float32)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df

def inst_features():
    path = join(HC_RAW, "installments_payments.csv")
    i = read_csv(path, usecols=["SK_ID_CURR","SK_ID_PREV","NUM_INSTALMENT_VERSION","NUM_INSTALMENT_NUMBER","DAYS_INSTALMENT","DAYS_ENTRY_PAYMENT","AMT_INSTALMENT","AMT_PAYMENT"])
    # one instalment can be paid in several rows, so collapse to one row per instalment first
    i = i.groupby(["SK_ID_CURR","SK_ID_PREV","NUM_INSTALMENT_VERSION","NUM_INSTALMENT_NUMBER"], as_index=False).agg(
        DAYS_INSTALMENT=("DAYS_INSTALMENT","first"), DAYS_ENTRY_PAYMENT=("DAYS_ENTRY_PAYMENT","max"),
        AMT_INSTALMENT=("AMT_INSTALMENT","first"), AMT_PAYMENT=("AMT_PAYMENT","sum"))
    i["DPD"] = (i["DAYS_ENTRY_PAYMENT"] - i["DAYS_INSTALMENT"]).clip(lower=0)
    i["DBD"] = (i["DAYS_INSTALMENT"] - i["DAYS_ENTRY_PAYMENT"]).clip(lower=0)
    missed = i["DAYS_ENTRY_PAYMENT"].isna()
    i["LATE"] = ((i["DPD"] > 0) | missed).astype(int)
    i["LATE_30"] = ((i["DPD"] > 30) | missed).astype(int)
    i["PAY_RATIO"] = div(i["AMT_PAYMENT"], i["AMT_INSTALMENT"])
    i["UNDERPAID"] = (i["AMT_PAYMENT"] < i["AMT_INSTALMENT"] - 1).astype(int)
    i["IS_RECENT"] = (i["DAYS_INSTALMENT"] >= -365).astype(int)
    i["RECENT_LATE"] = i["LATE"] * i["IS_RECENT"]
    agg = i.groupby("SK_ID_CURR").agg(
        CNT=("DPD","count"), 
        DPD_MAX=("DPD","max"), 
        DPD_MEAN=("DPD","mean"), 
        DBD_MEAN=("DBD","mean"),
        LATE_MEAN=("LATE","mean"), 
        LATE_30_SUM=("LATE_30","sum"), 
        PAY_RATIO_MEAN=("PAY_RATIO","mean"), 
        PAY_RATIO_MIN=("PAY_RATIO","min"),
        UNDERPAID_MEAN=("UNDERPAID","mean"), 
        RECENT_LATE_SUM=("RECENT_LATE","sum"), 
        RECENT_CNT=("IS_RECENT","sum"),
        AMT_PAYMENT_SUM=("AMT_PAYMENT","sum"), 
        AMT_INSTALMENT_SUM=("AMT_INSTALMENT","sum")
    ).reset_index()
    agg["RECENT_LATE_RATE"] = div(agg["RECENT_LATE_SUM"], agg["RECENT_CNT"])
    agg["PAID_TO_DUE"] = div(agg["AMT_PAYMENT_SUM"], agg["AMT_INSTALMENT_SUM"])
    agg.rename(columns=lambda x: "INS_" + x if x != "SK_ID_CURR" else x, inplace=True)
    del i
    return agg

def pos_features():
    path = join(HC_RAW, "POS_CASH_balance.csv")
    p = read_csv(path)
    p["LATE"] = (p["SK_DPD"] > 0).astype(int)
    p["COMPLETED"] = (p["NAME_CONTRACT_STATUS"] == "Completed").astype(int)
    agg = p.groupby("SK_ID_CURR").agg(
        MONTHS=("MONTHS_BALANCE","count"), 
        DPD_MAX=("SK_DPD","max"), 
        DPD_DEF_MAX=("SK_DPD_DEF","max"),
        LATE_MEAN=("LATE","mean"), 
        COMPLETED_MEAN=("COMPLETED","mean"), 
        FUTURE_INST_MEAN=("CNT_INSTALMENT_FUTURE","mean")
    ).reset_index()
    latest = p.sort_values("MONTHS_BALANCE").drop_duplicates("SK_ID_PREV", keep="last")
    remaining = latest.groupby("SK_ID_CURR").agg(REMAINING_INST_SUM=("CNT_INSTALMENT_FUTURE","sum")).reset_index()
    agg = agg.merge(remaining, on="SK_ID_CURR", how="left")
    agg.rename(columns=lambda x: "POS_" + x if x != "SK_ID_CURR" else x, inplace=True)
    del p, latest, remaining
    return agg

def cc_features():
    path = join(HC_RAW, "credit_card_balance.csv")
    c = read_csv(path)
    c["UTIL"] = div(c["AMT_BALANCE"], c["AMT_CREDIT_LIMIT_ACTUAL"])
    c["PAY_TO_MIN"] = div(c["AMT_PAYMENT_CURRENT"], c["AMT_INST_MIN_REGULARITY"])
    c["LATE"] = (c["SK_DPD"] > 0).astype(int)
    c["ATM_SHARE"] = div(c["AMT_DRAWINGS_ATM_CURRENT"], c["AMT_DRAWINGS_CURRENT"])
    agg = c.groupby("SK_ID_CURR").agg(
        MONTHS=("MONTHS_BALANCE","count"), 
        UTIL_MEAN=("UTIL","mean"), 
        UTIL_MAX=("UTIL","max"),
        PAY_TO_MIN_MEAN=("PAY_TO_MIN","mean"), 
        LATE_MEAN=("LATE","mean"), 
        DPD_MAX=("SK_DPD","max"),
        BALANCE_MEAN=("AMT_BALANCE","mean"), 
        LIMIT_MEAN=("AMT_CREDIT_LIMIT_ACTUAL","mean"), 
        ATM_SHARE_MEAN=("ATM_SHARE","mean")
    ).reset_index()
    agg.rename(columns=lambda x: "CC_" + x if x != "SK_ID_CURR" else x, inplace=True)
    del c
    return agg

def main():
    makedirs(HC_FEAT, exist_ok=True)
    df_inst = inst_features()
    df_pos = pos_features()
    df_cc = cc_features()
    df = df_inst.merge(df_pos, on="SK_ID_CURR", how="outer").merge(df_cc, on="SK_ID_CURR", how="outer")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    for col in df.select_dtypes(include=np.number).columns:
        if df[col].dtype == np.float64:
            df[col] = df[col].astype(np.float32)
    df.to_parquet(join(HC_FEAT, "behav.parquet"), index=False)
    print(f"Saved behav.parquet with shape: {df.shape}")
    del df_inst, df_pos, df_cc, df
    import gc
    gc.collect()

if __name__ == "__main__":
    main()
