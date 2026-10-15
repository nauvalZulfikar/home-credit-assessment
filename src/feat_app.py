import os
import numpy as np
import pandas as pd

def div(a, b):
    return a / b.replace(0, np.nan)

def main():
    RAW = os.environ.get("HC_RAW", "/Volumes/SongSJ/data/home-credit")
    FEAT = os.environ.get("HC_FEAT", "/Volumes/SongSJ/data/home-credit/feat")
    os.makedirs(FEAT, exist_ok=True)

    train = pd.read_csv(os.path.join(RAW, "application_train.csv"))
    test = pd.read_csv(os.path.join(RAW, "application_test.csv"))
    test["TARGET"] = np.nan
    df = pd.concat([train, test], ignore_index=True)
    df["IS_TRAIN"] = df["TARGET"].notna().astype(int)

    df = df[(df["CODE_GENDER"] != "XNA") | df["TARGET"].isna()].copy()

    df["DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == 365243).astype(int)
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace({365243: np.nan})

    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25
    df["EMPLOYED_YEARS"] = -df["DAYS_EMPLOYED"] / 365.25
    df["EMPLOYED_TO_AGE"] = div(df["DAYS_EMPLOYED"], df["DAYS_BIRTH"])
    df["CREDIT_TO_INCOME"] = div(df["AMT_CREDIT"], df["AMT_INCOME_TOTAL"])
    df["ANNUITY_TO_INCOME"] = div(df["AMT_ANNUITY"], df["AMT_INCOME_TOTAL"])
    df["CREDIT_TO_ANNUITY"] = div(df["AMT_CREDIT"], df["AMT_ANNUITY"])
    df["CREDIT_TO_GOODS"] = div(df["AMT_CREDIT"], df["AMT_GOODS_PRICE"])
    df["GOODS_MINUS_CREDIT"] = df["AMT_GOODS_PRICE"] - df["AMT_CREDIT"]
    df["INCOME_PER_PERSON"] = div(df["AMT_INCOME_TOTAL"], df["CNT_FAM_MEMBERS"])
    df["CHILDREN_RATIO"] = div(df["CNT_CHILDREN"], df["CNT_FAM_MEMBERS"])
    df["EXT_MEAN"] = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(axis=1, skipna=True)
    df["EXT_MIN"] = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].min(axis=1, skipna=True)
    df["EXT_MAX"] = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].max(axis=1, skipna=True)
    df["EXT_STD"] = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].std(axis=1, skipna=True)
    df["EXT_PROD"] = df["EXT_SOURCE_1"] * df["EXT_SOURCE_2"] * df["EXT_SOURCE_3"]
    df["EXT_NULL_CNT"] = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].isna().sum(axis=1)
    df["PHONE_CHANGE_TO_AGE"] = div(df["DAYS_LAST_PHONE_CHANGE"], df["DAYS_BIRTH"])
    df["ID_PUBLISH_TO_AGE"] = div(df["DAYS_ID_PUBLISH"], df["DAYS_BIRTH"])
    df["DOC_CNT"] = df.filter(like="FLAG_DOCUMENT_").sum(axis=1)
    df["REQ_BUREAU_TOTAL"] = df.filter(like="AMT_REQ_CREDIT_BUREAU_").sum(axis=1, min_count=1)
    df["SOCIAL_DEF_RATIO_30"] = div(df["DEF_30_CNT_SOCIAL_CIRCLE"], df["OBS_30_CNT_SOCIAL_CIRCLE"])
    df["REGION_MISMATCH_CNT"] = (
        df["REG_REGION_NOT_LIVE_REGION"] +
        df["REG_REGION_NOT_WORK_REGION"] +
        df["LIVE_REGION_NOT_WORK_REGION"] +
        df["REG_CITY_NOT_LIVE_CITY"] +
        df["REG_CITY_NOT_WORK_CITY"] +
        df["LIVE_CITY_NOT_WORK_CITY"]
    )

    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna("MISSING").astype("category")

    df = df.rename(columns=lambda x: "APP_" + x.upper() if x not in ["SK_ID_CURR", "TARGET", "IS_TRAIN"] else x)

    for col in df.select_dtypes(include=np.number).columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        if df[col].dtype == np.float64:
            df[col] = df[col].astype(np.float32)

    df.to_parquet(os.path.join(FEAT, "app.parquet"), index=False)
    print(df.shape)

if __name__ == "__main__":
    main()
