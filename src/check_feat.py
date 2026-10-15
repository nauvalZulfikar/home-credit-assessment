"""Sanity check for a feature parquet: check_feat.py <file> <prefix> <min_rows> <min_cols> [required cols...]"""
import sys, numpy as np, pandas as pd
f, prefix, min_rows, min_cols, *req = sys.argv[1:]
df = pd.read_parquet(f)
errs = []
if len(df) < int(min_rows): errs.append(f"rows {len(df)} < {min_rows}")
if df.shape[1] < int(min_cols): errs.append(f"cols {df.shape[1]} < {min_cols}")
if "SK_ID_CURR" not in df: errs.append("SK_ID_CURR missing (must be a column)")
elif df["SK_ID_CURR"].duplicated().any(): errs.append("SK_ID_CURR not unique")
bad = [c for c in df.columns if c not in ("SK_ID_CURR", "TARGET", "IS_TRAIN") and not c.startswith(tuple(prefix.split(",")))]
if bad: errs.append(f"columns without prefix {prefix}: {bad[:10]}")
num = df.select_dtypes("number")
if np.isinf(num.to_numpy(dtype="float64", na_value=np.nan)).any(): errs.append("inf values present")
miss = [c for c in req if c not in df]
if miss: errs.append(f"required columns missing: {miss}")
allnan = [c for c in num.columns if num[c].isna().all()]
if allnan: errs.append(f"all-NaN columns (bug): {allnan}")
if errs: print("\n".join(errs)); sys.exit(1)
print("OK", df.shape)
