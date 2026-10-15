import os
import json
import nbformat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

ROOT = os.getcwd()
RAW = os.environ.get("HC_RAW", "/Volumes/SongSJ/data/home-credit")
FEAT = os.path.join(RAW, "feat")
REP = os.path.join(ROOT, "reports")
FIG = os.path.join(ROOT, "slides")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
pd.set_option("display.max_columns", 50)

def md(t):
    nb.cells.append(nbformat.v4.new_markdown_cell(t))

def code(t):
    nb.cells.append(nbformat.v4.new_code_cell(t))

nb = nbformat.v4.new_notebook()
md("# Home Credit Indonesia, Credit Default Risk Model\nData Scientist assessment by Nauval Zulfikar\n\n## Objective\nThe main objective is to create a model that can predict credit default, which is crucial to the business.\n\n## Project description\nHome Credit lends to people with little or no credit history, who are often turned away by banks. To lend to them responsibly, it has to judge from the application itself, and from whatever loan history exists at other lenders and at Home Credit, whether a client is likely to have trouble repaying. A good default model lets Home Credit approve more good clients and refuse fewer of them, while keeping losses from bad loans under control.\n\nThe data has 307,511 past applications with a known outcome (about 8% defaulted), plus six tables of the same clients' loan history.\n\n## Expected output\n- A credit risk model in at least two versions, a **boosting model** and a **logistic regression**.\n- A Python notebook and Python scripts.\n- A presentation of at most 5 slides summarising model performance, insights, the workflow used to build the model, and other important points.\n\n## Key success criteria\n- Data understanding.\n- Varied features that follow business logic, especially for the logistic regression.\n- Stable performance.\n- Incorporating Gen AI in the modeling process is a plus.\n\n## Success measure (set before modelling)\n- Both models must rank clients better (higher AUC) than the best simple rule a credit officer could use without a model.\n- Stable, with fold-to-fold AUC standard deviation below 0.01, and score PSI between train and test below 0.1.\n- Judged on what a cut-off does, meaning the defaults caught, the bad rate of approved clients and the good clients turned away.\n\n## Target and validation\n- **Target.** Kaggle's `TARGET` = 1 means the client had payment difficulties, a payment more than X days late on at least one of the first Y instalments of this loan. Kaggle does not publish X and Y, so \"default\" in this notebook means exactly this label.\n- **Unit and moment.** One row is one client's current application, and the prediction is made at the moment of application, so features use only the application and the history that existed before it.\n- **Validation.** Every model is scored with 5-fold cross-validation on the 307,511 labelled clients, and each client is scored by the fold model that never saw it (out-of-fold, OOF). The 48,744 Kaggle test clients have no label, so they are used only to check that the inputs and scores look like train (drift), never to measure accuracy.\n\n## Approach\nTwo models, **LightGBM** (performance) and a **WoE Logistic Regression scorecard** (interpretable, regulator-friendly). Pipeline code lives in `src/`; run `bash run_all.sh` to rebuild everything from the raw CSVs.")

md("## Running on Google Colab (optional)\nThe outputs below are already saved, so reading needs nothing. To rerun from scratch on Colab, add these in the 🔑 Secrets panel, `KAGGLE_API_TOKEN` (Kaggle → Settings → API Tokens → Generate New Token; a legacy `KAGGLE_USERNAME` + `KAGGLE_KEY` pair also works) and, only if the public clone fails, `GH_TOKEN` (a GitHub token with read access). Join the competition and accept its rules at kaggle.com/c/home-credit-default-risk. If a secret is missing, the next cell asks for it in a password box instead. It then clones the code, downloads the Kaggle data (about 700 MB) and rebuilds features and models with `run_all.sh` (about 20 to 30 minutes). Run locally, it does nothing.")
code("""
import os, sys, subprocess
if "google.colab" in sys.modules and not os.path.exists("src"):
    from google.colab import userdata
    def secret(k):
        try:
            return userdata.get(k)
        except userdata.SecretNotFoundError:
            return None
    from getpass import getpass
    clone = lambda url: subprocess.run(["git", "clone", "-q", url, "/content/home-credit-assessment"], capture_output=True, text=True, env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
    r = clone("https://github.com/nauvalZulfikar/home-credit-assessment.git")
    if r.returncode != 0:  # repo still private, a read token is needed
        gh = secret("GH_TOKEN") or getpass("GitHub token (GH_TOKEN secret not found): ").strip()
        r = clone(f"https://nauvalZulfikar:{gh}@github.com/nauvalZulfikar/home-credit-assessment.git")
        assert r.returncode == 0, "GitHub refused the token: " + r.stderr.replace(gh, "***")
    os.chdir("/content/home-credit-assessment")
    for k in ["KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_API_TOKEN"]:
        if secret(k):
            os.environ[k] = secret(k)
    if not (os.environ.get("KAGGLE_API_TOKEN") or os.environ.get("KAGGLE_KEY")):
        os.environ["KAGGLE_API_TOKEN"] = getpass("Kaggle API token (KAGGLE_API_TOKEN secret not found): ").strip()
    !pip -q install -r requirements.txt kagglehub
    import kagglehub
    os.environ["HC_RAW"] = kagglehub.competition_download("home-credit-default-risk")
    os.environ["HC_FEAT"] = os.path.join(os.environ["HC_RAW"], "feat")
    !PY=python bash run_all.sh models
    assert os.path.exists(os.path.join(os.environ["HC_FEAT"], "logreg_test.parquet")), "pipeline failed, see the log above"
""")
md("On Colab the code, data, features and models are now in place; everything below reads them exactly as it does locally.")

code("""
import os
import json
import nbformat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

ROOT = os.getcwd()
RAW = os.environ.get("HC_RAW", "/Volumes/SongSJ/data/home-credit")
FEAT = os.path.join(RAW, "feat")
REP = os.path.join(ROOT, "reports")
FIG = os.path.join(ROOT, "slides")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
pd.set_option("display.max_columns", 50)

import ast
from IPython.display import Code
def show_code(script, *names):
    # print functions straight from a pipeline script, so the notebook shows the code that actually ran
    src = open(os.path.join(ROOT, "src", script)).read()
    parts = [ast.get_source_segment(src, n) for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name in names]
    return Code((chr(10) * 2).join(parts), language="python")
""")

desc = pd.read_csv(os.path.join(RAW, "HomeCredit_columns_description.csv"), encoding="latin-1")
# the official file uses one generic sentence for all 47 building columns; spell each one out
part = {"APARTMENTS": "apartment size", "BASEMENTAREA": "basement area", "YEARS_BEGINEXPLUATATION": "years since the building was first occupied",
        "YEARS_BUILD": "age of the building", "COMMONAREA": "shared (common) area", "ELEVATORS": "number of elevators",
        "ENTRANCES": "number of entrances", "FLOORSMAX": "highest number of floors", "FLOORSMIN": "lowest number of floors",
        "LANDAREA": "land area", "LIVINGAPARTMENTS": "size of living apartments", "LIVINGAREA": "living area",
        "NONLIVINGAPARTMENTS": "size of non-living apartments", "NONLIVINGAREA": "non-living area", "TOTALAREA": "total area",
        "FONDKAPREMONT": "type of building repair fund", "HOUSETYPE": "type of house", "WALLSMATERIAL": "wall material",
        "EMERGENCYSTATE": "whether the building is in emergency state"}
text_cols = {"FONDKAPREMONT_MODE", "HOUSETYPE_MODE", "WALLSMATERIAL_MODE", "EMERGENCYSTATE_MODE"}
stat = {"AVG": "average", "MODE": "most common value", "MEDI": "median"}
building = desc.Description.str.startswith("Normalized information about building")
desc.loc[building, "Description"] = desc.loc[building, "Row"].map(
    lambda r: f"Client's building, {part[r.rsplit('_', 1)[0]]} ({stat[r.rsplit('_', 1)[1]]}" + ("" if r in text_cols else ", scaled 0-1") + ")")
# the file names SK_ID_BUREAU as SK_BUREAU_ID and lists NFLAG_MICRO_CASH, which is not in the data
desc["Row"] = desc.Row.str.strip().replace("SK_BUREAU_ID", "SK_ID_BUREAU")
desc = desc[desc.Row != "NFLAG_MICRO_CASH"]

ld = lambda f: json.load(open(os.path.join(REP, f)))
age_ext1 = pd.read_parquet(os.path.join(FEAT, "app.parquet"), columns=["APP_AGE_YEARS", "APP_EXT_SOURCE_1"]).corr().iloc[0, 1]
m1, m2, ev, vs = ld("lgbm_metrics.json"), ld("logreg_metrics.json"), ld("evaluate_summary.json"), ld("value_summary.json")
mx, mc = ld("lgbm_noext_metrics.json"), ld("model_checks.json")
ci = ev["auc_ci95"]
pc = lambda v: f"{v:.1%}"
rd = lambda f: pd.read_csv(os.path.join(REP, f))
seg_df, fair_df, base_df, cal_df, val_df, imp_df, fpsi_df, iv_df = (rd(f) for f in ["segments.csv", "fairness.csv", "baselines.csv", "calibration.csv", "value.csv", "lgbm_importance.csv", "feature_psi.csv", "logreg_iv_all.csv"])

def columns_md(table):
    rows = desc[desc.Table == table]
    return f"<details>\n<summary><b>Column descriptions ({len(rows)} columns)</b> (click to expand)</summary>\n\n| Column | Description |\n|---|---|\n" + "\n".join(
        f"| `{r.Row.strip()}` | {' '.join(str(r.Description).split()).replace('|', '/').replace(': ', ', ').replace(' — ', ', ')} |" for r in rows.itertuples()) + "\n\n</details>"

md("## 1. Data understanding\nEight files, the **current application** split into train and test (one row per client) plus six tables of the client's **loan history**, at other lenders (credit bureau) and at Home Credit itself.\n\n"
   "- Every table links to the client through `SK_ID_CURR`.\n- `bureau_balance` links to `bureau` through `SK_ID_BUREAU`.\n"
   "- `installments_payments`, `POS_CASH_balance` and `credit_card_balance` link to `previous_application` through `SK_ID_PREV`.\n\n"
   "History tables have many rows per client, so section 2 summarises each one to one row per client before joining it to the application.\n\n"
   "**Unit of observation.** One row of the modelling table is one client's current application. A client appears once, in train or in test, never both (checked in 1.3), so the cross-validation folds cannot put the same client on both sides. "
   "The data dictionary (meaning of every column) is the official `HomeCredit_columns_description.csv`, shown under each table below.")

md("""### Overview of the tables
| Table | Rows | Columns | One row is | What it is about |
|---|---|---|---|---|
| `application_train` | 307,511 | 122 | one client | The current loan application (income, loan size, job, family, housing). Has `TARGET` (1 = payment difficulties). |
| `application_test` | 48,744 | 121 | one client | Same as `application_train` but without `TARGET`, the clients to predict. |
| `bureau` | 1,716,428 | 17 | one loan at another lender | The client's loans at other lenders, from the credit bureau (amount, debt left, overdue). |
| `bureau_balance` | 27,299,925 | 3 | one bureau loan per month | Monthly status of each bureau loan, on time or how many days late. |
| `previous_application` | 1,670,214 | 37 | one earlier application | The client's earlier applications to Home Credit, whether approved, refused or cancelled, and on what terms. |
| `installments_payments` | 13,605,401 | 8 | one instalment | Each past Home Credit instalment, what was due and when vs what was paid and when. |
| `POS_CASH_balance` | 10,001,358 | 8 | one loan per month | Monthly status of Home Credit shop (point-of-sale) and cash loans. |
| `credit_card_balance` | 3,840,312 | 23 | one card per month | Monthly status of Home Credit credit cards (limit, balance, spending, payments). |
""")

md("### application_train, the current application (307,511 rows, one per client)\nThe main table, with the client's income, loan amount, annuity, age, employment, family, assets, housing and submitted documents. "
   "`TARGET` is the answer the model learns, 1 = the client had payment difficulties, 0 = paid on time.\n\n" + columns_md("application_{train|test}.csv"))
code("""
app = pd.read_csv(os.path.join(RAW, "application_train.csv"))
print(app.shape)
print("default rate:", round(app.TARGET.mean(), 4))
display(app.head())
""")

md("### application_test, applications to score (48,744 rows)\nSame columns as `application_train` but without `TARGET`. These are the clients the model predicts.")
code("""
display(pd.read_csv(os.path.join(RAW, "application_test.csv"), nrows=5))
""")

history = [
    ("bureau", "loans at other lenders from the credit bureau (1,716,428 rows, one per past loan)",
     "Every loan the client has at other financial institutions, active or closed, amount, current debt, amount overdue, start and end dates."),
    ("bureau_balance", "monthly status of credit bureau loans (27,299,925 rows, one per loan per month)",
     "Month-by-month status of each `bureau` loan, from paid on time to 1-30 days late, 31-60 days late and so on."),
    ("previous_application", "previous Home Credit applications (1,670,214 rows, one per application)",
     "Every earlier application to Home Credit, approved, refused or cancelled, amount asked vs granted, down payment, term and product type."),
    ("installments_payments", "Home Credit repayment history (13,605,401 rows, one per instalment)",
     "For each instalment, when and how much the client should have paid vs when and how much they actually paid. The most informative history table."),
    ("POS_CASH_balance", "Home Credit point-of-sale and cash loans, monthly (10,001,358 rows, one per loan per month)",
     "Monthly snapshot of point-of-sale and cash loans, with instalments left, contract status and days past due."),
    ("credit_card_balance", "Home Credit credit cards, monthly (3,840,312 rows, one per card per month)",
     "Monthly snapshot of Home Credit credit cards, with limit, balance, cash withdrawals, purchases, payment vs minimum payment and days past due."),
]
for name, title, text in history:
    md(f"### {name}, {title}\n{text}\n\n" + columns_md(f"{name}.csv"))
    code(f"""
display(pd.read_csv(os.path.join(RAW, "{name}.csv"), nrows=5))
""")

md("""### Exploratory analysis
From here on the checks are mostly on the main table `application_train`, with a data-quality pass over the history tables, in this order.
1. Column types. Are the stored types what the columns really are?
2. Odd values, meaning placeholders, impossible values, extreme outliers.
3. Duplicates, keys and near-constant columns.
4. Missing values, how much, in which blocks, and whether missing itself says something about risk.
5. Distributions of key numbers and categories.
6. Relationship with default.
7. Correlation between variables.
8. Train vs test. Is the data to score like the data the model learns from?""")

md("#### 1.1 Column types\n\nThe stored type does not always match the meaning. Many 0/1 flags and ratings are stored as numbers, counts are stored as decimals, and every date is stored as *days before the application* (a negative number). Flags and ratings are analysed as categories (default rate per value), counts as whole numbers, and day columns are turned into positive years (age, years employed), which also makes placeholder values such as 365243 stand out.")
code("""
rating_cols = {"REGION_RATING_CLIENT", "REGION_RATING_CLIENT_W_CITY"}
kinds = []
for c in app.columns:
    s = app[c]
    if c == "SK_ID_CURR":
        kind = "ID"
    elif c == "TARGET":
        kind = "target (0/1)"
    elif not pd.api.types.is_numeric_dtype(s):
        kind = "category stored as text"
    elif s.nunique() == 2:
        kind = "0/1 flag stored as number"
    elif c in rating_cols:
        kind = "rating 1-3 stored as number"
    elif c.startswith("DAYS_"):
        kind = "days before application (negative number)"
    elif s.dtype.kind == "f" and s.dropna().mod(1).eq(0).all():
        kind = "count stored as decimal"
    else:
        kind = "continuous number"
    kinds.append({"column": c, "stored_as": str(s.dtype), "actual_meaning": kind, "unique_values": s.nunique()})
kinds = pd.DataFrame(kinds)
summary = kinds.groupby(["actual_meaning", "stored_as"]).agg(columns=("column", "size"), examples=("column", lambda x: ", ".join(x[:4])))
with pd.option_context("display.max_colwidth", None):
    display(summary.reset_index())
""")
md('Of the 122 columns, 51 are stored as something other than what they mean. In total 32 numbers are really yes/no flags, 12 decimals are really counts, 2 numbers are 1-3 ratings and 5 day columns are negative day counts. From here on flags and ratings are read as categories, counts as whole numbers and day columns as positive years.')
code("""
text_cols = kinds[kinds.actual_meaning == "category stored as text"].column
cardinality = pd.DataFrame({
    "categories": [app[c].nunique() for c in text_cols],
    "rarest_category": [app[c].value_counts().index[-1] for c in text_cols],
    "rarest_count": [app[c].value_counts().iloc[-1] for c in text_cols],
    "missing_share": [round(app[c].isna().mean(), 3) for c in text_cols],
}, index=text_cols)
display(cardinality.sort_values("categories", ascending=False))
""")

md("Most text columns have only a handful of categories, but `ORGANIZATION_TYPE` has 58 and some categories are tiny, such as gender `XNA` (4 clients), family status `Unknown` (2), income type maternity leave (5). Groups this small give unreliable default rates, so the scorecard's binning merges them with neighbours instead of scoring them on their own.")
md("#### 1.2 Odd values\n\nEach row of the table is one rule and the number of clients who break it.")
code("""
years_employed_at_start = (app.DAYS_BIRTH - app.DAYS_EMPLOYED) / -365.25
rules = {
    "DAYS_EMPLOYED = 365243 (about 1,000 years)": app.DAYS_EMPLOYED == 365243,
    "ORGANIZATION_TYPE = 'XNA' (not available)": app.ORGANIZATION_TYPE == "XNA",
    "CODE_GENDER = 'XNA'": app.CODE_GENDER == "XNA",
    "NAME_FAMILY_STATUS = 'Unknown'": app.NAME_FAMILY_STATUS == "Unknown",
    "Income above 10 million": app.AMT_INCOME_TOTAL > 1e7,
    "Owns a car but car age is missing": (app.FLAG_OWN_CAR == "Y") & app.OWN_CAR_AGE.isna(),
    "Car older than 60 years": app.OWN_CAR_AGE > 60,
    "More than 100 people in social circle (OBS_30_CNT_SOCIAL_CIRCLE)": app.OBS_30_CNT_SOCIAL_CIRCLE > 100,
    "More than 50 bureau enquiries in one quarter": app.AMT_REQ_CREDIT_BUREAU_QRT > 50,
    "Children not fewer than family members": app.CNT_CHILDREN >= app.CNT_FAM_MEMBERS,
    "Annuity larger than the loan": app.AMT_ANNUITY > app.AMT_CREDIT,
    "Started the current job before age 14": (app.DAYS_EMPLOYED != 365243) & (years_employed_at_start < 14),
    "Any DAYS_ column in the future (positive)": (app.filter(like="DAYS_").drop(columns="DAYS_EMPLOYED") > 0).any(axis=1),
}
checks = pd.DataFrame({"rule": list(rules), "clients": [int(m.sum()) for m in rules.values()]})
checks["share"] = (checks.clients / len(app)).round(4)
with pd.option_context("display.max_colwidth", None):
    display(checks)
placeholder = app.DAYS_EMPLOYED == 365243
print("365243 rows are exactly the ORGANIZATION_TYPE = 'XNA' rows:", (placeholder == (app.ORGANIZATION_TYPE == "XNA")).all())
print("income type of those rows:", app.loc[placeholder, "NAME_INCOME_TYPE"].value_counts().to_dict())
display(app.loc[[app.AMT_INCOME_TOTAL.idxmax()], ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "NAME_INCOME_TYPE", "OCCUPATION_TYPE", "TARGET"]])
""")
md("The one large issue is `365243`. It sits on 55,374 clients (18%), exactly the rows with employer `XNA`, and they are pensioners or unemployed. It is a code for 'no current job', not a real duration, so it becomes the flag `DAYS_EMPLOYED_ANOM` and the value becomes missing, rather than dropping 18% of clients. Everything else is small. Four `XNA` genders are dropped from training, one labourer reports an income of 117 million on a small loan (most likely a typo, and the features use ratios rather than raw income), and old cars are kept. The impossible combinations at the bottom of the table never occur.")
code("""
bureau_q = pd.read_csv(os.path.join(RAW, "bureau.csv"), usecols=["DAYS_CREDIT_ENDDATE", "DAYS_ENDDATE_FACT", "AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM_LIMIT"])
prev_q = pd.read_csv(os.path.join(RAW, "previous_application.csv"), usecols=["DAYS_FIRST_DRAWING", "DAYS_FIRST_DUE", "DAYS_LAST_DUE_1ST_VERSION", "DAYS_LAST_DUE", "DAYS_TERMINATION", "NAME_CASH_LOAN_PURPOSE", "AMT_DOWN_PAYMENT"])
cc_q = pd.read_csv(os.path.join(RAW, "credit_card_balance.csv"), usecols=["AMT_BALANCE", "AMT_TOTAL_RECEIVABLE"])
hist_rules = [
    ("bureau", "DAYS_CREDIT_ENDDATE more than 100 years away", bureau_q.DAYS_CREDIT_ENDDATE.abs() > 36525),
    ("bureau", "DAYS_ENDDATE_FACT more than 100 years ago", bureau_q.DAYS_ENDDATE_FACT < -36525),
    ("bureau", "Negative debt (AMT_CREDIT_SUM_DEBT < 0)", bureau_q.AMT_CREDIT_SUM_DEBT < 0),
    ("bureau", "Negative credit limit", bureau_q.AMT_CREDIT_SUM_LIMIT < 0),
] + [("previous_application", f"{c} = 365243", prev_q[c] == 365243) for c in prev_q.filter(like="DAYS_")] + [
    ("previous_application", "NAME_CASH_LOAN_PURPOSE = 'XNA'", prev_q.NAME_CASH_LOAN_PURPOSE == "XNA"),
    ("previous_application", "Negative down payment", prev_q.AMT_DOWN_PAYMENT < 0),
    ("credit_card_balance", "Negative balance (card overpaid)", cc_q.AMT_BALANCE < 0),
    ("credit_card_balance", "Negative total receivable", cc_q.AMT_TOTAL_RECEIVABLE < 0),
]
hist_checks = pd.DataFrame([{"table": t, "rule": r, "rows": int(m.sum()), "share": round(m.mean(), 4)} for t, r, m in hist_rules])
with pd.option_context("display.max_colwidth", None):
    display(hist_checks)
""")
md('The history tables carry their own placeholders and impossible values. The `365243` code fills up to 56% of a date column in `previous_application`, and `bureau` has 8,418 loans with negative debt and 134 ending more than 100 years away. Placeholders and impossible dates become missing, negative debt becomes 0 and a negative limit becomes missing; a negative card balance (the client overpaid) really happens and is kept. Every fix decided here is coded in the feature scripts, shown in full in section 2.1.')

md("#### 1.3 Duplicates, keys and near-constant columns")
code("""
test = pd.read_csv(os.path.join(RAW, "application_test.csv"))
print("duplicate SK_ID_CURR in train:", app.SK_ID_CURR.duplicated().sum())
print("duplicate rows ignoring the ID:", app.drop(columns="SK_ID_CURR").duplicated().sum())
print("clients in both train and test:", app.SK_ID_CURR.isin(test.SK_ID_CURR).sum())
print("columns in train but not in test:", sorted(set(app.columns) - set(test.columns)))
top_share = app.drop(columns=["SK_ID_CURR"]).apply(lambda s: s.value_counts(normalize=True, dropna=False).iloc[0])
near_constant = top_share[top_share > 0.99].sort_values(ascending=False).round(4)
print(len(near_constant), "columns have one value in more than 99% of rows (almost no information):")
display(near_constant.to_frame("share of most common value"))
""")

md("The basics are clean, with no duplicate clients or rows, and no client in both train and test, so a model is never scored on someone it trained on. The 18 near-constant columns are mostly document flags; `feat_app.py` adds them up into one `DOC_CNT` column (how many documents were handed in), and the scorecard's IV filter keeps columns without signal out.")
md("#### 1.4 Missing values")
code("""
miss = app.isna().mean().sort_values(ascending=False)
print("application_train:", (miss > 0).sum(), "of", app.shape[1], "columns have missing values,", (miss > 0.5).sum(), "are >50% missing")
blocks = miss[miss > 0].groupby(app.isna().sum()[miss > 0]).apply(lambda s: list(s.index))
blocks = blocks[blocks.map(len) >= 3]
print("groups of 3+ columns missing on exactly the same number of rows (filled in together):")
for n, cols in blocks.items():
    print(f"  {n:>7,} rows missing: {len(cols)} columns, e.g. {', '.join(cols[:3])}")

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(15, 7), gridspec_kw={"width_ratios": [1.2, 1]})
m = miss[miss > 0].head(40)[::-1]
ax0.barh(m.index, m.values, color="#3b6ea5")
ax0.set_title("Missing share, top 40 columns")
ax0.tick_params(axis="y", labelsize=7)
cols = ["EXT_SOURCE_1", "EXT_SOURCE_3", "OWN_CAR_AGE", "OCCUPATION_TYPE", "APARTMENTS_AVG", "AMT_REQ_CREDIT_BUREAU_YEAR"]
rates = pd.DataFrame({c: app.TARGET.groupby(app[c].isna()).mean() for c in cols}).T.rename(columns={False: "value present", True: "value missing"})
rates.plot(kind="barh", ax=ax1, color=["#3b6ea5", "#d1793b"])
ax1.set_title("Default rate when the value is present vs missing")
ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
plt.tight_layout()
plt.show()
display(rates.round(3))

has_missing = app.columns[app.isna().any()]
nullity = app[has_missing].isna().corr()
fig, ax = plt.subplots(figsize=(10, 8.5))
im = ax.imshow(nullity, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(has_missing)), has_missing, rotation=90, fontsize=5)
ax.set_yticks(range(len(has_missing)), has_missing, fontsize=5)
ax.set_title("Nullity correlation (1 = always missing together)")
fig.colorbar(im, ax=ax, shrink=0.7)
plt.tight_layout()
plt.show()
""")

md('Missing values are not random. Whole parts of the form are left empty together (the building block on about half of the clients, the bureau-enquiry block on 41,519). Missing also carries risk. No `EXT_SOURCE_3` means 9.3% default against 7.8%, no bureau-enquiry record 10.3% against 7.7%; the exception is occupation, where missing is mostly pensioners (6.5% against 8.8%). Nothing is filled with a guess. LightGBM reads missing directly, the scorecard gives it its own bin, text columns get a `MISSING` category and `EXT_NULL_CNT` counts the missing external scores.')
md("#### 1.5 Distributions\n\nKey numbers, split by outcome (blue = repaid, orange = default). Income and loan amounts are shown on a log scale because of the extreme outliers.")
code("""
dist = pd.DataFrame({
    "income (log10)": np.log10(app.AMT_INCOME_TOTAL),
    "loan amount (log10)": np.log10(app.AMT_CREDIT),
    "annuity (log10)": np.log10(app.AMT_ANNUITY),
    "age (years)": -app.DAYS_BIRTH / 365.25,
    "years employed": (-app.DAYS_EMPLOYED / 365.25).where(app.DAYS_EMPLOYED != 365243),
    "EXT_SOURCE_1": app.EXT_SOURCE_1,
    "EXT_SOURCE_2": app.EXT_SOURCE_2,
    "EXT_SOURCE_3": app.EXT_SOURCE_3,
})
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for ax, c in zip(axes.flat, dist.columns):
    for t, color in [(0, "#3b6ea5"), (1, "#d1793b")]:
        ax.hist(dist.loc[app.TARGET == t, c].dropna(), bins=50, density=True, alpha=0.55, color=color, label="default" if t else "repaid")
    ax.set_title(c)
    ax.set_yticks([])
axes.flat[0].legend()
plt.tight_layout()
plt.show()
raw_cols = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE", "CNT_CHILDREN", "CNT_FAM_MEMBERS", "OWN_CAR_AGE"]
display(app[raw_cols].describe(percentiles=[0.01, 0.5, 0.99]).T.assign(skew=app[raw_cols].skew()).round(2))
""")
md('The external scores and age separate the two groups best. Defaulters have a lower median `EXT_SOURCE_2` (0.44 vs 0.57) and are younger (median 39 vs 43.5 years). Income and loan size overlap almost completely, and income is extremely skewed (skew 391, driven by the 117 million record). That is why section 2 builds ratios such as annuity-to-income instead of using raw amounts.')
code("""
fig, axes = plt.subplots(2, 2, figsize=(15, 9))
for ax, c in zip(axes.flat, ["NAME_INCOME_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "OCCUPATION_TYPE"]):
    g = app.groupby(c).TARGET.agg(["mean", "size"]).sort_values("mean")
    ax.barh(g.index, g["mean"], color=np.where(g["size"] < 100, "#bbbbbb", "#3b6ea5"))
    for y, (r, n) in enumerate(zip(g["mean"], g["size"])):
        ax.text(r, y, f" {n:,}", va="center", fontsize=7)
    ax.axvline(app.TARGET.mean(), color="#d1793b", ls="--", lw=1)
    ax.set_title(f"Default rate by {c} (label = clients; grey = under 100 clients)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
plt.tight_layout()
plt.show()
""")

md('Job and housing matter. Low-skill labourers default at 17.2% against 4.8% for accountants, renters at 12.3% against 6.6% in office apartments, and working clients at 9.6% against 5.4% for pensioners. Grey bars are groups under 100 clients and are too small to read.')
md("#### 1.6 Relationship with default")

code("""
fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15, 4))
r = app.groupby(pd.qcut(app.EXT_SOURCE_2, 10, labels=range(1, 11)), observed=True).TARGET.mean()
ax0.bar(r.index.astype(str), r.values, color="#3b6ea5")
ax0.set_title("Default rate by EXT_SOURCE_2 decile (1 = lowest score)")
r = app.groupby(pd.cut(-app.DAYS_BIRTH / 365.25, [20, 25, 30, 35, 40, 45, 50, 55, 60, 70]), observed=True).TARGET.mean()
ax1.bar([f"{int(b.left)}-{int(b.right)}" for b in r.index], r.values, color="#3b6ea5")
ax1.set_title("Default rate by age band")
r = app.groupby(np.where(app.DAYS_EMPLOYED == 365243, "placeholder 365243", "employed")).TARGET.mean()
ax2.bar(r.index, r.values, color="#3b6ea5")
ax2.set_title("Default rate by employment placeholder")
for ax in (ax0, ax1, ax2):
    ax.set_ylabel("default rate")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1%}".replace(".0%", "%")))
plt.tight_layout()
plt.savefig(os.path.join(FIG, "eda.png"), bbox_inches="tight")
plt.show()
""")

md('The external score is the strongest single signal, and the default rate falls step by step from 18.4% in the lowest `EXT_SOURCE_2` decile to 3.0% in the highest. Risk also falls with age, from 12.3% at 20-25 to 4.9% at 60-70, and the 365243 group defaults less (5.4%) than clients with a real employment value (8.7%), which is why it is kept as a flag.')
code("""
def rate_by(label, groups):
    g = app.TARGET.groupby(groups).agg(["mean", "size"])
    return pd.DataFrame({"segment": label, "group": g.index.astype(str), "default_rate": g["mean"].round(3), "clients": g["size"]})

bureau_ids = pd.read_csv(os.path.join(RAW, "bureau.csv"), usecols=["SK_ID_CURR"]).SK_ID_CURR
prev = pd.read_csv(os.path.join(RAW, "previous_application.csv"), usecols=["SK_ID_CURR", "NAME_CONTRACT_STATUS"])
inst = pd.read_csv(os.path.join(RAW, "installments_payments.csv"), usecols=["SK_ID_CURR", "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT"])
late_share = ((inst.DAYS_ENTRY_PAYMENT - inst.DAYS_INSTALMENT) > 0).groupby(inst.SK_ID_CURR).mean()
late = app.SK_ID_CURR.map(late_share)

segments = pd.concat([
    rate_by("employment placeholder", np.where(app.DAYS_EMPLOYED == 365243, "365243 (no job)", "real value")),
    rate_by("education", app.NAME_EDUCATION_TYPE),
    rate_by("region rating", app.REGION_RATING_CLIENT),
    rate_by("contract type", app.NAME_CONTRACT_TYPE),
    rate_by("gender", app.CODE_GENDER),
    rate_by("has credit bureau record", app.SK_ID_CURR.isin(bureau_ids)),
    rate_by("ever refused by Home Credit", app.SK_ID_CURR.isin(prev[prev.NAME_CONTRACT_STATUS == "Refused"].SK_ID_CURR)),
    rate_by("late on past instalments", np.select([late.isna(), late == 0, late <= 0.1], ["no instalment history", "never late", "late on 1-10%"], "late on >10%")),
], ignore_index=True)
display(segments)
""")

md('Almost every segment points the expected way. A higher education, a better region rating, a credit bureau record, no earlier refusal and a clean repayment record all go with less default (for example 6.8% for clients never late against 10.8% for those late on more than 10% of instalments). The exception is clients with no instalment history, who default at 6.0%, slightly less than never-late clients. These are associations, not causes. Men default more than women (10.1% vs 7.0%), but gender is left out of both models for fair lending.')
md(f"Information Value (IV) of all {len(iv_df)} candidate variables of the scorecard, the standard credit-scoring measure of how well one variable separates good and bad clients (computed in `src/train_logreg.py`). The usual reading is below 0.02 useless, 0.02-0.1 weak, 0.1-0.3 medium, 0.3-0.5 strong, above 0.5 suspiciously strong.")
code("""
iv_all = pd.read_csv(os.path.join(REP, "logreg_iv_all.csv")).sort_values("iv", ascending=False)
iv_all["strength"] = pd.cut(iv_all.iv, [-np.inf, 0.02, 0.1, 0.3, 0.5, np.inf], labels=["useless", "weak", "medium", "strong", "suspicious"])
print(iv_all.strength.value_counts().reindex(["suspicious", "strong", "medium", "weak", "useless"]).to_dict())
display(iv_all[["name", "iv", "strength", "selected"]].head(15).round(3))
""")

iv_top = iv_df.sort_values("iv", ascending=False).iloc[0]
md(f"The external scores carry most of the signal. The strongest variable (`{iv_top['name']}`, IV {iv_top.iv:.2f}) " + ("crosses" if iv_top.iv > 0.5 else "comes close to") + " the 0.5 'suspiciously strong' line, which usually warns of a leak. Kaggle does not say when the external scores are produced; this notebook assumes they are known when the client applies, so they are treated as real signal, and the risk of leaning on them too much is tested later with a LightGBM trained without them. "
   f"Most other variables are weak on their own ({iv_df.iv.between(0.02, 0.1).sum()} between 0.02 and 0.1, {(iv_df.iv < 0.02).sum()} below 0.02), which is normal for a scorecard, since it adds many weak signals together.")
md("#### 1.7 Correlation\n\nThe left chart shows the numbers most related to default, the right one the pairs of columns that carry almost the same information. That matters for the logistic regression, where near-duplicate inputs make the coefficients unstable.")
code("""
num = app.select_dtypes("number").drop(columns=["SK_ID_CURR", "TARGET"])
corr_t = num.corrwith(app.TARGET)
top = corr_t.reindex(corr_t.abs().sort_values(ascending=False).index).head(15)[::-1]
corr = num.corr()
key = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "DAYS_BIRTH", "DAYS_EMPLOYED", "AMT_CREDIT", "AMT_GOODS_PRICE", "AMT_ANNUITY", "AMT_INCOME_TOTAL", "REGION_RATING_CLIENT", "REGION_RATING_CLIENT_W_CITY", "CNT_CHILDREN", "CNT_FAM_MEMBERS"]
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, 6.5), gridspec_kw={"width_ratios": [1, 1.3]})
ax0.barh(top.index, top.values, color=np.where(top.values < 0, "#3b6ea5", "#d1793b"))
ax0.set_title("Correlation with default (top 15)")
im = ax1.imshow(corr.loc[key, key], cmap="RdBu_r", vmin=-1, vmax=1)
ax1.set_xticks(range(len(key)), key, rotation=90, fontsize=8)
ax1.set_yticks(range(len(key)), key, fontsize=8)
ax1.set_title("Correlation between key columns")
fig.colorbar(im, ax=ax1, shrink=0.8)
plt.tight_layout()
plt.show()
upper = corr.abs().where(np.triu(np.ones(corr.shape, dtype=bool), 1)).stack()
print((upper > 0.9).sum(), "pairs of numeric columns have |correlation| > 0.9")
display(upper.sort_values(ascending=False).head(12).round(3).to_frame("abs correlation"))
""")

md("No single number is strongly related to default. The strongest are the three external scores (correlation -0.16 to -0.18) and age. Many columns are near-copies. In all, 62 pairs correlate above 0.9, mostly the AVG/MEDI/MODE versions of the same building measure, and `FLAG_EMP_PHONE` matches the 365243 placeholder exactly. Near-copies make logistic regression coefficients unstable, so the scorecard's IV and direction filter keeps most of them out; LightGBM is not hurt by them.")
md("#### 1.8 Train vs test\n\nTwo checks are used, **PSI** per column (below 0.1 = stable, 0.1-0.2 = some shift, above 0.2 = large shift), and **adversarial validation**, where a model is trained to tell train rows from test rows. An AUC near 0.5 means the two sets look alike; the higher it is, the more they differ.")
code("""
import lightgbm as lgb
from sklearn.model_selection import cross_val_predict

def psi(expected, actual, bins=10):
    edges = np.unique(np.nanquantile(expected, np.linspace(0, 1, bins + 1)))
    e = pd.Series(pd.cut(expected, edges, include_lowest=True)).value_counts(normalize=True, dropna=False, sort=False)
    a = pd.Series(pd.cut(actual, edges, include_lowest=True)).value_counts(normalize=True, dropna=False, sort=False)
    e, a = e.align(a, fill_value=0)
    e, a = e.clip(lower=1e-4), a.clip(lower=1e-4)
    return float(((a - e) * np.log(a / e)).sum())

psi_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "DAYS_BIRTH", "DAYS_EMPLOYED", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE", "DAYS_ID_PUBLISH", "DAYS_REGISTRATION", "DAYS_LAST_PHONE_CHANGE"]
psi_table = pd.Series({c: psi(app[c], test[c]) for c in psi_cols}).sort_values(ascending=False).round(4).to_frame("PSI train vs test")
display(psi_table)
print("cash loans share  train:", round((app.NAME_CONTRACT_TYPE == "Cash loans").mean(), 3), " test:", round((test.NAME_CONTRACT_TYPE == "Cash loans").mean(), 3))

both = pd.concat([app.drop(columns="TARGET").sample(50000, random_state=0), test], ignore_index=True)
is_test = np.r_[np.zeros(50000), np.ones(len(test))]
X = both.drop(columns="SK_ID_CURR")
for c in X.columns[~X.dtypes.map(pd.api.types.is_numeric_dtype)]:
    X[c] = X[c].astype("category")
adv = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, verbose=-1)
pred = cross_val_predict(adv, X, is_test, cv=3, method="predict_proba")[:, 1]
print("adversarial validation AUC:", round(roc_auc_score(is_test, pred), 3))
adv.fit(X, is_test)
display(pd.Series(adv.booster_.feature_importance("gain"), index=X.columns).sort_values(ascending=False).head(8).round(0).to_frame("gain: what separates train from test"))
""")

md('The 12 raw columns checked look like train (every PSI below 0.1), but that is not the whole picture. Engineered features are checked in section 6.2, where the loan-term ratio shifts strongly, and the product mix differs. Test is 99.1% cash loans against 90.5% in train, and a model can tell the two sets apart with AUC 0.877, mainly through loan amount, annuity and the bureau-enquiry counts. The model is therefore checked separately on cash loans in section 5, and the shift is listed as a limitation.')
md("""### Key observations
**Target**
- Only 8.1% of clients defaulted (24,825 of 307,511), so accuracy is meaningless. Models are judged on AUC / Gini / KS and trained with stratified folds.

**Column types**
- Stored types do not match meaning. In total 32 numeric columns are really 0/1 flags, 12 decimals are really counts, 2 columns are 1-3 ratings and all `DAYS_` columns are negative day counts. They are treated by meaning, not by stored type.
- `ORGANIZATION_TYPE` has 58 categories (the rarest has 24 clients); the scorecard's binning merges such small groups, LightGBM handles them natively.

**Odd values**
- `DAYS_EMPLOYED = 365243` (about 1,000 years) is a placeholder on 55,374 clients (18%), 55,352 pensioners and 22 unemployed, exactly the rows with `ORGANIZATION_TYPE = 'XNA'`. It becomes a flag (`DAYS_EMPLOYED_ANOM`) and the value is set to missing. These clients default *less* (5.4% vs 8.7%).
- One client reports an income of 117 million (99th percentile is 472,500, skew 391), so ratios such as credit-to-income are used instead of raw income. Only 4 clients have gender `XNA`, and they are dropped.
- In all, 3,339 clients have a car older than 60 years, likely a data-entry convention; kept as is.
- History tables use the same 365243 placeholder in 5 date columns of `previous_application` (set to missing in feature engineering). `bureau` has 134 loans ending more than 100 years away and 8,418 with negative debt; `feat_bureau.py` sets the impossible dates to missing and the negative debt to 0 before aggregating.
- Checked and clean, no positive `DAYS_` values, no duplicate clients or rows, no client in both train and test, no test category unseen in training.

**Missing values**
- Of 122 columns, 67 have missing values and 41 are more than half empty. They go missing in blocks, the building columns (AVG/MODE/MEDI triplets) together, and the 6 `AMT_REQ_CREDIT_BUREAU_*` columns together on 41,519 rows.
- Missing is informative. Clients with no `EXT_SOURCE_3` default at 9.3% vs 7.8%, with no bureau enquiry record 10.3% vs 7.7%. LightGBM uses missing directly; the scorecard gives it its own bin.

**Redundant columns**
- Some 18 columns have one value in more than 99% of rows (mostly `FLAG_DOCUMENT_*`, `FLAG_MOBIL`), so they carry almost no information.
- In all, 62 pairs correlate above 0.9, mostly the AVG/MEDI/MODE versions of the same building measure, and `FLAG_EMP_PHONE` duplicates the employment placeholder. The scorecard's IV + direction filter keeps most of these out, but there is no explicit correlation filter; adding one is a next step.

**Strongest signals**
- The external scores `EXT_SOURCE_1..3` are the strongest predictors (IV 0.16-0.34; their combined mean reaches 0.63, above the 0.5 "too good to be true" mark, so it is the first thing to question for leakage; the scores come with the application data, so they are kept).
- Risk falls with age (12.3% default under 25 vs 4.9% over 60), with education (5.4% higher education vs 10.9% lower secondary) and with region rating (4.8% vs 11.1%).
- Gender differs (men 10.1%, women 7.0%) but is left out of both models for fair lending.

**Loan history**
- Clients ever refused by Home Credit default at 10.3% vs 7.0%; clients late on more than 10% of past instalments at 10.8% vs 6.8% for never late.
- About 14% of clients have no credit bureau record and default more (10.1% vs 7.7%), so a thin file is itself a risk signal.

**Train vs test**
- Raw columns are stable (all PSI below 0.1; a derived one, the loan term, is not, see 6.2), but a model can tell train from test apart (adversarial AUC 0.88), mainly through loan amount, bureau enquiries and contract type. The test set is 99% cash loans vs 90.5% in training. Expect test performance to differ from validation; stability is checked again in section 5.

**How to read these findings**
- The questions set before looking at the data were the success measure in the introduction. Everything above was found while exploring, so it is a **hypothesis** to check in the models, not a conclusion.
- These are associations, **not causes**. Older clients default less, but that does not mean age makes people repay. Claims of cause need an experiment.
""")

md("## 2. Feature engineering (business logic)\n| Group | Examples | Business meaning |\n|---|---|---|\n| Affordability | CREDIT_TO_INCOME, ANNUITY_TO_INCOME, CREDIT_TO_ANNUITY | debt burden and implied loan term |\n| Stability | AGE_YEARS, EMPLOYED_TO_AGE, ID_PUBLISH_TO_AGE | tenure in job, life stage |\n| External scores | EXT_MEAN, EXT_MIN, EXT_NULL_CNT | outside risk scores (source not documented) + data thinness |\n| Credit bureau | BUR_ACTIVE_CNT, BUR_TOTAL_DEBT_RATIO, BUR_BB_DPD_MAX | outside debt and past delinquency |\n| Past Home Credit apps | PREV_REFUSED_MEAN, PREV_RECENT_REFUSED_CNT | credit hunger, past rejections |\n| Repayment behaviour | INS_RECENT_LATE_RATE, INS_PAY_RATIO_MIN, CC_UTIL_MAX | how they actually pay |\n\nScripts are `src/feat_app.py`, `feat_bureau.py`, `feat_prev.py` and `feat_behav.py`. CODE_GENDER is excluded from both models (fair-lending).\n\n"
   "### Cleaning decisions\n"
   "- The raw CSVs are never changed; every script reads them and writes its output to separate files (`feat/*.parquet`), so `run_all.sh` rebuilds everything from scratch.\n"
   "- **Missing, zero and not applicable are kept apart.** A client with no bureau record gets empty (missing) `BUR_` features, not 0, because zero would claim \"no debt\" when the truth is \"no record\". The `365243` employment placeholder becomes a flag plus missing. Negative bureau debt (an overpayment) becomes 0, because the debt really is zero; a negative credit limit is impossible and becomes missing.\n"
   "- Outliers are flagged, not deleted. The 117-million income stays in the data and ratios are used instead of raw income. Only the 4 `XNA` gender rows are dropped.\n"
   "- Each feature script is checked automatically after it runs for one row per client, no client lost, the expected columns present.")

code("""
for n in ["app","bureau","prev","behav"]:
    df = pd.read_parquet(os.path.join(FEAT, n + ".parquet"))
    print(n, df.shape)
    display(df.head())
""")
md('Each history table is now one row per client and joins on `SK_ID_CURR` without duplicating anyone. Not every client has every history. Of the 356,251 applications, 50,440 (14%) have no credit bureau record, 17,394 (5%) no earlier Home Credit application and 15,886 (4%) no repayment history. Their history columns stay missing after the join, which section 1.6 showed is itself a signal (no bureau record means 10.1% default against 7.7%). Joined together the four tables give 224 columns. LightGBM uses 220 of them (everything except the client id, the target, the train/test flag and gender), and the scorecard keeps 29 of its 73 candidates after the IV and direction checks in section 4.')

md("### Do the new features help?\nEach engineered feature is compared with the raw column it was built from, on the training clients. **Gini** is how well the feature alone ranks defaulters above non-defaulters (0 = no better than chance, 1 = perfect), and the two default rates are for the 20% of clients with the lowest and the highest value.")
code("""
tr = pd.read_parquet(os.path.join(FEAT, "app.parquet"))
for n in ["bureau", "prev", "behav"]:
    tr = tr.merge(pd.read_parquet(os.path.join(FEAT, n + ".parquet")), on="SK_ID_CURR", how="left")
tr = tr[tr.TARGET.notna()]

def single_gini(c):
    s = tr[[c, "TARGET"]].dropna()
    q = pd.qcut(s[c].rank(method="first"), 5, labels=False)
    r = s.TARGET.groupby(q).mean()
    return abs(2 * roc_auc_score(s.TARGET, s[c]) - 1), r.iloc[0], r.iloc[-1], len(s) / len(tr)

pairs = [("APP_EXT_MEAN", "APP_EXT_SOURCE_2"), ("BUR_TOTAL_DEBT_RATIO", "BUR_AMT_DEBT_SUM"), ("PREV_REFUSED_MEAN", "PREV_CNT"),
         ("APP_CREDIT_TO_GOODS", "APP_AMT_GOODS_PRICE"), ("APP_CREDIT_TO_ANNUITY", "APP_AMT_CREDIT"), ("INS_RECENT_LATE_RATE", "INS_LATE_MEAN"),
         ("APP_ANNUITY_TO_INCOME", "APP_AMT_INCOME_TOTAL"), ("APP_EMPLOYED_TO_AGE", "APP_EMPLOYED_YEARS")]
rows = []
for new, raw in pairs:
    for kind, c in [("new", new), ("raw", raw)]:
        g, lo, hi, cov = single_gini(c)
        rows.append({"pair": new, "kind": kind, "column": c, "gini": g, "default_low_20%": lo, "default_high_20%": hi, "clients_with_value": cov})
display(pd.DataFrame(rows).set_index(["pair", "kind"]).round(3))
""")
md("In six of the eight pairs the engineered feature ranks clients better than the raw column behind it. The external score average reaches a Gini of 0.43 (default 18.7% in the lowest fifth vs 2.5% in the highest) against 0.31 for `EXT_SOURCE_2` alone, the most complete single score. `EXT_SOURCE_3` alone is stronger (Gini 0.36) but is missing for a fifth of clients, which the average covers. Debt relative to credit beats raw debt (0.18 vs 0.10), and the share of past applications refused beats their plain count (0.12 vs 0.02), because how often a client was turned down says more than how often they applied. Loan amount relative to goods price doubles the raw price (0.14 vs 0.07), and the implied loan term beats the raw credit amount (0.06 vs 0.04). The recent late rate (last 365 days) beats the all-time late rate (0.15 vs 0.13), which is why the last year is used as \"recent\" (it reflects current habits), even though only 71% of clients have instalments in that window against 95% overall. Not every ratio wins. Annuity relative to income is level with raw income (0.04 both), and time employed relative to age is a little weaker than years employed alone (0.14 vs 0.17). Both are kept because the models pick among them, and the scorecard IV check in section 4 drops what adds nothing.")

md("### 2.1 The cleaning and feature code\nThe four feature scripts, printed straight from `src/` so this is the code that produced the tables above. Each reads the raw CSV, applies the fixes from section 1.2, and writes one row per client.")
code("""
show_code("feat_app.py", "div", "main")
""")
md("In the main application the 4 `XNA` gender rows are dropped (train only), `365243` becomes the `DAYS_EMPLOYED_ANOM` flag plus missing, days become years, and the affordability and stability ratios are built. Text columns become categories, with missing kept as its own `MISSING` value.")
code("""
show_code("feat_bureau.py", "main")
""")
md("In the credit bureau data (1,716,428 loans), dates more than 100 years away become missing (134 end dates, 95 update dates and 1 actual end date), 8,418 negative debts become 0 and 351 negative limits become missing, then every client's loans are summed up (counts, active share, debt ratio, worst days past due), with the monthly balance table added on top.")
code("""
show_code("feat_prev.py", "main")
""")
md("In previous Home Credit loans (1,670,214 applications) the `365243` placeholder in the five date columns becomes missing. It is common there, 934,444 first-drawing dates, 225,913 termination dates, 211,221 last-due dates, 93,864 first-version last-due dates and 40,645 first-due dates, so it means \"did not happen\" rather than a real date. Refusal rate, refusals in the last 365 days and amounts asked vs granted are then summed up per client.")
code("""
show_code("feat_behav.py", "inst_features", "pos_features", "cc_features", "main")
""")
behav = pd.read_parquet(os.path.join(FEAT, "behav.parquet"), columns=["INS_LATE_MEAN", "INS_UNDERPAID_MEAN"])
md(f"For payment behaviour, late and under-paid instalments, overdue months on POS loans, and credit-card use against the limit, each summed up per client and joined into one table. Among clients with instalment history, {pc((behav.INS_LATE_MEAN[behav.INS_LATE_MEAN.notna()] > 0).mean())} paid at least one instalment late or missed it, so the useful signal is how often, not whether, while only {pc((behav.INS_UNDERPAID_MEAN[behav.INS_UNDERPAID_MEAN.notna()] > 0).mean())} ever paid less than an instalment's amount (several partial payments of one instalment are added up first). A late month on a POS loan shows up for 18% of clients, and 29% have a credit card. About 12% of all clients (4 in 10 card holders) went over their limit at least once.")

md(f"## 3. Model A (LightGBM)\nStratified 5-fold CV, {m1['n_features']} features, native categorical handling. Early stopping uses an inner 10% split of each training fold, so the fold being scored never decides when training stops (the first version stopped on the scored fold and had uncleaned bureau dates; now OOF AUC {m1['oof_auc']:.4f}). Each fold's model is saved in `models/`; `src/score.py` reloads them and reproduces the stored test predictions. Script `src/train_lgbm.py`.")
md("The training script, printed from `src/train_lgbm.py`.")
code("""
show_code("train_lgbm.py", "load_data", "main")
""")
md("It joins the four feature tables, runs 5-fold stratified CV with early stopping on an inner split, saves the out-of-fold and test predictions, and saves each fold's model to `models/`. The results below come from this run.")

code("""
m1 = json.load(open(os.path.join(REP, "lgbm_metrics.json")))
metrics = pd.DataFrame({
    "oof_auc": [m1["oof_auc"]],
    "oof_gini": [m1["oof_gini"]],
    "oof_ks": [m1["oof_ks"]],
    "auc_mean": [m1["auc_mean"]],
    "auc_std": [m1["auc_std"]],
    "train_auc_last_fold": [m1["train_auc_last_fold"]],
    "psi_train_vs_test": [round(m1["psi_train_vs_test"], 4)]
})
display(metrics.round(4))
""")
md(f"The fold AUC standard deviation is only {m1['auc_std']:.4f} (folds {min(m1['fold_auc']):.3f} to {max(m1['fold_auc']):.3f}), so the result does not depend on which clients landed in which fold. The gap between the train AUC of the last fold's model ({m1['train_auc_last_fold']:.3f}) and validation ({m1['oof_auc']:.3f}) shows some overfitting, which is why early stopping is used.")

code("""
imp = pd.read_csv(os.path.join(REP, "lgbm_importance.csv")).head(20)
fig, ax = plt.subplots(figsize=(7, 6))
imp.sort_values("gain", ascending=False).plot(kind="barh", x="feature", y="gain", ax=ax)
ax.set_title("LightGBM, top 20 features (gain)")
ax.invert_yaxis()
plt.savefig(os.path.join(FIG, "lgbm_importance.png"), bbox_inches="tight")
plt.show()
""")
g = imp_df.sort_values("gain", ascending=False).set_index("feature").gain / imp_df.gain.sum()
md(f"The external scores lead by far. `{g.index[0]}` alone gives {g.iloc[0]:.0%} of the model's total gain, and all external-score features together {g[g.index.str.contains('EXT_')].sum():.0%}. The strongest of the rest are "
   + ", ".join(f"`{f}` ({v:.1%})" for f, v in g[~g.index.str.contains("EXT_")].head(5).items()) + ". Importance shows what the model uses, not what causes default.")

md("## 4. Model B (WoE Logistic Regression scorecard)\nOptimal monotonic binning (optbinning) → Weight of Evidence → L2 logistic regression; IV ≥ 0.02 filter, top 30 by IV. Scaled to points, where 600 points means good-to-bad odds of 50 to 1 and every 20 points (PDO) double the odds. Script `src/train_logreg.py`.")
md("The scorecard script, printed from `src/train_logreg.py`.")
code("""
show_code("train_logreg.py", "signed_vars", "make_bp", "main")
""")
md("`make_bp` bins each variable so default risk only rises or only falls across bins, `signed_vars` keeps only variables whose effect has the expected direction, and `main` fits the logistic regression inside each CV fold, then turns it into points. The results below come from this run.")

code("""
m2 = json.load(open(os.path.join(REP, "logreg_metrics.json")))
metrics = pd.DataFrame({
    "oof_auc": [m2["oof_auc"]],
    "oof_gini": [m2["oof_gini"]],
    "oof_ks": [m2["oof_ks"]],
    "auc_mean": [m2["auc_mean"]],
    "auc_std": [m2["auc_std"]],
    "train_auc_full": [m2["train_auc_full"]],
    "psi_train_vs_test": [round(m2["psi_train_vs_test"], 4)]
})
display(metrics.round(4))
iv = pd.read_csv(os.path.join(REP, "logreg_iv.csv"))
coef = pd.read_csv(os.path.join(REP, "logreg_coef.csv"))
display(iv[iv.selected == True].head(15))
print("all coefficients same sign:", (coef.coef < 0).all() or (coef.coef > 0).all())
""")
dropped = m2["dropped_wrong_sign"]
md(f"The scorecard almost never overfits (train AUC of the full fit {m2['train_auc_full']:.3f} vs validation {m2['oof_auc']:.3f}), and every remaining variable pushes risk in the expected direction, so each point can be explained to a client. "
   + (f"To get there {', '.join(f'`{d}`' for d in dropped)} " + ("was" if len(dropped) == 1 else "were") + " dropped, because the fitted coefficient came out with the wrong sign. "
      + ("For age this is most likely overlap with the external scores, since `APP_EXT_SOURCE_1` correlates {:.2f} with age and five external-score variables are in the scorecard, so age's own effect is already counted. Age therefore has no direct points, but it still acts through the external scores (see the age fairness check in 5.3).".format(age_ext1) if "APP_AGE_YEARS" in dropped else "")
      if dropped else "No variable had to be dropped for a wrong sign."))

code("""
sc = pd.read_csv(os.path.join(REP, "scorecard.csv"))
top_iv_var = iv[iv.selected == True].name.iloc[0]
display(sc[sc.Variable == top_iv_var][["Variable","Bin","Count (%)","Event rate","WoE","Points"]])
""")
scb = rd("scorecard.csv")
top_var = pd.read_csv(os.path.join(REP, "logreg_iv.csv")).query("selected == True").name.iloc[0]
scb = scb[scb.Variable == top_var]
real = scb[~scb.Bin.isin(["Special", "Missing"]) & (scb.Count > 0)]
lo_b, hi_b, miss_b = real.iloc[0], real.iloc[-1], scb[scb.Bin == "Missing"].iloc[0]
md(f"Each bin carries points, and the lower the bin's default rate, the more points it gives. For `{top_var}`, clients in the lowest bin `{lo_b.Bin}` default at {pc(lo_b['Event rate'])} and get {lo_b.Points:.0f} points, clients in the highest bin `{hi_b.Bin}` default at {pc(hi_b['Event rate'])} and get {hi_b.Points:.0f} points. "
   f"A missing value is not given neutral points by rule; it is its own bin, scored by its own default rate ({pc(miss_b['Event rate'])}, {miss_b.Points:.0f} points here). The points of all {m2['n_selected']} variables add up to the client's score. On the test applications scores run from {m2['score_min']:.0f} to {m2['score_max']:.0f} (median {m2['score_median']:.0f}). This is the table a credit officer reads.")

md("## 5. Comparison, baselines & stability")

code("""
o1 = pd.read_parquet(os.path.join(FEAT, "lgbm_oof.parquet"))
o2 = pd.read_parquet(os.path.join(FEAT, "logreg_oof.parquet"))
display(o1.head(), o2.head())
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 4.5))

fpr1, tpr1, _ = roc_curve(o1.TARGET, o1.PRED)
auc1 = round(roc_auc_score(o1.TARGET, o1.PRED), 3)
ax0.plot(fpr1, tpr1, label=f"LightGBM (AUC={auc1})")
fpr2, tpr2, _ = roc_curve(o2.TARGET, o2.PRED)
auc2 = round(roc_auc_score(o2.TARGET, o2.PRED), 3)
ax0.plot(fpr2, tpr2, label=f"Scorecard (AUC={auc2})")
ax0.plot([0, 1], [0, 1], "k--")
ax0.set_title("ROC curves")
ax0.legend()

decile_bad_rate1 = m1["decile_bad_rate"]
decile_bad_rate2 = m2["decile_bad_rate"]
x = np.arange(1, 11)
ax1.bar(x - 0.2, decile_bad_rate1, 0.4, label="LightGBM")
ax1.bar(x + 0.2, decile_bad_rate2, 0.4, label="Scorecard")
ax1.set_xticks(x)
ax1.set_xlabel("predicted-risk decile (10 = riskiest)")
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax1.set_title("Actual default rate by predicted-risk decile")
ax1.legend()

plt.tight_layout()
plt.savefig(os.path.join(FIG, "compare.png"), bbox_inches="tight")
plt.show()
""")
md(f"Both models sort clients well. In LightGBM the safest tenth of clients defaults at {pc(m1['decile_bad_rate'][0])} and the riskiest tenth at {pc(m1['decile_bad_rate'][-1])}; in the scorecard {pc(m2['decile_bad_rate'][0])} against {pc(m2['decile_bad_rate'][-1])}. LightGBM spreads the risk wider at both ends, and its ROC line sits above the scorecard's everywhere.")

code("""
comparison = pd.DataFrame({
    "Model": ["LightGBM", "Scorecard"],
    "OOF AUC": [m1["oof_auc"], m2["oof_auc"]],
    "Gini": [m1["oof_gini"], m2["oof_gini"]],
    "KS": [m1["oof_ks"], m2["oof_ks"]],
    "Fold AUC std": [m1["auc_std"], m2["auc_std"]],
    "Train AUC (LightGBM last fold, scorecard full fit)": [m1["train_auc_last_fold"], m2["train_auc_full"]],
    "PSI train→test": [round(m1["psi_train_vs_test"], 4), round(m2["psi_train_vs_test"], 4)]
})
display(comparison.round(4))
comparison.to_csv(os.path.join(REP, "comparison.csv"), index=False)
""")
md(f"LightGBM leads by {m1['oof_auc'] - m2['oof_auc']:.3f} AUC (95% range of the gap {ci['diff'][0]:.3f} to {ci['diff'][1]:.3f}, from 200 resamples of the clients), so the lead is real, not noise. Both scores keep the same distribution between train and test (score PSI {m1['psi_train_vs_test']:.4f} and {m2['psi_train_vs_test']:.4f}), but not every input does, see section 6.2.")

md(f"""### 5.1 Against simple rules (baselines)
Before trusting a model, compare it with what a credit officer could do without one. Every row is scored the same way, AUC / Gini / KS for ranking, and what happens if the riskiest 10% of applicants are refused.
- **B0** approve everyone, **B1** refuse by the most complete single external score (`EXT_SOURCE_2`, filled for almost every client), **B2** refuse by the average of the three external scores, **B3** refuse by annuity-to-income.
- The best simple rule is **{ev['best_baseline']}** (AUC {ev['best_baseline_auc']:.3f}). Both models beat it, **{'yes' if ev['models_beat_baseline'] else 'NO'}**.
- LightGBM without any external score reaches AUC {mx['oof_auc']:.3f}. The external scores are the backbone, but the rest of the data alone still beats the best rule built from them {'(yes)' if mx['oof_auc'] > ev['best_baseline_auc'] else '(no)'}.""")
code("""
from IPython.display import Image, display
display(pd.read_csv(os.path.join(REP, "baselines.csv")).set_index("model"))
display(Image(os.path.join(REP, "fig_reject_curve.png")))
""")
bb = base_df.set_index("model").caught_pct
md(f"With the riskiest 10% of clients refused, LightGBM catches {pc(bb['LightGBM'])} of all defaulters and the scorecard {pc(bb['Scorecard (WoE logistic)'])}, against {pc(bb.filter(like='B2').iloc[0])} for the best simple rule (average of the external scores) and {pc(bb.filter(like='B3').iloc[0])} for an annuity-to-income rule. Even LightGBM without any external score catches {pc(bb['LightGBM without external scores'])}, so the models add value beyond the external scores alone. The curve shows the same order at every refusal level.")

md(f"""### 5.2 Segments
The same model can work well overall and badly for one group. Groups with fewer than 1,000 clients or 100 defaults are marked `small` (their AUC is noisy). `flag_drop` marks an AUC more than 0.03 below the overall {m1['oof_auc']:.3f}.
- Flagged groups (not small) are {', '.join(s.replace(': ', ' ') for s in ev['flagged_segments']) or 'none'}.
- The test set is {pc(ev['test_cash_share'])} cash loans vs {pc(ev['train_cash_share'])} in training. LightGBM AUC on cash loans is {ev['lgbm_auc_cash']:.3f} (revolving {ev['lgbm_auc_revolving']:.3f}). The cash-loan figure is the closer guide for the test set, but only an upper bound, since the test loans also have shorter terms than train (section 6.2).""")
code("""
seg = pd.read_csv(os.path.join(REP, "segments.csv"))
display(seg.set_index(["segment", "group"]))
display(pd.read_csv(os.path.join(REP, "test_mix.csv")))
""")
sg = seg_df[~seg_df.small]
near = sg[~sg.flag_drop & (sg.auc_drop_lgbm > 0.02)]
fg = sg[sg.flag_drop]
lab = lambda r: f"{r.segment} {r.group} ({r.auc_lgbm:.3f}, default rate {pc(r.default_rate)})"
md(f"AUC ranges from {sg.auc_lgbm.min():.3f} to {sg.auc_lgbm.max():.3f} across the groups that are not small, against {m1['oof_auc']:.3f} overall. "
   + (f"Flagged are {', '.join(lab(r) for r in fg.itertuples())}. " if len(fg) else "No group is flagged. ")
   + (f"Just inside the line are {', '.join(lab(r) for r in near.itertuples())}. " if len(near) else "")
   + "Pensioners, the 365243 employment code and the 60+ band are largely the same clients. The model separates good from bad less sharply among older and younger clients, so these are the groups to watch after launch. Groups marked `small` should not be judged on their AUC.")

md(f"""### 5.3 Fairness and calibration
Gender is not a model input, but other columns can stand in for it. With the riskiest 10% refused, the table shows per gender and age group how many are refused, how many good clients are refused and how many defaulters are caught. LightGBM's rejection rate differs by {ev['fairness_max_gap_lgbm']*100:.1f} percentage points between women and men, against a {ev['default_rate_gap_gender']*100:.1f}-point gap in their actual default rates; age is judged the same way below.

Calibration means that if the model says 20%, about 20% should default. The chart puts predicted against actual default rate per decile; points on the dashed line mean the predicted probability matches the observed default rate in this data, not just the ranking.""")
code("""
display(pd.read_csv(os.path.join(REP, "fairness.csv")).set_index(["model", "segment", "group"]))
display(Image(os.path.join(REP, "fig_calibration.png")))
sc_chk = json.load(open(os.path.join(REP, "score_check.json")))
print("saved models reproduce the stored test predictions:", sc_chk["match"], sc_chk)
""")
bt = mc["behaviour_tests"]; v = mc["versions"]
fl = fair_df[fair_df.model == "LightGBM"].set_index(["segment", "group"])
fs = fair_df[fair_df.model != "LightGBM"].set_index(["segment", "group"])
F, M, Y, O = fl.loc[("Gender", "F")], fl.loc[("Gender", "M")], fl.loc[("Age band", "<30")], fl.loc[("Age band", "60+")]
cl = cal_df[cal_df.model == "LightGBM"].set_index("decile")
miss = (cl.actual_rate - cl.mean_pred).abs().idxmax()
md(f"""Both groups are judged by one standard, the ratio of refusal rates against the ratio of default rates. With the riskiest 10% refused, LightGBM refuses {pc(F.rejection_rate)} of women and {pc(M.rejection_rate)} of men, a ratio of {M.rejection_rate / F.rejection_rate:.1f} to 1, while men default {M.default_rate / F.default_rate:.1f} times as often ({pc(M.default_rate)} vs {pc(F.default_rate)}). Good clients refused are {pc(F.good_rejected_pct)} against {pc(M.good_rejected_pct)}.

By age the same standard shows a much larger gap. Among under-30s {pc(Y.rejection_rate)} are refused against {pc(O.rejection_rate)} of clients aged 60+, a ratio of {Y.rejection_rate / O.rejection_rate:.1f} to 1, while their default rates differ only {Y.default_rate / O.default_rate:.1f} times ({pc(Y.default_rate)} vs {pc(O.default_rate)}), and good young clients are refused {pc(Y.good_rejected_pct)} of the time against {pc(O.good_rejected_pct)}. Part of this is mechanical, because refusing only the riskiest tenth magnifies any difference in risk, as the smaller gender gap also shows. Even so, the age gap is far larger than the risk gap and needs a review with the credit and compliance teams before use. The scorecard, which has no direct age points, still refuses {pc(fs.loc[('Age band', '<30')].rejection_rate)} of under-30s, so age reaches it through other variables.

Calibration is good on this data. The predicted and actual default rates per decile stay close, for example {pc(cl.loc[10].mean_pred)} predicted against {pc(cl.loc[10].actual_rate)} actual in LightGBM's riskiest decile; the largest miss is decile {miss} ({pc(cl.loc[miss].mean_pred)} vs {pc(cl.loc[miss].actual_rate)}). This holds for the approved, mostly cash-loan population the model was trained on. Before the probabilities are used for pricing or provisions on new applications they must be re-checked, and if needed recalibrated, on recent loans with a known outcome, since the test applications already differ (section 6.2). The saved model files give the same test predictions as training did, so what is shipped is what was evaluated.

### 5.4 Behaviour tests and uncertainty
A good AUC does not prove the model reacts sensibly. On 3,000 training clients, each test changes one thing and checks the direction of the predictions (`src/model_checks.py`).
- **Should not move.** Flipping gender (not a model input) and changing the client ID.
- **Should fall.** Raising the external scores by 0.1.
- **Should rise.** Raising the share of late payments by 0.2, or days past due by 30.

Result, {sum(t['passed'] for t in bt)} of {len(bt)} tests passed{'.' if mc['all_passed'] else ', ' + ', '.join(t['test'] for t in bt if not t['passed']) + ' failed.'}

**Uncertainty (95% range, 200 resamples of the clients).** LightGBM AUC {ci['LightGBM'][0]:.3f} to {ci['LightGBM'][1]:.3f}; scorecard {ci['Scorecard (WoE logistic)'][0]:.3f} to {ci['Scorecard (WoE logistic)'][1]:.3f}; the difference {ci['diff'][0]:.3f} to {ci['diff'][1]:.3f}.

**Reproducibility.** Seed {v['seed']}, {v['folds']} folds; Python {v['python']}, LightGBM {v['lightgbm']}, pandas {v['pandas']}, numpy {v['numpy']}, scikit-learn {v['sklearn']}, optbinning {v['optbinning']}.""")
code("""
mc = json.load(open(os.path.join(REP, "model_checks.json")))
display(pd.DataFrame(mc["behaviour_tests"]).set_index("test"))
""")
B = {t["test"]: t for t in bt}
L = lambda k: B[f"LightGBM: {k}"]
md(f"{'All' if mc['all_passed'] else 'Not all'} {len(bt)} tests pass. Changing gender or client ID changes no prediction at all (largest change {max(t['max_abs_change'] for t in bt if t['expect'] == 'same'):.4f}); since neither is an input, these two only prove the pipeline does not pick them up by accident. "
   f"Raising the external scores by 0.1 lowers LightGBM's predicted risk by {-L('external scores +0.1')['mean_change'] * 100:.1f} points on average ({pc(L('external scores +0.1')['share_moved_right_way'])} of clients move down); a late-payment share 0.2 higher raises it by {L('late-payment share +0.2')['mean_change'] * 100:.1f} points ({pc(L('late-payment share +0.2')['share_moved_right_way'])} move up), and 30 more days past due by {L('days past due +30')['mean_change'] * 100:.1f} points ({pc(L('days past due +30')['share_moved_right_way'])} up). "
   f"The scorecard moves the right way for at least {pc(min(t['share_moved_right_way'] for t in bt if t['test'].startswith('Scorecard')))} of clients, as its signs are fixed. The model reacts the way a credit officer would expect.")

l10, s10 = vs["lgbm_at_10"], vs["scorecard_at_10"]
md(f"""## 6. Business use
### 6.1 Cut-off
The riskiest LightGBM decile defaults at {pc(m1['decile_bad_rate'][-1])} vs {pc(m1['decile_bad_rate'][0])} in the safest. The table shows, for each share of applicants refused, the defaulters turned away and the good clients lost, in clients and in credit amount (`AMT_CREDIT`, in the data's currency). **No margin or loss rate is assumed.** The last column is the break-even condition instead.

Refusing the riskiest 10% with LightGBM turns away {l10['defaults_avoided']:,} defaulters holding {l10['credit_defaults_avoided']/1e9:.2f} bn of credit, and {l10['good_clients_rejected']:,} good clients holding {l10['credit_good_rejected']/1e9:.2f} bn (scorecard {s10['defaults_avoided']:,} and {s10['good_clients_rejected']:,}). The break-even ratio is {l10['break_even_ratio']:.2f}, so this cut-off pays off **only if the money lost on one unit of defaulted credit is more than {l10['break_even_ratio']:.2f} times the profit earned on one unit of good credit**. Across 5-50% refused the ratio runs from {vs['break_even_ratio_range_lgbm'][0]:.2f} to {vs['break_even_ratio_range_lgbm'][1]:.2f}; the business picks the cut-off by putting its own loss and margin figures into this condition.

**Limit (reject inference).** All outcomes come from loans that were approved. Applicants refused in the past have no outcome, so the effect on the full applicant flow is not known from this data. The safe way to confirm it is a small live test that applies the cut-off to a random share of applications and compares default rates after a few months.

**Live test size, fixed before it starts.** To see the approved bad rate fall from {pc(mc['ab_test']['bad_rate_now'])} to {pc(mc['ab_test']['bad_rate_expected'])} (the drop the model predicts at 10% refused), with the usual 5% false-alarm risk and 80% chance of seeing a real effect, each group needs about **{mc['ab_test']['clients_per_group']:,} approved loans**; to detect only half that drop, about {mc['ab_test']['clients_per_group_half_effect']:,}. The test ends at that size, not when the result first looks good, and the outcome is read only after the loans have had time to default.""")
code("""
val = pd.read_csv(os.path.join(REP, "value.csv"))
display(val.set_index(["model", "reject_pct"]))
display(Image(os.path.join(REP, "fig_value.png")))
""")
vl = val_df[val_df.model == "LightGBM"].set_index(val_df.reject_pct.round(2)[val_df.model == "LightGBM"])
md(f"Refusing more clients always avoids more bad credit, but the good credit lost grows faster. The ratio rises from {vl.loc[0.05].break_even_ratio:.2f} at 5% refused to {vl.loc[0.1].break_even_ratio:.2f} at 10%, {vl.loc[0.2].break_even_ratio:.2f} at 20% and {vl.loc[0.5].break_even_ratio:.2f} at 50%. "
   f"In clients, going from 10% to 20% refused avoids another {vl.loc[0.2].defaults_avoided - vl.loc[0.1].defaults_avoided:,.0f} defaulters but turns away another {vl.loc[0.2].good_clients_rejected - vl.loc[0.1].good_clients_rejected:,.0f} good clients. "
   "Credit is counted at the full loan amount on both sides. A loan that defaults after some repayments, or with some money recovered, loses less than its full amount, so the business's loss figure per unit of defaulted credit has to include its recovery rate before it is compared with this ratio.")

md(f"""### 6.2 Monitoring and retraining
| What | How often | Watch | Act |
|---|---|---|---|
| PSI of the model score (train vs new applications) | monthly | > 0.1 | > 0.25, review and retrain |
| PSI of the top-10 features | monthly | > 0.1 | > 0.25 on any, find the cause and retrain if real |
| Product mix (cash vs revolving share) | monthly | any shift | re-check the segment AUC of the growing product |
| Early default (missed first instalments) of new loans | monthly | above the decile's expected rate | review the cut-off |
| Gini on loans with a known outcome | quarterly | drop > 5% relative | drop > 10%, retrain |
| Calibration (predicted vs actual per decile) | quarterly | points leave the diagonal | recalibrate |
| Fairness (refusal rate by gender and age) | quarterly | gap widens | review features that may stand in for gender |

**Champion and challenger.** A retrained model replaces the current one only if it is better on recent data on Gini, stability, calibration and fairness, and both run side by side on a share of traffic first. The current train vs test check gives score PSI {vs['score_psi_lgbm']:.4f} (LightGBM) and {vs['score_psi_scorecard']:.4f} (scorecard); the largest feature PSI is {vs['feature_psi_max']:.3f} (`{vs['feature_psi_max_name']}`), {vs['n_features_watch_or_act']} of the top 10 features at "watch" or above.

**Rollback.** Every trained model is saved as files in `models/` with its feature list. A new version is stored next to the old one, so going back means pointing `src/score.py` at the previous files, a few minutes of work.""")
code("""
display(pd.read_csv(os.path.join(REP, "feature_psi.csv")).set_index("feature"))
""")
drift = [r for r in fpsi_df.itertuples() if r.status != "ok" and not r.feature.startswith("Model")]
rank = {f: i + 1 for i, f in enumerate(imp_df.sort_values("gain", ascending=False).feature)}
app_d = pd.read_parquet(os.path.join(FEAT, "app.parquet"), columns=["IS_TRAIN", "APP_NAME_CONTRACT_TYPE"] + [r.feature for r in drift if r.feature.startswith("APP_")])
def drift_md(r):
    t = f"`{r.feature}`, the number {rank[r.feature]} LightGBM feature, shifted strongly (PSI {r.psi:.2f}, status `{r.status}`)."
    if r.feature in app_d:
        cash = app_d[app_d.APP_NAME_CONTRACT_TYPE == "Cash loans"].groupby("IS_TRAIN")[r.feature]
        t += f" Its median is {app_d[app_d.IS_TRAIN == 1][r.feature].median():.1f} in train and {app_d[app_d.IS_TRAIN == 0][r.feature].median():.1f} in test, and among cash loans alone {cash.median()[1]:.1f} against {cash.median()[0]:.1f}, so the shift is not just the product mix."
    if r.feature == "APP_CREDIT_TO_ANNUITY":
        t += f" Credit divided by the instalment is roughly the number of monthly instalments (a little fewer, as instalments include interest), so the test applications ask for shorter loans (the longest is {app_d[app_d.IS_TRAIN == 0][r.feature].max():.0f} against {app_d[app_d.IS_TRAIN == 1][r.feature].max():.0f} in train)."
    return t
md(f"The model scores barely moved (PSI {vs['score_psi_lgbm']:.4f} and {vs['score_psi_scorecard']:.4f}), so the overall score distribution of the test applications looks like train. "
   + ("No top feature shifted enough to need a review." if not drift else " ".join(drift_md(r) for r in drift)
      + " A stable score with a shifted input means the model is being asked about loans it has seen few of, so its accuracy there is unproven. In production this is exactly what the monthly check must catch and explain before trusting the score."))

a10 = ev["at_10"]
hot = [r for r in fpsi_df.itertuples() if r.status == "act" and not r.feature.startswith("Model")]
go_md = ((f"{', '.join(f'`{r.feature}` (PSI {r.psi:.2f}, number {rank[r.feature]} feature)' for r in hot)} shifted between train and test. Go live only when the cause is known and LightGBM, checked on recent cash loans with a known outcome, keeps its AUC within 5% of {m1['oof_auc']:.3f} (at least {m1['oof_auc'] * 0.95:.3f}); otherwise retrain on recent applications first.") if hot
         else f"No input shifted beyond the act line. Go live after the shadow run confirms an AUC within 5% of {m1['oof_auc']:.3f} on recent loans.")
ok_std = m1["auc_std"] < 0.01 and m2["auc_std"] < 0.01
ok_psi = m1["psi_train_vs_test"] < 0.1 and m2["psi_train_vs_test"] < 0.1
md(f"""## 7. Conclusion & limitations
### Decision
- **LightGBM ranks best** (OOF AUC {m1['oof_auc']:.3f} vs {m2['oof_auc']:.3f}). With 10% refused it catches {pc(ev['at_10']['LightGBM']['caught_pct'])} of defaulters vs {pc(ev['at_10']['Scorecard (WoE logistic)']['caught_pct'])} for the scorecard, leaving an approved bad rate of {pc(ev['at_10']['LightGBM']['approved_bad_rate'])} vs {pc(ev['at_10']['Scorecard (WoE logistic)']['approved_bad_rate'])}.
- **The scorecard is the explainable version**, with {m2['n_selected']} variables, all with the expected direction ({', '.join(m2['dropped_wrong_sign']) or 'none'} dropped for a wrong sign, section 4), points per attribute that can be shown to a client or to the regulator.
- The recommendation is to use LightGBM to rank applications and the scorecard alongside it to give reasons for each refusal and as a fallback; switch the roles only if explainability rules require one simple model.

### Success measure check
| Measure | Target | LightGBM | Scorecard | Met |
|---|---|---|---|---|
| Beats best simple rule ({ev['best_baseline']}, AUC {ev['best_baseline_auc']:.3f}) | higher AUC | {m1['oof_auc']:.3f} | {m2['oof_auc']:.3f} | {'yes' if ev['models_beat_baseline'] else 'no'} |
| Fold AUC std | < 0.01 | {m1['auc_std']:.4f} | {m2['auc_std']:.4f} | {'yes' if ok_std else 'no'} |
| Score PSI train vs test | < 0.1 | {m1['psi_train_vs_test']:.4f} | {m2['psi_train_vs_test']:.4f} | {'yes' if ok_psi else 'no'} |
| Cut-off effect at 10% refused | reported | caught {pc(a10['LightGBM']['caught_pct'])}, approved bad rate {pc(a10['LightGBM']['approved_bad_rate'])}, good refused {pc(a10['LightGBM']['good_rejected_pct'])} | caught {pc(a10['Scorecard (WoE logistic)']['caught_pct'])}, approved bad rate {pc(a10['Scorecard (WoE logistic)']['approved_bad_rate'])}, good refused {pc(a10['Scorecard (WoE logistic)']['good_rejected_pct'])} | yes, the cut-off itself is the business's choice (section 6.1) |

The score PSI passes, but one strong input does not (section 6.2), so stability on the test applications is only partly shown.

### Go-live condition
The models are ready for a shadow run, not for live decisions. {go_md}

### Limitations
- **Uncertainty.** LightGBM AUC 95% range {ci['LightGBM'][0]:.3f}-{ci['LightGBM'][1]:.3f}, scorecard {ci['Scorecard (WoE logistic)'][0]:.3f}-{ci['Scorecard (WoE logistic)'][1]:.3f} (section 5.4).
- **Validation is out-of-fold only.** The data has no application dates, so a split by time (train on older, test on newer) is not possible; real performance on future clients may be lower.
- **Reject inference.** Outcomes exist only for approved loans (section 6.1).
- **Test applications differ.** {pc(ev['test_cash_share'])} cash loans in test vs {pc(ev['train_cash_share'])} in training, and shorter loan terms even within cash loans (section 6.2). The cash-loan AUC ({ev['lgbm_auc_cash']:.3f}) is the closer guide, but its accuracy on the test loans is unproven.
- **Dependence on the external scores.** Without them LightGBM drops from {m1['oof_auc']:.3f} to {mx['oof_auc']:.3f}. If the score provider changes its method, both models must be re-checked.
- **Fairness.** Gender is excluded, but correlated columns can still carry it. Refusals fall on under-30s {Y.rejection_rate / O.rejection_rate:.1f} times as often as on clients aged 60+, against a {Y.default_rate / O.default_rate:.1f}-times gap in default (section 5.3).
- **Overlapping external scores.** The scorecard holds several versions of the external scores; they are strongly related, which most likely flipped the sign of age and removed it.
- **Correlated columns.** 62 pairs above 0.9 with no explicit filter; harmless for ranking, but it splits importance between near-copies.

### Next steps
1. Agree a cut-off with the business using its own loss and margin figures (section 6.1) and confirm it in a small live test.
2. Ask for application dates to validate by time.
3. Add a correlation filter and test reason codes from the scorecard with the credit team.
4. Set up the monitoring table in section 6.2 on monthly data.
5. This is an offline model. It still has to be connected to the approval system, run in shadow next to the current process, then the live test above.""")

nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbformat.write(nb, os.path.join(ROOT, "notebook.ipynb"))
