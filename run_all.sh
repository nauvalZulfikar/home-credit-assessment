#!/bin/bash
# Rebuild everything from raw CSVs. Set HC_RAW to the folder holding the Kaggle CSVs.
# 'bash run_all.sh models' stops after the model checks (no notebook, no slides).
set -e; cd "$(dirname "$0")"; PY=${PY:-.venv/bin/python}
for s in feat_app feat_bureau feat_prev feat_behav train_lgbm; do echo "== $s"; $PY src/$s.py; done
echo "== train_lgbm (no external scores)"; HC_NO_EXT=1 $PY src/train_lgbm.py
for s in train_logreg score evaluate value_monitor model_checks; do echo "== $s"; $PY src/$s.py; done
[ "$1" = models ] && exit 0
$PY src/build_notebook.py
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebook.ipynb
$PY src/build_slides.py
