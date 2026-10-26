# Predicting Student At-Risk / Fail Status with a Decision Tree

MCS 306 – Introduction to Artificial Intelligence (AY 2026-2027, 1st Sem) · Final Course Output

A **Decision Tree Classifier** that flags students likely to end with a **D or F** (the *positive class*,
"At-risk/Fail") from behavioural data: weekly self-study hours, attendance %, class participation.
Dataset: *Student Performance Dataset* (Kaggle), 1,000,000 students.

The code follows the workflow of **Chapter 9 – Machine Learning Foundations** of the course lecture notes
(train/validation/test split, `Pipeline` + stratified 5-fold `GridSearchCV`, confusion-matrix metrics,
threshold selection on validation data, cost model, learning curves, calibration, subgroup check).


## Headline results (test set: 150,000 students, used once)

| Model / threshold | t | Accuracy | Precision | Recall | F1 | ROC-AUC | Cost per student |
|---|---|---|---|---|---|---|---|
| Majority class | 0.50 | 0.949 | undefined | 0.000 | 0.000 | 0.500 | 0.512 |
| Logistic regression (F1 threshold) | 0.27 | 0.950 | 0.510 | 0.619 | 0.559 | 0.956 | 0.256 |
| **Decision tree** (depth 5, 32 leaves) – F1 threshold | 0.30 | 0.951 | 0.517 | 0.610 | 0.560 | 0.956 | 0.258 |
| **Decision tree** – cost threshold | 0.16 | 0.927 | 0.394 | 0.778 | 0.523 | 0.956 | 0.236 |

Thresholds were chosen on the **validation** set: `t_F1` maximises F1; `t_cost` minimises
`10·FN + 2·FP` (illustrative costs taken from the lecture-note example; replace with your own).
Key finding: **weekly self-study hours is the only informative predictor**; attendance and participation
have ~0 correlation with the outcome (the data look simulated – see the paper).

## Repository structure

```
.
├── main.py
├── requirements.txt
├── src/
│   ├── config.py
│   ├── data.py
│   ├── metrics.py
│   ├── model.py
│   └── plots.py
├── tests/test_lesson_examples.py
├── data/student_performance.csv
├── figures/
├── results/
├── models/decision_tree.joblib
├── paper/
└── docs/
```

## What each file does

| File | Purpose |
|---|---|
| `main.py` | Runs the whole workflow |
| `src/config.py` | Seed (11), split sizes, features, costs, hyperparameter grid |
| `src/data.py` | Loading, target `y`, stratified train/validation/test split |
| `src/metrics.py` | TP/FN/FP/TN, accuracy to F1, cost model, log loss, Gini (from the notes' formulas) |
| `src/model.py` | Pipelines, GridSearchCV (1-SE rule), thresholds, diagnostics, rules |
| `src/plots.py` | All 19 figures |
| `tests/test_lesson_examples.py` | Checks the metric code against worked examples in the lecture notes |
| `paper/build_paper.py` | Generates and compiles the LaTeX paper |

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m tests.test_lesson_examples
python main.py
python paper/build_paper.py
```

## The workflow in `main.py` (Chapter 9, Sec. 11.5)

1. Define target `y` (1 = grade D/F), positive class, cost criterion.
2. Reserve the **test set first** (15 %, stratified); split the rest into **training** (70 %) and **validation** (15 %).
3. Inspect data (EDA), build baselines (majority class, logistic regression).
4. Pipeline + candidate estimators (`DecisionTreeClassifier`).
5. `GridSearchCV` (stratified 5-fold, scoring = PR-AUC) on the training data; simplest tree within 1 SE of the best.
6. Diagnose under/overfitting with training vs validation curves, learning curve, calibration.
7. Freeze the model; choose thresholds **on validation data only**.
8. Evaluate **once** on the test set; report confusion matrix, several metrics, cost, subgroup error rates.

`total_score` is **not** a feature (it determines the grade → target leakage); `student_id` is only an identifier.

## Using the trained model

```python
import joblib, pandas as pd
tree = joblib.load("models/decision_tree.joblib")
X = pd.DataFrame([{"weekly_self_study_hours": 3.0, "attendance_percentage": 90, "class_participation": 6}])
p_fail = tree.predict_proba(X)[0, 1]
flag = p_fail >= 0.30
```

## Authors
MCS 306, 3rd Year – Section B, BS Mathematics with Specialization in Computer Science, Bulacan State University

- Calug, Adrian Samuel
- Hilario, Angelo
- Jaculina, Adrienne Christopher
- Mendoza, Clarence Dion Reigh
- Yumul, Aries
