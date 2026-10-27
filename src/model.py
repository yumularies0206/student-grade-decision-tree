import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, average_precision_score

from . import config as C
from . import metrics as M


def tree_pipeline(**params):
    return Pipeline([("model", DecisionTreeClassifier(random_state=C.RANDOM_STATE, **params))])


def logistic_pipeline():
    return Pipeline([("scale", StandardScaler()),
                     ("model", LogisticRegression(max_iter=2000, random_state=C.RANDOM_STATE))])


def majority_baseline():
    return DummyClassifier(strategy="most_frequent")


def stratified_cv():
    return StratifiedKFold(n_splits=C.CV_FOLDS, shuffle=True, random_state=C.RANDOM_STATE)


def tune(X_train, y_train):
    if len(X_train) > C.TUNING_SUBSAMPLE:
        X_sub, _, y_sub, _ = train_test_split(X_train, y_train, train_size=C.TUNING_SUBSAMPLE,
                                              stratify=y_train, random_state=C.RANDOM_STATE)
    else:
        X_sub, y_sub = X_train, y_train
    print(f"[tune] GridSearchCV on {len(X_sub):,} training rows, {C.CV_FOLDS}-fold stratified CV")
    search = GridSearchCV(tree_pipeline(), C.PARAM_GRID, scoring=C.TUNING_METRIC,
                          cv=stratified_cv(), n_jobs=-1, refit=False, return_train_score=True)
    search.fit(X_sub, y_sub)
    res = pd.DataFrame(search.cv_results_)
    top = res.sort_values("mean_test_score", ascending=False).iloc[0]
    se = top["std_test_score"] / np.sqrt(C.CV_FOLDS)
    ok = res[res["mean_test_score"] >= top["mean_test_score"] - se]
    chosen = ok.sort_values(["param_model__max_depth", "param_model__min_samples_leaf",
                             "mean_test_score"], ascending=[True, False, False]).iloc[0]
    params = {"criterion": chosen["param_model__criterion"],
              "max_depth": int(chosen["param_model__max_depth"]),
              "min_samples_leaf": int(chosen["param_model__min_samples_leaf"])}
    info = {"best_mean_cv_score": float(top["mean_test_score"]), "standard_error": float(se),
            "chosen_mean_cv_score": float(chosen["mean_test_score"]),
            "n_configurations": int(len(res)), "n_fits": int(len(res) * C.CV_FOLDS),
            "tuning_rows": int(len(X_sub)), "scoring": C.TUNING_METRIC}
    res = res.rename(columns=lambda c: c.replace("param_model__", "param_"))
    return params, info, res


def threshold_scan(y_val, p_val):
    rows = []
    for t in C.THRESHOLD_GRID:
        tp, fn, fp, tn = M.confusion_counts(y_val, (p_val >= t).astype(int))
        m = M.classification_metrics(tp, fn, fp, tn)
        rows.append({"threshold": float(t), "precision": m["precision"], "recall": m["recall"],
                     "f1": m["f1"], "flagged_share": (tp + fp) / m["N"],
                     "total_cost": M.total_cost(fn, fp, C.COST_FN, C.COST_FP),
                     "cost_per_student": M.total_cost(fn, fp, C.COST_FN, C.COST_FP) / m["N"]})
    return pd.DataFrame(rows)


def _middle_of_best(values, maximise=True):
    v = np.nan_to_num(np.asarray(values, dtype=float), nan=-np.inf if maximise else np.inf)
    best = v.max() if maximise else v.min()
    idx = np.flatnonzero(np.isclose(v, best, rtol=0, atol=1e-12))
    return int(idx[len(idx) // 2])


def select_thresholds(scan):
    t_f1 = float(scan.threshold.iloc[_middle_of_best(scan.f1.fillna(-1), True)])
    t_cost = float(scan.threshold.iloc[_middle_of_best(scan.total_cost, False)])
    return {"t_f1": t_f1, "t_cost": t_cost,
            "t_theory": M.cost_threshold(C.COST_FP, C.COST_FN)}


def evaluate(y, p, t):
    y = np.asarray(y)
    tp, fn, fp, tn = M.confusion_counts(y, (p >= t).astype(int))
    out = M.classification_metrics(tp, fn, fp, tn)
    out.update(threshold=float(t),
               roc_auc=float(roc_auc_score(y, p)),
               pr_auc=float(average_precision_score(y, p)),
               log_loss=M.log_loss(y, p),
               total_cost=M.total_cost(fn, fp, C.COST_FN, C.COST_FP))
    out["cost_per_student"] = out["total_cost"] / out["N"]
    return out


def depth_diagnosis(X_train, y_train, X_val, y_val, base_params):
    rows = []
    for d in C.DEPTH_SWEEP:
        pipe = tree_pipeline(**{**base_params, "max_depth": d}).fit(X_train, y_train)
        ptr, pva = pipe.predict_proba(X_train)[:, 1], pipe.predict_proba(X_val)[:, 1]
        rows.append({"max_depth": d,
                     "train_log_loss": M.log_loss(y_train, ptr), "val_log_loss": M.log_loss(y_val, pva),
                     "train_pr_auc": average_precision_score(y_train, ptr),
                     "val_pr_auc": average_precision_score(y_val, pva)})
    df = pd.DataFrame(rows)
    df["generalization_gap"] = df["val_log_loss"] - df["train_log_loss"]
    return df


def learning_curve(X_train, y_train, X_val, y_val, params):
    rows = []
    for n in C.LEARNING_CURVE_SIZES:
        if n is None or n >= len(X_train):
            Xs, ys = X_train, y_train
        else:
            Xs, _, ys, _ = train_test_split(X_train, y_train, train_size=n, stratify=y_train,
                                            random_state=C.RANDOM_STATE)
        pipe = tree_pipeline(**params).fit(Xs, ys)
        ptr, pva = pipe.predict_proba(Xs)[:, 1], pipe.predict_proba(X_val)[:, 1]
        rows.append({"n_train": len(Xs),
                     "train_log_loss": M.log_loss(ys, ptr), "val_log_loss": M.log_loss(y_val, pva),
                     "train_pr_auc": average_precision_score(ys, ptr),
                     "val_pr_auc": average_precision_score(y_val, pva)})
    return pd.DataFrame(rows)


def calibration_table(y, p, edges=(0, .05, .1, .2, .3, .4, .5, .6, .8, 1.0001)):
    y, p = np.asarray(y), np.asarray(p)
    b = np.digitize(p, edges[1:-1])
    rows = []
    for k in range(len(edges) - 1):
        m = b == k
        if m.sum() == 0:
            continue
        rows.append({"bin": f"[{edges[k]:.2f}, {min(edges[k + 1], 1):.2f})", "n": int(m.sum()),
                     "mean_predicted": float(p[m].mean()), "observed_rate": float(y[m].mean())})
    return pd.DataFrame(rows)


def subgroup_table(X_test, y_test, p_test, X_train, thresholds):
    rows = []
    for col in ["attendance_percentage", "class_participation"]:
        lo, hi = np.quantile(X_train[col], [1 / 3, 2 / 3])
        groups = {"low": X_test[col] <= lo, "middle": (X_test[col] > lo) & (X_test[col] <= hi),
                  "high": X_test[col] > hi}
        for gname, mask in groups.items():
            yy, pp = y_test[mask.values], p_test[mask.values]
            for op, t in thresholds.items():
                tp, fn, fp, tn = M.confusion_counts(yy, (pp >= t).astype(int))
                m = M.classification_metrics(tp, fn, fp, tn)
                rows.append({"variable": col, "group": gname, "operating_point": op, "n": m["N"],
                             "actual_fail_rate": (tp + fn) / m["N"], "flag_rate": (tp + fp) / m["N"],
                             "precision": m["precision"], "recall": m["recall"],
                             "specificity": m["specificity"]})
    return pd.DataFrame(rows)


def importances(pipe, X_te, y_te):
    tree = pipe.named_steps["model"]
    imp = pd.DataFrame({"feature": C.FEATURES, "label": [C.FEATURE_LABELS[f] for f in C.FEATURES],
                        "gini_importance": tree.feature_importances_})
    sub = X_te.sample(min(100_000, len(X_te)), random_state=C.RANDOM_STATE)
    pi = permutation_importance(pipe, sub, y_te.loc[sub.index], scoring="average_precision",
                                n_repeats=10, random_state=C.RANDOM_STATE, n_jobs=-1)
    imp["permutation_importance"] = pi.importances_mean
    imp["permutation_std"] = pi.importances_std
    return imp


def root_split_example(X_train, y_train):
    stump = DecisionTreeClassifier(max_depth=1, criterion="gini", random_state=C.RANDOM_STATE).fit(X_train, y_train)
    t = stump.tree_
    j, s = int(t.feature[0]), float(t.threshold[0])
    left = X_train.iloc[:, j].values <= s
    y = np.asarray(y_train)
    n_l, n_r = int(left.sum()), int((~left).sum())
    p_l, p_r, p_0 = y[left].mean(), y[~left].mean(), y.mean()
    out = {"feature": C.FEATURES[j], "threshold": s, "n_root": len(y), "n_left": n_l, "n_right": n_r,
           "p_fail_root": float(p_0), "p_fail_left": float(p_l), "p_fail_right": float(p_r),
           "gini_root": M.gini(p_0), "gini_left": M.gini(p_l), "gini_right": M.gini(p_r)}
    out["gini_weighted_children"] = (n_l * out["gini_left"] + n_r * out["gini_right"]) / len(y)
    out["impurity_decrease"] = M.impurity_decrease(n_l, p_l, n_r, p_r)
    out["sklearn_root_impurity"] = float(t.impurity[0])
    return out


def export_rules(pipe, threshold):
    t = pipe.named_steps["model"].tree_
    lines = []

    def rec(node, depth):
        pad = "|   " * depth
        if t.children_left[node] == -1:
            v = t.value[node][0]
            p = v[1] / v.sum()
            lines.append(f"{pad}--> n={int(t.n_node_samples[node]):,}  P(Fail)={p:.3f}  => "
                         f"{'FLAG (at-risk)' if p >= threshold else 'ok'}")
            return
        f, thr = C.FEATURES[t.feature[node]], t.threshold[node]
        lines.append(f"{pad}if {f} <= {thr:.2f}:")
        rec(t.children_left[node], depth + 1)
        lines.append(f"{pad}else:  # {f} > {thr:.2f}")
        rec(t.children_right[node], depth + 1)

    rec(0, 0)
    return "\n".join(lines)


def leaf_table(pipe, threshold):
    t = pipe.named_steps["model"].tree_
    rows = []

    def rec(node, bounds):
        if t.children_left[node] == -1:
            v = t.value[node][0]
            p = v[1] / v.sum()
            conds = []
            for f, (lo, hi) in bounds.items():
                if lo == -np.inf:
                    conds.append(f"{f} <= {hi:.2f}")
                elif hi == np.inf:
                    conds.append(f"{f} > {lo:.2f}")
                else:
                    conds.append(f"{lo:.2f} < {f} <= {hi:.2f}")
            rows.append({"rule": " AND ".join(conds) or "(all students)",
                         "n_students": int(t.n_node_samples[node]), "p_fail": float(p),
                         "flag": bool(p >= threshold)})
            return
        f, thr = C.FEATURES[t.feature[node]], float(t.threshold[node])
        lo, hi = bounds.get(f, (-np.inf, np.inf))
        rec(t.children_left[node], {**bounds, f: (lo, min(hi, thr))})
        rec(t.children_right[node], {**bounds, f: (max(lo, thr), hi)})

    rec(0, {})
    return pd.DataFrame(rows)
