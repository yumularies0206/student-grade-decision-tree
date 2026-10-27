import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report

from src import config as C
from src import data as D
from src import metrics as M
from src import model as MD
from src import plots as P


def jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


def dump(obj, name):
    with open(C.RES_DIR / name, "w") as f:
        json.dump(obj, f, indent=2, default=jsonable)


def main():
    t0 = time.time()
    for d in (C.FIG_DIR, C.RES_DIR, C.MODEL_DIR):
        d.mkdir(exist_ok=True)

    df = D.add_target(D.load_dataset())
    D.quality_report(df).to_csv(C.RES_DIR / "dataset_summary.csv", index=False)
    summary = {
        "n_rows": int(len(df)), "n_duplicate_ids": int(df["student_id"].duplicated().sum()),
        "n_missing_cells": int(df.isna().sum().sum()),
        "grade_counts": df["grade"].value_counts().reindex(list("ABCDF")).to_dict(),
        "fail_rate": float(df["fail"].mean()),
        "share_total_score_100": float((df["total_score"] == 100).mean()),
        "corr_with_fail": df[C.FEATURES + ["total_score"]].corrwith(df["fail"]).to_dict(),
        "corr_with_total_score": df[C.FEATURES].corrwith(df["total_score"]).to_dict(),
        "median_study_hours": {"pass": float(df.loc[df.fail == 0, C.FEATURES[0]].median()),
                               "fail": float(df.loc[df.fail == 1, C.FEATURES[0]].median())},
        "cost_fn": C.COST_FN, "cost_fp": C.COST_FP, "random_state": C.RANDOM_STATE,
    }

    S = D.three_way_split(df)
    Xtr, ytr, Xva, yva, Xte, yte = (S[k] for k in ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"])
    summary.update(n_train=len(Xtr), n_val=len(Xva), n_test=len(Xte),
                   fail_rate_train=float(ytr.mean()), fail_rate_val=float(yva.mean()),
                   fail_rate_test=float(yte.mean()))
    print(f"[split] train={len(Xtr):,}  validation={len(Xva):,}  test={len(Xte):,}")

    P.fig_class_distribution(df); P.fig_feature_distributions(df); P.fig_boxplots(df)
    P.fig_correlation(df); P.fig_scatter(df); P.fig_gini_curve()
    P.fig_split_diagram(len(Xtr), len(Xva), len(Xte))

    best, tune_info, grid = MD.tune(Xtr, ytr)
    print("[tune] chosen:", best, "|", {k: round(v, 4) if isinstance(v, float) else v for k, v in tune_info.items()})
    grid.drop(columns=["params"]).to_csv(C.RES_DIR / "grid_search_results.csv", index=False)
    P.fig_grid_heatmap(grid)

    depth_df = MD.depth_diagnosis(Xtr, ytr, Xva, yva, {k: v for k, v in best.items() if k != "max_depth"})
    depth_df.to_csv(C.RES_DIR / "depth_diagnosis.csv", index=False)
    P.fig_depth_diagnosis(depth_df, best["max_depth"])
    lc = MD.learning_curve(Xtr, ytr, Xva, yva, best)
    lc.to_csv(C.RES_DIR / "learning_curve.csv", index=False)
    P.fig_learning_curve(lc, best["max_depth"])

    tree = MD.tree_pipeline(**best).fit(Xtr, ytr)
    p_tr, p_va, p_te = (tree.predict_proba(X)[:, 1] for X in (Xtr, Xva, Xte))
    joblib.dump(tree, C.MODEL_DIR / "decision_tree.joblib")

    scan = MD.threshold_scan(yva, p_va)
    scan.to_csv(C.RES_DIR / "threshold_scan_validation.csv", index=False)
    th = MD.select_thresholds(scan)
    print("[threshold] validation-selected:", {k: round(v, 3) for k, v in th.items()})
    P.fig_threshold_cost(scan, th)
    cal = MD.calibration_table(yva, p_va)
    cal.to_csv(C.RES_DIR / "calibration_validation.csv", index=False)
    P.fig_calibration(cal)

    maj = MD.majority_baseline().fit(Xtr, ytr)
    lr = MD.logistic_pipeline().fit(Xtr, ytr)
    lr_va, lr_te = lr.predict_proba(Xva)[:, 1], lr.predict_proba(Xte)[:, 1]
    lr_scan = MD.threshold_scan(yva, lr_va)
    lr_th = MD.select_thresholds(lr_scan)

    test = {
        "majority_class": MD.evaluate(yte, maj.predict_proba(Xte)[:, 1], 0.5),
        "logistic_regression_F1_threshold": MD.evaluate(yte, lr_te, lr_th["t_f1"]),
        "logistic_regression_cost_threshold": MD.evaluate(yte, lr_te, lr_th["t_cost"]),
        "tree_F1_threshold": MD.evaluate(yte, p_te, th["t_f1"]),
        "tree_cost_threshold": MD.evaluate(yte, p_te, th["t_cost"]),
        "tree_theory_threshold": MD.evaluate(yte, p_te, th["t_theory"]),
    }
    validation = {"tree_F1_threshold": MD.evaluate(yva, p_va, th["t_f1"]),
                  "tree_cost_threshold": MD.evaluate(yva, p_va, th["t_cost"])}
    train_eval = MD.evaluate(ytr, p_tr, th["t_f1"])

    n_te, pos = len(yte), int(yte.sum())
    costs = {"Flag nobody": M.total_cost(pos, 0, C.COST_FN, C.COST_FP) / n_te,
             "Flag everybody": M.total_cost(0, n_te - pos, C.COST_FN, C.COST_FP) / n_te,
             "Logistic regression (cost threshold)": test["logistic_regression_cost_threshold"]["cost_per_student"],
             "Decision tree (F1 threshold)": test["tree_F1_threshold"]["cost_per_student"],
             "Decision tree (cost threshold)": test["tree_cost_threshold"]["cost_per_student"]}

    ev = test["tree_F1_threshold"]
    bayes_precision = M.bayes_binary(prior_h=float(yte.mean()), p_e_given_h=ev["recall"], p_e_given_not_h=ev["fpr"])

    cvr = cross_validate(MD.tree_pipeline(**best), Xtr, ytr, cv=MD.stratified_cv(),
                         scoring=["average_precision", "roc_auc"], n_jobs=-1)
    cv_stab = {k.replace("test_", ""): [float(v.mean()), float(v.std())] for k, v in cvr.items() if k.startswith("test_")}

    imp = MD.importances(tree, Xte, yte)
    imp.to_csv(C.RES_DIR / "feature_importance.csv", index=False)
    sub = MD.subgroup_table(Xte, yte.values, p_te, Xtr, {"F1 threshold": th["t_f1"], "cost threshold": th["t_cost"]})
    sub.to_csv(C.RES_DIR / "subgroup_metrics.csv", index=False)

    P.fig_confusion(test["tree_F1_threshold"], test["tree_cost_threshold"])
    P.fig_roc_pr(yte.values, {"Decision tree": (p_te, P.PASS_COL), "Logistic regression": (lr_te, P.FAIL_COL)})
    P.fig_feature_importance(imp)
    shallow = MD.tree_pipeline(max_depth=3, criterion=best["criterion"]).fit(Xtr, ytr)
    P.fig_tree(shallow)
    P.fig_decision_surface(tree, df)
    P.fig_model_comparison({"Majority class": test["majority_class"],
                            "Logistic regression (F1 thr.)": test["logistic_regression_F1_threshold"],
                            "Tree (F1 thr.)": test["tree_F1_threshold"],
                            "Tree (cost thr.)": test["tree_cost_threshold"]})
    P.fig_cost_comparison(costs)

    root = MD.root_split_example(Xtr, ytr)
    n_leaves = int(tree.named_steps["model"].get_n_leaves())
    final_depth = int(tree.named_steps["model"].get_depth())
    t_ = tree.named_steps["model"].tree_
    internal = t_.feature[t_.children_left != -1]
    splits_by_feature = {f: int(np.sum(internal == i)) for i, f in enumerate(C.FEATURES)}
    dump({"best_params": best, "tuning": tune_info, "thresholds": th, "logistic_thresholds": lr_th,
          "depth": final_depth, "n_leaves": n_leaves}, "best_params.json")
    dump({"test": test, "validation": validation, "train_tree_F1_threshold": train_eval,
          "cost_per_student_test": costs, "cv_stability": cv_stab,
          "bayes_check": {"bayes_precision": bayes_precision, "empirical_precision": ev["precision"]},
          "depth": final_depth, "n_leaves": n_leaves, "splits_by_feature": splits_by_feature}, "metrics.json")
    summary["root_split_gini_example"] = root
    summary["runtime_seconds"] = round(time.time() - t0, 1)
    dump(summary, "data_summary.json")

    (C.RES_DIR / "tree_rules.txt").write_text(MD.export_rules(tree, th["t_f1"]))
    leaves = MD.leaf_table(tree, th["t_f1"]).rename(columns={"flag": "flag_F1_threshold"})
    leaves["flag_cost_threshold"] = leaves["p_fail"] >= th["t_cost"]
    leaves.to_csv(C.RES_DIR / "leaf_rules.csv", index=False)
    pd.DataFrame({k: v for k, v in test.items()}).T.drop(columns=["N"]).to_csv(C.RES_DIR / "test_metrics_table.csv")
    for key in ["tree_F1_threshold", "tree_cost_threshold"]:
        e = test[key]
        (C.RES_DIR / f"classification_report_{key}.txt").write_text(
            f"Threshold t = {e['threshold']:.3f}\nConfusion matrix (rows: actual positive, actual negative; "
            f"columns: predicted positive, predicted negative)\n[[{e['TP']}, {e['FN']}],\n [{e['FP']}, {e['TN']}]]\n\n" +
            classification_report(yte, (p_te >= e["threshold"]).astype(int), target_names=["Pass", "At-risk/Fail"], digits=4))
    out = Xte.head(500).copy()
    out["actual_fail"] = yte.head(500).values
    out["prob_fail"] = p_te[:500].round(4)
    out["flag_F1_threshold"] = (p_te[:500] >= th["t_f1"]).astype(int)
    out["flag_cost_threshold"] = (p_te[:500] >= th["t_cost"]).astype(int)
    out.to_csv(C.RES_DIR / "test_predictions_sample.csv", index=False)

    for key in ["tree_F1_threshold", "tree_cost_threshold"]:
        e = test[key]
        print(f"\n{key}: t = {e['threshold']:.2f}")
        print("Confusion matrix:\n", [[e['TP'], e['FN']], [e['FP'], e['TN']]])
        for k in ["accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc", "cost_per_student"]:
            print(f"{k}: {e[k]:.4f}" if e[k] is not None else f"{k}: undefined")
    print(f"\n[done] {summary['runtime_seconds']} s")


if __name__ == "__main__":
    main()
