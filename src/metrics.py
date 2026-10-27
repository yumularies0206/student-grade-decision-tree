import numpy as np


def confusion_counts(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    return tp, fn, fp, tn


def _div(num, den):
    return num / den if den > 0 else None


def classification_metrics(tp, fn, fp, tn):
    recall = _div(tp, tp + fn)
    specificity = _div(tn, tn + fp)
    return {
        "TP": tp, "FN": fn, "FP": fp, "TN": tn, "N": tp + fn + fp + tn,
        "accuracy": _div(tp + tn, tp + tn + fp + fn),
        "precision": _div(tp, tp + fp),
        "recall": recall,
        "specificity": specificity,
        "fpr": _div(fp, fp + tn),
        "fnr": _div(fn, fn + tp),
        "f1": _div(2 * tp, 2 * tp + fp + fn),
        "balanced_accuracy": (None if recall is None or specificity is None
                              else (recall + specificity) / 2),
    }


def total_cost(fn, fp, c_fn, c_fp):
    return c_fn * fn + c_fp * fp


def cost_threshold(c_fp, c_fn):
    return c_fp / (c_fp + c_fn)


def log_loss(y_true, p, eps=1e-15):
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def bayes_binary(prior_h, p_e_given_h, p_e_given_not_h):
    p_e = p_e_given_h * prior_h + p_e_given_not_h * (1.0 - prior_h)
    return p_e_given_h * prior_h / p_e


def gini(p):
    return 1.0 - p ** 2 - (1.0 - p) ** 2


def entropy(p):
    if p <= 0 or p >= 1:
        return 0.0
    return float(-(p * np.log2(p) + (1 - p) * np.log2(1 - p)))


def impurity_decrease(n_left, p_left, n_right, p_right, impurity=gini):
    n = n_left + n_right
    p_parent = (n_left * p_left + n_right * p_right) / n
    child = (n_left * impurity(p_left) + n_right * impurity(p_right)) / n
    return impurity(p_parent) - child
