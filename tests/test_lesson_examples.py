import math
from src import metrics as M


def close(a, b, tol=1e-4):
    return abs(a - b) < tol


def test_example_12_8_shuttle_metrics():
    m = M.classification_metrics(tp=24, fn=16, fp=6, tn=154)
    assert close(m["accuracy"], 0.89) and close(m["precision"], 0.80)
    assert close(m["recall"], 0.60) and close(m["specificity"], 0.9625)
    assert close(m["fpr"], 0.0375) and close(m["fnr"], 0.40)
    assert close(m["f1"], 48 / 70) and close(m["balanced_accuracy"], 0.78125)


def test_example_13_1_two_thresholds():
    y = [1, 0, 1, 0, 1, 0, 1, 0]
    p = [.92, .81, .72, .65, .61, .42, .31, .10]
    for t, (tp, fp, fn, tn) in {0.5: (3, 2, 1, 2), 0.7: (2, 1, 2, 3)}.items():
        pred = [int(v >= t) for v in p]
        assert M.confusion_counts(y, pred) == (tp, fn, fp, tn)


def test_example_15_1_cost_and_threshold():
    assert M.total_cost(fn=16, fp=6, c_fn=10, c_fp=2) == 172
    assert close(M.cost_threshold(c_fp=2, c_fn=10), 1 / 6)
    assert close(M.cost_threshold(c_fp=3, c_fn=12), 0.20)


def test_section_16_5_models_a_and_b():
    a = M.classification_metrics(tp=16, fn=4, fp=20, tn=60)
    b = M.classification_metrics(tp=12, fn=8, fp=5, tn=75)
    assert close(a["precision"], 16 / 36) and close(b["precision"], 12 / 17)
    assert M.total_cost(4, 20, 10, 2) == 80 and M.total_cost(8, 5, 10, 2) == 90


def test_practice_problems_p26_p27():
    m = M.classification_metrics(tp=18, fn=12, fp=6, tn=64)
    assert close(m["accuracy"], 0.82) and close(m["precision"], 0.75)
    assert close(m["recall"], 0.60) and close(m["specificity"], 64 / 70)
    assert close(m["f1"], 2 / 3) and close(m["fpr"], 6 / 70) and close(m["balanced_accuracy"], 0.7571)


def test_undefined_ratio_convention():
    assert M.classification_metrics(0, 5, 0, 95)["precision"] is None


def test_log_loss_values_from_notes():
    assert close(M.log_loss([1], [0.9]), -math.log(0.9), 1e-9)
    assert close(M.log_loss([1], [0.01]), -math.log(0.01), 1e-9)


def test_bayes_binary_chapter_8():
    assert close(M.bayes_binary(0.01, 0.90, 0.05), 0.154, 1e-3)


def test_gini_properties():
    assert M.gini(0.0) == 0.0 and M.gini(1.0) == 0.0 and close(M.gini(0.5), 0.5)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
