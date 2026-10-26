from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "student_performance.csv"
FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"
MODEL_DIR = ROOT / "models"

RANDOM_STATE = 11
TEST_SIZE = 0.15
VAL_SIZE = 0.15
CV_FOLDS = 5
TUNING_SUBSAMPLE = 100_000

FEATURES = ["weekly_self_study_hours", "attendance_percentage", "class_participation"]
FEATURE_LABELS = {
    "weekly_self_study_hours": "Weekly self-study hours",
    "attendance_percentage": "Attendance (%)",
    "class_participation": "Class participation (0-10)",
}

FAIL_GRADES = {"D", "F"}

COST_FN = 10.0
COST_FP = 2.0

PARAM_GRID = {
    "model__criterion": ["gini", "entropy"],
    "model__max_depth": [2, 3, 4, 5, 6, 8, 10, 12],
    "model__min_samples_leaf": [1, 50, 500, 2000],
}
TUNING_METRIC = "average_precision"
DEPTH_SWEEP = list(range(1, 16))
LEARNING_CURVE_SIZES = [2_000, 5_000, 10_000, 25_000, 50_000, 100_000, 250_000, None]
THRESHOLD_GRID = np.round(np.arange(0.01, 1.0, 0.01), 2)
