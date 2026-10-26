import pandas as pd
from sklearn.model_selection import train_test_split
from . import config as C


def load_dataset() -> pd.DataFrame:
    if not C.DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found at {C.DATA_FILE}. See data/README.md for download steps."
        )
    print(f"[data] loading {C.DATA_FILE.name}")
    return pd.read_csv(C.DATA_FILE)


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["fail"] = df["grade"].isin(C.FAIL_GRADES).astype(int)
    return df


def quality_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in C.FEATURES + ["total_score"]:
        rows.append({"variable": col, "missing": int(df[col].isna().sum()),
                     "min": df[col].min(), "max": df[col].max(), "mean": df[col].mean(),
                     "std": df[col].std(), "median": df[col].median()})
    return pd.DataFrame(rows)


def three_way_split(df: pd.DataFrame):
    X, y = df[C.FEATURES], df["fail"]
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=C.TEST_SIZE, stratify=y, random_state=C.RANDOM_STATE)
    val_fraction = C.VAL_SIZE / (1.0 - C.TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=val_fraction, stratify=y_dev, random_state=C.RANDOM_STATE)
    return dict(X_dev=X_dev, y_dev=y_dev, X_train=X_train, y_train=y_train,
                X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test)
