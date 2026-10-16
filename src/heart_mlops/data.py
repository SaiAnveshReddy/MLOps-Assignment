"""
Loading and cleaning for the Cleveland Heart Disease dataset.

Design note (why imputation instead of dropping rows): only 6 of the 303
Cleveland records have a missing value, all in `ca` or `thal`. Dropping them
would shrink an already small dataset by 2%, so I impute with the column
mode instead -- both columns are effectively categorical (ca: 0-3 vessels,
thal: 3/6/7 codes), so the mode is a defensible fill value and we keep every
patient record for the downstream CV folds.
"""

import pandas as pd

from heart_mlops.config import (
    BINARY_TARGET_COL,
    COLUMNS_WITH_MISSING,
    PROCESSED_DATA_PATH,
    RAW_COLUMNS,
    RAW_DATA_PATH,
    TARGET_COL,
)


def load_raw_data(path=RAW_DATA_PATH) -> pd.DataFrame:
    """Read the headerless UCI Cleveland file and attach column names."""
    df = pd.read_csv(path, header=None, names=RAW_COLUMNS, na_values="?")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values and collapse the target to a binary label.

    The raw `num` column is disease severity 0 (no disease) to 4 (most
    severe). The assignment asks for a binary target, so anything > 0
    becomes 1 ("disease present"), following the convention used in the
    original Cleveland study itself.
    """
    df = df.copy()

    for col in COLUMNS_WITH_MISSING:
        mode_value = df[col].mode(dropna=True).iloc[0]
        df[col] = df[col].fillna(mode_value)

    df[BINARY_TARGET_COL] = (df[TARGET_COL] > 0).astype(int)
    df = df.drop(columns=[TARGET_COL])

    # thal/ca are stored as floats in the raw file (e.g. 3.0) purely because
    # of how the sentinel "?" forced pandas to read the column as object;
    # cast back to int now that missing values are gone.
    df["ca"] = df["ca"].astype(int)
    df["thal"] = df["thal"].astype(int)

    return df


def load_clean_dataset(path=RAW_DATA_PATH) -> pd.DataFrame:
    return clean_data(load_raw_data(path))


if __name__ == "__main__":
    cleaned = load_clean_dataset()
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Wrote {len(cleaned)} cleaned rows to {PROCESSED_DATA_PATH}")
