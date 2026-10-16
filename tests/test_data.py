from heart_mlops.config import RAW_DATA_PATH
from heart_mlops.data import clean_data, load_clean_dataset, load_raw_data


def test_raw_data_loads_with_expected_shape():
    df = load_raw_data(RAW_DATA_PATH)
    assert df.shape == (303, 14)
    assert "num" in df.columns


def test_raw_data_has_question_mark_missing_values():
    df = load_raw_data(RAW_DATA_PATH)
    assert df["ca"].isna().sum() > 0
    assert df["thal"].isna().sum() > 0


def test_clean_data_has_no_missing_values():
    raw = load_raw_data(RAW_DATA_PATH)
    cleaned = clean_data(raw)
    assert cleaned.isna().sum().sum() == 0


def test_clean_data_produces_binary_target():
    cleaned = load_clean_dataset()
    assert set(cleaned["target"].unique()) <= {0, 1}
    assert "num" not in cleaned.columns


def test_clean_data_preserves_row_count():
    cleaned = load_clean_dataset()
    assert len(cleaned) == 303


def test_ca_and_thal_are_integer_after_cleaning():
    cleaned = load_clean_dataset()
    assert cleaned["ca"].dtype.kind == "i"
    assert cleaned["thal"].dtype.kind == "i"
