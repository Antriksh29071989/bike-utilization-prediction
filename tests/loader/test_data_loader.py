import pytest
from pandas import DataFrame

from src.loader.data_loader import load_data_from_csv
from unittest.mock import patch


def test_load_data_from_csv():
    csv_path = "../test_data/test_hour.csv"
    df = load_data_from_csv(csv_path)

    assert isinstance(df, DataFrame)
    assert not df.empty


def test_load_data_from_csv_file_not_found():
    csv_path = "nonexistent_file.csv"

    with patch("builtins.open", side_effect=FileNotFoundError):
        with pytest.raises(FileNotFoundError):
            load_data_from_csv(csv_path)
