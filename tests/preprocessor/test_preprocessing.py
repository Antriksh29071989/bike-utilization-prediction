import pandas as pd

from src.preprocessor.preprocessing import split_data


def test_split_data():
    df = pd.read_csv("../test_data/test_hour.csv")
    x = df.drop(columns=['cnt'], axis=1)
    y = df['cnt']
    x_train, x_test, y_train, y_test = split_data(x, y)

    # Check the sizes of the splits
    assert len(x_train) == 749
    assert len(x_test) == 250
    assert len(y_train) == 749
    assert len(y_test) == 250

    # Check that the split is reproducible
    processed_x_train, processed_x_test, processed_y_train, processed_y_test = split_data(x, y)
    assert x_train.equals(processed_x_train)
    assert x_test.equals(processed_x_test)
    assert y_train.equals(processed_y_train)
    assert y_test.equals(processed_y_test)
