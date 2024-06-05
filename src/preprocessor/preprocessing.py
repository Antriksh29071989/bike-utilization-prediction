import logging

import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def split_data(x: DataFrame, y: DataFrame):
    """ Split dataframe in 75-25 ratio
    :param x:Features
    :param y:Labels
    :return:
    """
    return train_test_split(x, y, test_size=0.25, random_state=24)


def perform_one_hot_encoding(data, column):
    data = pd.concat([data, pd.get_dummies(data[column], prefix=column, drop_first=True)], axis=1)
    return data.drop([column], axis=1)
