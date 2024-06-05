import logging
from typing import List

import pandas as pd
from pandas import DataFrame

logger = logging.getLogger(__name__)


def print_df_columns(df: DataFrame) -> None:
    with pd.option_context('display.max_columns', None):
        logger.info(df.head(5))


def rename_columns(df: DataFrame) -> None:
    df.rename(
        columns={'instant': 'unique_id', 'dteday': 'date', 'yr': 'year', 'mnth': 'month', 'hr': 'hour',
                 'weathersit': 'weather',
                 'hum': 'humidity', 'cnt': 'count', 'casual': 'unregistered'}, inplace=True)


def drop_columns(df: DataFrame, columns_to_drop: List[str]):
    """
    Drop specified columns from the DataFrame.

    Parameters:
    - df: DataFrame from which columns need to be dropped.
    - columns_to_drop: List of column names to be dropped.

    Returns:
    - DataFrame with specified columns dropped.
    """
    df.drop(columns=columns_to_drop, inplace=True)


def convert_to_category(df: DataFrame, columns: List[str]) -> DataFrame:
    for column in columns:
        df[column] = df[column].astype('category')
    return df
