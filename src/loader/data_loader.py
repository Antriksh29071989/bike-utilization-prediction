import logging

import pandas as pd
from pandas import DataFrame

logger = logging.getLogger(__name__)


def load_data_from_csv(csv_path: str) -> DataFrame:
    """
    :param csv_path: Path to csv file.
    :return: Pandas dataframe
    """
    try:
        logger.info(f"Reading CSV file from path {csv_path}")
        return pd.read_csv(csv_path)
    except FileNotFoundError:
        logger.error(f"File not found at path: {csv_path}")
        raise
    except Exception as e:
        logger.error(f"Manual intervention is required.: {e}")
        raise
