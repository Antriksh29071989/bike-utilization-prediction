import os

import pandas as pd

from src.analyzer import data_analyzer
from src.analyzer.data_analyzer import create_point_plot
from src.utils import data_utils

test_dir = "../plots"
df = pd.read_csv("../test_data/test_hour.csv")
data_utils.rename_columns(df)
columns = ['unique_id', 'date']
data_utils.drop_columns(df=df, columns_to_drop=columns)

cols = ['season', 'month', 'hour', 'holiday', 'weekday', 'workingday', 'weather']
converted_df = data_utils.convert_to_category(df=df, columns=cols)


def test_plot_correlation():
    file_path = os.path.join(test_dir, 'correlation.png')
    data_analyzer.plot_correlation(df=converted_df, local_dir_path=file_path)
    assert os.path.exists(file_path)


def test_create_point_plot_registered():
    file_path = os.path.join(test_dir, 'registered_users.png')
    data_analyzer.create_point_plot(df=converted_df,
                                    local_dir_path=file_path,
                                    x_axis='hour',
                                    y_axis='registered',
                                    hue='weekday',
                                    title='Count of bikes during weekdays and weekends: Registered users')
    assert os.path.exists(file_path)


def test_create_point_plot_unregistered():
    file_path = os.path.join(test_dir, 'unregistered_users.png')
    create_point_plot(df=converted_df,
                      local_dir_path=file_path,
                      x_axis='hour',
                      y_axis='unregistered',
                      hue='weekday',
                      title='Count of bikes during weekdays and weekends: Unregistered users')
    assert os.path.exists(file_path)


def test_create_point_plot_count_weekday():
    file_path = os.path.join(test_dir, 'users_count_weekday.png')
    data_analyzer.create_point_plot(df=converted_df,
                                    local_dir_path=file_path,
                                    x_axis='hour',
                                    y_axis='count',
                                    hue='weekday',
                                    title='Weekly bike usage')
    assert os.path.exists(file_path)


def test_create_bar_plot_months():
    file_path = os.path.join(test_dir, 'bikes_monthly_count.png')
    data_analyzer.plot_bar_count(df=converted_df,
                                 local_dir_path=file_path,
                                 x_axis='month',
                                 y_axis='count',
                                 title='Count of bikes during different months')
    assert os.path.exists(file_path)


def test_create_bar_plot_weekday():
    file_path = os.path.join(test_dir, 'bikes_day_count.png')
    data_analyzer.plot_bar_count(df=converted_df,
                                 local_dir_path=file_path,
                                 x_axis='weekday',
                                 y_axis='count',
                                 title='Count of bikes during different days')
    assert os.path.exists(file_path)
