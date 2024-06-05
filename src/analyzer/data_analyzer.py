import os

import seaborn as sns
from matplotlib import pyplot as plt
from pandas import DataFrame


def _is_dir_exists(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def plot_correlation(df: DataFrame, local_dir_path: str):
    """ Plot correlation matrix for a given dataframe.
    :param df: Pandas Dataframe
    :param local_dir_path: Directory path to save the plot.
    :return:
    """
    corr = df.corr()
    plt.figure(figsize=(18, 8))
    sns.heatmap(corr, annot=True, annot_kws={'size': 10})
    _is_dir_exists(local_dir_path)
    plt.savefig(local_dir_path)
    plt.close()


def create_point_plot(df: DataFrame, local_dir_path: str, x_axis: str, y_axis: str, hue: str, title: str):
    """
    Create point plot based on x-axis and y-axis.
   :param df: Dataframe to plot the data.
   :param local_dir_path: Save plot on local.
   :param x_axis: x-axis column name.
   :param y_axis: y_axis column name.
   :param hue:Group data with different colors.
   :param title:Tile of the plot.
    """
    fig, ax = plt.subplots(figsize=(20, 5))
    sns.pointplot(data=df, x=x_axis, y=y_axis, hue=hue, ax=ax)
    ax.set(title=title)
    _is_dir_exists(local_dir_path)
    plt.savefig(local_dir_path)
    plt.close()


def plot_bar_count(df: DataFrame, local_dir_path: str, x_axis: str, y_axis: str, title: str):
    """
    Create bar plot based on x-axis and y-axis.
   :param df: Dataframe to plot the data.
   :param local_dir_path: Save plot on local.
   :param x_axis: x-axis column name.
   :param y_axis: y_axis column name.
   :param hue:Group data with different colors.
   :param title:Tile of the plot.
    """
    fig, ax = plt.subplots(figsize=(20, 5))
    sns.barplot(data=df, x=x_axis, y=y_axis, ax=ax)
    ax.set(title=title)
    _is_dir_exists(local_dir_path)
    plt.savefig(local_dir_path)
    plt.close()
