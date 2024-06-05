import logging.config

from src.analyzer import data_analyzer
from src.loader import data_loader
from src.model import model_builder
from src.preprocessor import preprocessing
from src.utils import data_utils

logging.config.fileConfig('log_conf.yaml', disable_existing_loggers=False)

# Loading
logging.info("Loading data from CSV...")
bike_hourly_df = data_loader.load_data_from_csv(csv_path="bike_sharing_dataset/hour.csv")
data_utils.print_df_columns(bike_hourly_df)

# Preprocessing
logging.info("Rename columns...")
data_utils.rename_columns(bike_hourly_df)
data_utils.print_df_columns(bike_hourly_df)

# removing redundant columns
columns = ['unique_id', 'date']
logging.info(f"Removing columns {columns}")
data_utils.drop_columns(df=bike_hourly_df, columns_to_drop=columns)
data_utils.print_df_columns(bike_hourly_df)
logging.debug(bike_hourly_df.info())

# Exploratory Data Analysis
cols = ['season', 'month', 'hour', 'holiday', 'weekday', 'workingday', 'weather']
filtered_df = data_utils.convert_to_category(df=bike_hourly_df, columns=cols)
logging.debug(filtered_df.info())


data_analyzer.plot_correlation(df=filtered_df, local_dir_path="plots/correlation.png")

data_analyzer.create_point_plot(df=filtered_df,
                                local_dir_path="plots/registered_users.png",
                                x_axis='hour',
                                y_axis='registered',
                                hue='weekday',
                                title='Count of bikes during weekdays and weekends: Registered users')

data_analyzer.create_point_plot(df=filtered_df,
                                local_dir_path="plots/unregistered_users.png",
                                x_axis='hour',
                                y_axis='unregistered',
                                hue='weekday',
                                title='Count of bikes during weekdays and weekends: Unregistered users')

data_analyzer.create_point_plot(df=filtered_df,
                                local_dir_path="plots/users_count_weekday.png",
                                x_axis='hour',
                                y_axis='count',
                                hue='weekday',
                                title='Weekly bike usage')

data_analyzer.plot_bar_count(df=filtered_df,
                             local_dir_path="plots/bikes_monthly_count.png",
                             x_axis='month',
                             y_axis='count',
                             title='Count of bikes during different months')

data_analyzer.plot_bar_count(df=filtered_df,
                             local_dir_path="plots/bikes_day_count.png",
                             x_axis='weekday',
                             y_axis='count',
                             title='Count of bikes during different days')


# Feature engineering
cols = ['season', 'month', 'hour', 'holiday', 'weekday', 'workingday', 'weather']
for col in cols:
    filtered_df = preprocessing.perform_one_hot_encoding(filtered_df, col)
x = filtered_df.drop(columns=['atemp', 'windspeed', 'unregistered', 'count'], axis=1)
y = filtered_df['count']

# Model training and evaluation
x_train, x_test, y_train, y_test = preprocessing.split_data(x, y)

mean_deviation = model_builder.train_model(x_train, y_train, x_test, y_test, "model/bike_sharing.joblib")
logging.info(f'Mean Absolute Deviation: {mean_deviation}')
