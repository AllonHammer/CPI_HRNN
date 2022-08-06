import math
import pandas as pd
import numpy as np
import datetime
from datetime import datetime as dt
import json
import os
from scipy.stats import pearsonr
from config import params, look_back, look_forward
from config import LOAD_DATA, LOAD_MODELS, MODELS_TORCH


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def handle_missing_values(data):
    """
    Fill empty values (some of the values we create) with mean
    :param data: DataFrame
    :return: data: DataFrame
    """
    inflation_cols = [col for col in data if col.startswith('Inflation t-')]
    # Fill t-1... t-n with the mean
    data[inflation_cols] = data[inflation_cols].apply(lambda row: row.fillna(row.mean()), axis=1)
    inflation_cols.append('Inflation t')
    # Drop the remainig (i.e. items that appeared for the first time)
    data.dropna(subset=inflation_cols, inplace=True)
    return data


def pre_process_data(path: str, back: int, forward: int) -> pd.DataFrame:
    """
    Pre process the data that we can consume the data for our models as time series data.
    This function gonna take the data (for example CPI data) and create series of data.
    for example: the vector of the data will look like: Xt-back, ..., Xt-2, Xt-1, Xt, Xt+1, Xt+2,..., Xt+forward
    Every time we want to create tne inflation rate by 100*log(Xt/Xt-1)
    -------------
    :param path: str
        the path of the data
    :param back: int
        number of periods that we take back. This will also be the input of the GRU
    :param forward: int
        number of periods we want to predict.
    -------------
    :return: DataFrame
        the result of the vectors that create the time series
    """
    data = pd.read_csv(path)
    data.sort_values(by=["Category_id", "Date"], ascending=[True, True], inplace=True)
    cols = []
    for t in range(back + 1):
        data["Inflation t-{}".format(t)] = 100 * np.log(
            data.groupby("Category_id").Price.shift(t) / data.groupby("Category_id").Price.shift(t + 1))
        if t == 0:
            cols.append('Inflation t')
        else:
            cols.append('Inflation t-{}'.format(t))
    for t in range(1, forward + 1):
        data["Inflation t+{}".format(t)] = 100 * np.log(
            data.groupby("Category_id").Price.shift(-t) / data.groupby("Category_id").Price.shift(-t + 1))
        cols.append('Inflation t+{}'.format(t))

    data.rename(columns={'Inflation t-0': 'Inflation t'}, inplace=True)
    data['Year'] = pd.DatetimeIndex(data['Date']).year
    order = ['Category_id', 'Category', 'Year', 'Date', 'Price'] + cols + ['Indent', 'Weight', 'Parent', 'Parent_ID']
    data = data[order]
    return handle_missing_values(data)


def split_train_test(data: pd.DataFrame, split_train_size: float = 0.7, window: float = None):
    """Split to train/test based on time (not just a random shuffle, we want to avoid data leak), and reshape for GRU
    :param data: pandas df
    :param split: float that represents wanted percent of rows in train
    :param window: float number that makes the windowing
    :return X_train, X_test: numpy array
    :return y_train,y_test: numpy array
    :return train,test: pandas df (we use this as well for the horizon prediction)
    """
    # Get the diff in days between max and min dates
    data['Date'] = data['Date'].astype(str)
    max_date = datetime.datetime.strptime(data["Date"].max(), "%d/%m/%Y")
    min_date = datetime.datetime.strptime(data["Date"].min(), "%d/%m/%Y")
    print(max_date, min_date)
    days_diff = (max_date - min_date).days
    # train/test cutoff point by date
    cutoff_date_test_min = max_date - datetime.timedelta(days=int(days_diff * (1 - split_train_size)))
    if window is not None:
        cutoff = []
        cutoff_date_test_max = cutoff_date_test_min + datetime.timedelta(days=int(days_diff * (window)))
        for date in data['Date']:
            date_in_datetime = datetime.datetime.strptime(date, "%d/%m/%Y")
            if cutoff_date_test_min < date_in_datetime <= cutoff_date_test_max:
                cutoff.append(1)
            elif date_in_datetime < cutoff_date_test_min:
                cutoff.append(0)
            else:
                cutoff.append(2)
        data['cutoff'] = cutoff
    else:
        data['cutoff'] = data['Date'].apply(
            lambda x: 1 if datetime.datetime.strptime(x, "%d/%m/%Y") > cutoff_date_test_min else 0)

    train = data[data['cutoff'] == 0]
    test = data[data['cutoff'] == 1]
    x_cols = [col for col in data if col.startswith('Inflation t-')]
    x_cols = x_cols[::-1]  # reverse order to have t-4, t-3, t-2, t-1
    x_train = train[x_cols].values
    y_train = train['Inflation t'].values
    x_test = test[x_cols].values
    y_test = test['Inflation t'].values
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
    return x_train, y_train, x_test, y_test, train, test


def prepare_predict(data: pd.DataFrame, add_unemployment_data=False):
    indents = pd.unique(data['Indent']).tolist()
    x_cols = [col for col in data if col.startswith('Inflation t-')]
    x_cols = x_cols[::-1]
    print(x_cols)
    if add_unemployment_data:
        x_cols += [col for col in data if col.startswith('un')]
        print(x_cols)
    y_cols = ['Inflation t'] + [col for col in data if col.startswith('Inflation t+')]  # order is t, t+1, t+2.. t+8
    prediction = {}
    predicts_per_horizon = {}
    return indents, prediction, predicts_per_horizon, x_cols, y_cols


def calc_lambda_values(df: pd.DataFrame) -> dict:
    """ calculates stats for each node with relation to its parent such as
        variance ratio, depth, and correlation.
        :param df: pandas df
        :returns dict <key: tuple (item, level) , value: dict of stats>

         """
    d = {}
    pairs = df[['Category_id', 'Indent']].drop_duplicates()
    for item in zip(pairs['Category_id'], pairs['Indent']):
        category, indent = item[0], item[1]
        df_child = df[(df['Indent'] == indent) & (df['Category_id'] == category)]
        if indent > 0:
            parent_id = df_child.iloc[0]['Parent_ID']
            # child_weight = df_child.iloc[0]['Weight']
            child_samples = df_child['Inflation t'].values
            df_parent = df[(df['Indent'] == indent - 1) & (df['Category_id'] == parent_id)]
            try:
                # parent_weight = df_parent.iloc[0]['Weight']
                parent_samples = df_parent['Inflation t'].values
            except IndexError:  # if for some reason the parent is not in indent-1 but higher in the tree
                print('Parent higher for {}'.format((parent_id, indent - 1)))
                df_parent = df[(df['Category_id'] == parent_id)]
                parent_weight = df_parent.iloc[0]['Weight']
                parent_samples = df_parent['Inflation t'].values
            child_samples = child_samples[~np.isnan(child_samples)]
            parent_samples = parent_samples[~np.isnan(parent_samples)]
            # n = np.clip(child_weight / parent_weight, 0, 1)
            # v = sigmoid(np.std(child_samples) / np.std(parent_samples))
            t = min(len(child_samples), len(parent_samples))
            try:
                s = pearsonr(child_samples[0:t], parent_samples[0:t])
            except:
                s = [0]
            d[item] = {'corr': np.round(s[0], 2)}
    return d


def save_results(date_train: datetime, prediction: dict, full_results: pd.DataFrame, indents_results: pd.DataFrame,
                 horizon_results: pd.DataFrame,
                 score_per_horizon: pd.DataFrame, category_results: pd.DataFrame, train_test_split, window, folder_name,
                 extra_str=""):
    now = dt.now()
    year, month, day, hour, minute = now.year, now.month, now.day, now.hour, now.minute
    time_now = f"{year}_{month}_{day}_{hour}_{minute}"
    folder_name = os.path.join(os.getcwd(), 'results', f'{folder_name}_{time_now}')
    if not os.path.exists(f'{folder_name}'):
        print(f"Create Folder {folder_name}")
        os.mkdir(f'{folder_name}')
    full_results.to_csv(os.path.join(folder_name, f'full_results{extra_str}.csv'))
    indents_results.to_csv(os.path.join(folder_name, f'indents_avg_rmse_results{extra_str}.csv'))
    horizon_results.to_csv(os.path.join(folder_name, f'horizon_avg_rmse_results{extra_str}.csv'))
    category_results.to_csv(os.path.join(folder_name, f'category_avg_rmse_results{extra_str}.csv'))
    score_per_horizon.to_csv(os.path.join(folder_name, f'score_per_horizon{extra_str}.csv'))
    with open(os.path.join(folder_name, f'prediction{extra_str}.json'), 'w') as f:
        json.dump(prediction, f)
    with open(os.path.join(folder_name, f'params{extra_str}.txt'), 'w') as f:
        f.write(f"train start at date: {date_train}" + '\n')
        f.write(f"train end at date: {now}" + '\n')
        f.write(str(params) + '\n')
        f.write(f'look_back: {look_back}\n')
        f.write(f'look_forward: {look_forward}\n')
        f.write(f'train_test_split: {train_test_split}\n')
        f.write(f'window: {window}\n')
        f.write(f'load data: {LOAD_DATA}\n')
        f.write(f'load models: {LOAD_MODELS}\n')
        if LOAD_MODELS:
            f.write(f'load models path: {MODELS_TORCH}\n')
