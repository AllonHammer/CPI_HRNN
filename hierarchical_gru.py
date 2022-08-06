from datetime import datetime as dt
from utils import pre_process_data, calc_lambda_values, save_results
from model import hierarchical_GRU, save_models, hierarchical_predict_create_window, \
    calculate_results_to_dataframe, summary_results
from config import params, look_back, look_forward, train_test_split, window
from config import MODEL_DATA_PATH_CPI, LOAD_MODELS, TRAIN_NAME, SAVE_MODELS
import pandas as pd
import numpy as np
import torch
import random

# set random seed
pd.options.mode.chained_assignment = None
np.random.seed(1)
random.seed(2)
torch.random.manual_seed(3)


def data_prepare():
    data = pre_process_data(MODEL_DATA_PATH_CPI, look_back, look_forward)
    return data


def predict_with_load_model(data, split_size, window_size, extra_str=''):
    print("Load Model And Predict")
    prediction, score_per_horizon = hierarchical_predict_create_window(data,
                                                                       model_tree=None,
                                                                       train_test_split=split_size,
                                                                       load_save_model=LOAD_MODELS,
                                                                       window=window_size,
                                                                       train_name=TRAIN_NAME + extra_str)
    return prediction, score_per_horizon


def predict_from_scratch(data, params, stats, split_size, window_size, extra_str=''):
    print("Create And Train Models And Predict")
    model_tree = hierarchical_GRU(data, params, stats, split_size,
                                  top_indent=0, benchmark=params['benchmark'], window=window_size)
    prediction, score_per_horizon = hierarchical_predict_create_window(data, model_tree, split_size,
                                                                       load_save_model=LOAD_MODELS, window=window_size,
                                                                       train_name=TRAIN_NAME)
    return prediction, score_per_horizon, model_tree


def calc_metrics(date_train, prediction, score_per_horizon, split_size, window_size, look_forward, extra_str=''):
    results = calculate_results_to_dataframe(prediction, num_horizons=look_forward)
    indent_results, horizon_avg_results, category_results = summary_results(results)

    save_results(date_train, prediction, results, indent_results, horizon_avg_results, score_per_horizon,
                 category_results, split_size, window_size, f"{TRAIN_NAME}_{split_size}", extra_str)


if __name__ == '__main__':
    date_train = dt.now()
    data = data_prepare()
    stats = calc_lambda_values(data)
    for split in train_test_split:
        extra_str = f'_{split}_win_{window}'
        if LOAD_MODELS:
            prediction, score_per_horizon = predict_with_load_model(data, split_size=split, window_size=window,
                                                                    extra_str=extra_str)
        else:
            prediction, score_per_horizon, model_tree = predict_from_scratch(data, params, stats, split,
                                                                             window_size=window,
                                                                             extra_str=extra_str)
            if SAVE_MODELS:
                save_models(model_tree, f"{TRAIN_NAME}_{split}")
        calc_metrics(date_train, prediction, score_per_horizon, split, window, look_forward, extra_str)
