from utils.utils import preprocess_data, calc_lambda_values, fill_na
from models.h_gru_model import hierarchical_GRU, hierarchical_predict
from resources.configs import h_gru_params as params, look_back, look_forward, train_test_split
import pandas as pd
import numpy as np
from tensorflow import set_random_seed
import random


#set random seed
pd.options.mode.chained_assignment = None
np.random.seed(1)
random.seed(2)
set_random_seed(3)

path='resources/cpi_us_dataset.csv'


data=preprocess_data(path, look_back, look_forward)
data=fill_na(data)
stats=calc_lambda_values(data)


model_tree=hierarchical_GRU(data, params, stats, train_test_split,top_indent=0)

per_product,per_horizon=hierarchical_predict(data, model_tree, train_test_split)

print(per_horizon)