from utils.utils import preprocess_data, calc_corr, fill_na, top_5_neighbors
from models.vec_geu_model import calc_models, predict
from resources.configs import vec_gru_params as params, look_back, look_forward
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

corr=pd.read_csv('~/CPI/cpi_us/final/corr.csv', index_col=0)
#corr=calc_corr(data)

corr['top_5']=corr.apply(top_5_neighbors, axis=1)
k_neighbours_mapping=corr['top_5'].to_dict()

models=calc_models(data, k_neighbours_mapping, params)

per_product,per_horizon,_=predict(models)

print(per_horizon)