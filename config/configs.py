benchmark = False

h_gru_params = {'hidden_gru_units': 10, 'epochs': 10000, 'batch_size': 50, 'verbose': 1,
                'recurrent_reg': 0.05, 'kernel_reg': 0.1, "num_layers": 2, "lr": 0.0001,
                'smoothing': 'exp', 'benchmark': benchmark, 'alpha': 1.5, 'patience': 10}


look_back = 4
look_forward = 8

train_test_split = [0.7]
window = 0.3

MODEL_NAME = "HRNN 4"
TRAIN_NAME = f'{MODEL_NAME}_lr_{h_gru_params["lr"]}_hidden_{h_gru_params["hidden_gru_units"]}_batch_{h_gru_params["batch_size"]}_patience_{h_gru_params["patience"]}'

MODEL_DATA_PATH_CPI = "data/cpi/cpi_us_dataset.csv"
MODELS_TORCH = 'torch_models'
BASELINES_MODELS = False
LOAD_DATA = True
LOAD_MODELS = False
SAVE_MODELS = False

