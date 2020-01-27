h_gru_params={'hidden_gru_units' : 50 , 'epochs' :10000, 'batch_size' : 50,  'verbose' : 0, 'recurrent_reg': 0.05, 'kernel_reg': 0.1,
       'smoothing': 'exp', 'alpha':1.5}
vec_gru_params={'hidden_gru_units' : 50 , 'epochs' :10000, 'batch_size' : 50,  'verbose' : 0, 'recurrent_reg': 0.1, 'kernel_reg': 0.1}
look_back = 4
look_forward = 8
train_test_split = 0.8