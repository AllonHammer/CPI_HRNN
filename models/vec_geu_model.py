import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import GRU
from keras.layers import Dense
from keras.initializers import glorot_uniform
from keras.regularizers import l2
from scipy.stats import pearsonr, spearmanr
import math
from keras.callbacks import EarlyStopping
from keras import optimizers
from utils.utils import split_train_test_vec

def vec_gru(X_train, X_test, y_train, y_test, params):
    """ Model with one layer of vec GRU (Samples, Time stamps, Categories) and one Dense layer (1)
    :param X_train np.array
    :param X_test np.array
    :param y_train np.array
    :param y_test np.array
    :param: params a dict of hyperparameters for the model

    :return keras.models.Sequential().  A single model for a category"""

    hidden_gru_units = params['hidden_gru_units']
    rec_reg, ker_reg = l2(params['recurrent_reg']), l2(params['kernel_reg'])
    epochs = params['epochs']
    batch_size = params['batch_size']
    verbose = params['verbose']
    init = glorot_uniform(seed=1)
    T = X_train.shape[1]
    D = X_train.shape[2]
    model = Sequential()
    model.add(GRU(hidden_gru_units, activation='tanh',
                  input_shape=(T, D), kernel_initializer=init,
                  stateful=False, return_sequences=False,
                  recurrent_regularizer=rec_reg, kernel_regularizer=ker_reg))
    model.add(Dense(1, kernel_initializer=init))
    opt = optimizers.Adam(lr=0.005)
    model.compile(optimizer=opt, loss='mean_squared_error', metrics=['mse'])
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=2, patience=50)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose, validation_data=(X_test, y_test),
              callbacks=[es])

    return model


def calc_models(df, k_neighbours_mapping, params):
    """ Trains a vec-gru model for each category based on it's K most correlated neighbours

    :param df pd.dataFrame()
    :param k_neighbours_mapping dict <key: category_id, value: list of top 5 neighbours>
    :param params: a dict of hyperparameters for the model

    :return dict <key: category_id, value: dict with train, test, X_train, X_test, y_train, y_test, and keras.models.Sequential() """

    models = {}
    for i, k in enumerate(k_neighbours_mapping.keys()):
        print('************************************************')
        print(
            'Training model for category {}. Which is {} out of {}'.format(k, i + 1, len(k_neighbours_mapping.keys())))
        categories = k_neighbours_mapping[k]
        curr_df = df[df['Category_id'].isin(categories)]

        try:
            train, test, X_train, X_test, y_train, y_test = split_train_test_vec(curr_df, split=0.8, y_category_id=k)
            model = vec_gru(X_train, X_test, y_train, y_test, params)
            models[k] = {'train': train, 'test': test, 'X_train': X_train, 'X_test': X_test, 'y_train': y_train,
                     'y_test': y_test, 'model': model}
        except (ValueError, TypeError):
            continue
    return models



def predict(models):
    """This function iterates through all the categories models and generates the RMSE for each horizon.

    :param models: dict <key: category_id, value: keras.models.Sequential()>

    :return: pd.DataFrame RMSE per product
    :return: pd.DataFrame RMSE per horizon
    :return: dict <key: horizon value:
                                       <key: category, value: # of predictions>>

    """

    score_per_category = {'Category_id': [], 'Horizon': [], 'RMSE': []}
    score_per_horizon = {'Horizon': [], 'RMSE': []}
    x_cols = [col for col in models[8106]['train'] if col.startswith('Inflation t-')]
    x_cols = x_cols[::-1]  # reverse order to have t-4, t-3, t-2, t-1
    y_cols = ['Inflation t'] + [col for col in models[8106]['train'] if
                                col.startswith('Inflation t+')]  # order is t, t+1, t+2.. t+8
    predictions_per_product = {0: {}, 1: {}, 2: {}, 3: {}, 4: {}, 5: {}, 6: {}, 7: {}, 8: {}}
    # iterate horizons
    for horizon in range(0, len(y_cols)):
        print('Currently in horizon {}'.format(horizon))
        y_col = y_cols[horizon]  # t+horizon
        ys, yhats = np.array([]), np.array([])  # initiate arrays of y_true and y_hat for this horizion
        for category in models.keys():
            print('Currently in category {}'.format(category))
            test = models[category][
                'test']  # we will get the exact train/test split as before since we split based on time
            # drop (x,y) tuples with missing y
            df_test = pd.concat([test[x_cols + ['Category_id', 'Date']], test[y_col]], axis=1)
            df_test.dropna(inplace=True)
            # prepare to reshape X_test
            df_test['rnk'] = df_test.groupby('Category_id')['Date'].rank()
            test_rnks = df_test['rnk'].unique().tolist()
            df_test.sort_values(by=['rnk', 'Category_id'], axis=0, ascending=[True, True], inplace=True)
            # y_test is simply in shape (Samples,1,1), we are predicting a single category
            y_test = df_test[df_test['Category_id'] == category][y_col]
            lst = []
            # Reshape X_test to (Samples, Time stamps, Categories)
            for rnk in test_rnks:
                lst.append(df_test[df_test['rnk'] == rnk][x_cols].values.T)
            try:
                X_test = np.stack(lst)
            except ValueError:
                continue

            # Get rellevent model from model tree
            model = models[category]['model']
            y_hat = model.predict(X_test)

            # Append to current horizon ys and yhats

            current_score = math.sqrt(mean_squared_error(y_test.reshape(-1), y_hat.reshape(-1)))
            score_per_category['Category_id'].append(category)
            score_per_category['Horizon'].append(horizon)
            score_per_category['RMSE'].append(current_score)
            ys = np.append(ys, y_test)
            yhats = np.append(yhats, y_hat)
            predictions_per_product[horizon][category] = (y_test, y_hat)
        # calculate correlation once
        if horizon == 0:
            pearson = np.round(pearsonr(ys, yhats)[0], 2)
            spearman = np.round(spearmanr(ys, yhats)[0], 2)
            print('Correltaion between Y_test and Y_hat is, Pearson {}, Spearman {}'.format(pearson, spearman))
            dfg = pd.DataFrame(columns=['y_true', 'y_hat'])
            dfg['y_true'] = ys
            dfg['y_hat'] = yhats
            dfg.to_csv('/home/alon/Downloads/vecGRUresults.csv', index=False)
            # plot_pair(x=range(len(ys)), y1=ys,y2=yhats, label1='y_true',label2='y_hat', title="", xlabel='sample', ylabel='inflation', dpi=100)
        score = math.sqrt(mean_squared_error(ys, yhats))  # score for horizon

        score_per_horizon['Horizon'].append(horizon)
        score_per_horizon['RMSE'].append(score)
        if X_test.shape[1] > 1:  # if there is more than one X (i.e t-1, t-2 ...)
            X_test = X_test[:, 1:,
                     :]  # remove the oldest time stamp from X and set the prediction as the latest time stamp
            X_test[:, -1, :] = y_hat
        else:  # if there is only one X (which is t-1) then simply replace it with the prediction
            X_test[:, 0, :] = y_hat
    return pd.DataFrame.from_dict(score_per_category), pd.DataFrame.from_dict(score_per_horizon), predictions_per_product




