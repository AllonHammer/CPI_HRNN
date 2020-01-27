import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import GRU
from keras.layers import Dense
from keras.initializers import glorot_uniform
from tensorflow import set_random_seed
import keras.backend as K
from keras.regularizers import l2
from scipy.stats import pearsonr, spearmanr
import math
import random
from keras.callbacks import EarlyStopping

from utils.utils import split_train_test

#set random seed

pd.options.mode.chained_assignment = None
np.random.seed(1)
random.seed(2)
set_random_seed(3)




def hierarchical_GRU(data, params, corr, train_test_split=0.7, top_indent=0, benchmark=False):
    """Train multiple hieratchical GRU models for each indent and category
    :param data: pandas df
    :param params: a dict of hyper-parameters for the model
    :param corr: dict. output of calc_lambda_values()
    :param train_test_split: int that represents wanted percent of rows in train
    :param top_indent: int that represents the highest level in the hierarchy
    :param benchmark: bool if True then do not utilize hierarchy of the model (alpha=0)

    :return model_tree: a nested dict {indent_ldevel : {category_id: (model, weight i.e. importance)}}
                        remember that an item can have diffrent parents and different indents across the years,
                        that's why we store a different model for each indent+category_id tuple
    """
    model_tree = {}
    indents = sorted(data['Indent'].value_counts().index.tolist())  # get all of the hierarchies in the data
    for indent in indents:  # iterate hierarchies
        model_tree[indent] = {}
        categories = data[data['Indent'] == indent][
            'Category_id'].value_counts().index.tolist()  # get all category_ids in the specific heirarchy
        for i, category in enumerate(categories):  # iterate categories
            print('Training indent {} category {}. Which is the {} out of {} categories in the layer'.format(indent,
                                                                                                             category,
                                                                                                             i + 1, len(
                    categories)))
            df = data[(data['Indent'] == indent) & (
            data['Category_id'] == category)]  # filter df for this indent and category
            if indent > 0:
                pearson = corr[(category, indent)]['corr']  # get the importance of the child
            else:
                pearson = 0
            parent_id = df.iloc[0]['Parent_ID']  # get the parent's category_id
            X_train, y_train, X_test, y_test, _, _ = split_train_test(df, train_test_split)
            model_tree = train_model(X_train, y_train, X_test, y_test, category, parent_id, indent, pearson,
                                      model_tree, params, top_indent, benchmark)  # update the tree with a new model
    return model_tree


def train_model(X_train, y_train, X_test, y_test, category, parent_id, indent, pearson, model_tree, params,
                 top_indent=0, benchmark=False):
    """Train a single hierarchical GRU for a given indent and category_id, add it to the model_tree.
    :param X_train: numpy array
    :param y_train: numpy array
    :param category: int that represents the category_id of the child
    :param parent_id: int that represents the category_id of the parent
    :param indent: int that represents the hierarchy of the child
    :param weight: float that represents the importance of the child
    :param model_tree: a nested dict {indent_ldevel : {category_id: (model, weight i.e. importance)}}
                       represents the current model tree, before adding the new model.
    :param params: a dict of hyperparameters for the model
    :param top_indent: int that represents the highest level in the hierarchy
    :param benchmark: boolean , if True then set alpha=0, essentially given us a normal (non-hierarchical GRU)

    :return model_tree: a nested dict {indent_ldevel : {category_id: (model, weight i.e. importance)}}
                        represents the new model tree after adding the new model.
    """
    model = Sequential()
    # get hyper params
    hidden_gru_units = params['hidden_gru_units']
    T, D = X_train.shape[1], X_train.shape[2]
    rec_reg, ker_reg = l2(params['recurrent_reg']), l2(params['kernel_reg'])
    init = glorot_uniform(seed=1)
    epochs, batch_size, verbose = params['epochs'], params['batch_size'], params['verbose']
    # initiate parent's weights
    parent_gru_layer = None

    if indent == top_indent:  # when training the top hierarchy we want a simple GRU with L2 regularization
        model.add(GRU(hidden_gru_units, activation='tanh',
                      input_shape=(T, D), kernel_initializer=init,
                      stateful=False, return_sequences=False,
                      recurrent_regularizer=rec_reg, kernel_regularizer=ker_reg))
        model.add(Dense(1, kernel_initializer=init))
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)

        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])

        model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), batch_size=batch_size,
                  callbacks=[es], verbose=verbose)
        model_tree[indent][category] = model  # add the (model,importance) tuple to the model tree
    else:  # now we want to add the hierarchy to the model: lambda(n)*||parent_kernel_weights-child_kernel_weights||^2
        if parent_id in model_tree[indent - 1].keys():
            parent_gru_layer = model_tree[indent - 1][parent_id].layers[
                0]  # get the parent GRU layer from the model in the model tree

        # If for some reason the parent does not exist in indent-1 we will start searching for it in higher hierarchies
        else:
            for j in range(1, indent):
                if parent_id in model_tree[indent - 1 - j].keys():
                    parent_gru_layer = model_tree[indent - 1 - j][parent_id].layers[0]
                    break
        # If we still can't find no parent, then assign the parent to be the All items (8106)
        if parent_gru_layer is None:
            parent_gru_layer = model_tree[0][8106].layers[0]

        parent_weights = concat_weights(parent_gru_layer)  # concat input_weights, recurrent_weights and bias
        alpha = params['alpha']

        if benchmark:
            n = 0
        elif params['smoothing'] == 'linear':
            n = alpha * np.clip(pearson, 0, 1)
        else:
            n = np.exp(alpha * pearson) - 1
        model.add(GRU(hidden_gru_units, activation='tanh',
                      input_shape=(T, D), kernel_initializer=init,
                      stateful=False, return_sequences=False))
        model.add(Dense(1, kernel_initializer=init))
        print('n is ', n)
        # train with the custom loss function
        model.compile(loss=custom_loss_wrapper(parent_weights, concat_weights(model.layers[0]), n), metrics=['mse'],
                      optimizer='adam')
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
        model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), batch_size=batch_size,
                  callbacks=[es], verbose=verbose)
        print(model.evaluate(X_test, y_test))
        model_tree[indent][category] = model  # add the (model,importance) tuple to the model tree
    return model_tree


def concat_weights(layer):
    """Concatnate the weights of the GRU layer.
    :param layer: a Keras GRU layer
    :return model_tree: a tensor in shape  (D+H+1,3H)
    """
    # There are D*H*3+H*H*3+1*H*3 variables in GRU (D=dimensions, H=hidden units)
    # The weight matrix is = [D X 3*H]
    # Kernel weights shape (D,3*H)--> wxx (D,H)   wxr (D, H)    wxh (D,H) stacked together to form (D, 3H)
    # Recurrent weights shape (H, 3*H)--> whx (H,H) whr (H,H) whh (H,H) stacked together to form (H,3H)
    # Bias weights shape= (3*H) --> bi (H,)  bc (H,)  bu (H,) stacked together to form (3H,)

    input_kernel = layer.weights[0]  # shape (D,3H)
    recurrent_kernel = layer.weights[1]  # shape (H,3H)
    bias = K.reshape(layer.weights[2], (1, -1))  # shape (1,3H)
    return K.concatenate([input_kernel, recurrent_kernel, bias], axis=0)  # altogether (D+H+1,3H)


def custom_loss_wrapper(parent_weights, son_weights, n):
    """Wrapper function for custom loss. Keras can only handle a loss function with 2 arguments (y_true, y_pred).
               We want to add more arguments, thus the wrapper.
    :param layer: parent_weights a tensor in shape (D+H+1,3H) representing the kernel weights  of the GRU layer of the parent
    :param layer: son_weights a tensor in shape (D+H+1,3H) representing the kernel weights  of the GRU layer of the child
    :param n: a float representing the ratio between the child's importance and the parent's importance
    :return custom_loss: a function that has two arguments and calls "hierarchical_RMSE" function.
    """
    def custom_loss(y_true, y_pred):
        return hierarchical_MSE(y_true, y_pred, parent_weights, son_weights, n)
    return custom_loss


def hierarchical_MSE(y_true, y_pred, parent_weights, son_weights, n):
    """calculate the custom loss function that is combined of a regular rmse and a special constraint explained below.

    :param y_true: tensor of size (batch_size,1)
    :param y_pred: tensor of size (batch_size,1)
    :param layer: parent_weights a tensor in shape (D+H+1,3H) representing the kernel weights  of the GRU layer of the parent
    :param layer: son_weights a tensor in shape (D+H+1,3H) representing the kernel weights  of the GRU layer of the child
    :param n: a float representing the ratio between the child's importance and the parent's importance

    :return custom_loss: a function that has two arguments and calls "hierarchical_MSE" function.
    """

    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)

    mse = K.mean(K.square(y_pred - y_true), keepdims=False)  # calcaulte the  mean squared error
    norm = K.sum(K.square(parent_weights - son_weights), keepdims=False)  # calculate sum of all indices in norm
    return n * norm + mse


def hierarchical_predict(data, model_tree, train_test_split=0.7):
    """This function iterate through the model tree and generates the RMSE for every horizon.

    :param data: pandas df
    :param model_tree: a nested dict {indent_level :
                                                    {category_id: (model, weight i.e. importance)}}
    :param train_test_split: int that represents wanted percent of rows in train

    :return: pd.DataFrame RMSE per product
    :return: pd.DataFrame RMSE per horizon

    """

    score_per_product = {'Category_id': [], 'Horizon': [], 'RMSE': []}
    score_per_horizon = {'Horizon': [], 'RMSE': []}
    x_cols = [col for col in data if col.startswith('Inflation t-')]
    x_cols = x_cols[::-1]  # reverse order to have t-4, t-3, t-2, t-1
    y_cols = ['Inflation t'] + [col for col in data if col.startswith('Inflation t+')]  # order is t, t+1, t+2.. t+8
    # iterate horizons
    for horizon in range(0, len(y_cols)):
        print('Currently in horizon {}'.format(horizon))
        y_col = y_cols[horizon]  # t+horizon
        indents = sorted(data['Indent'].value_counts().index.tolist())  # get all of the hierarchies in the data
        ys, yhats = np.array([]), np.array([])  # initiate arrays of y_true and y_hat for this horizion
        for indent in indents:
            categories = data[data['Indent'] == indent][
                'Category_id'].value_counts().index.tolist()  # get all of the categories in the specific hierarcy
            for i, category in enumerate(categories):
                print(
                    'Predicting indent {} category {}. Which is the {} out of {} categories in the layer'.format(indent,
                                                                                                                 category,
                                                                                                                 i + 1,
                                                                                                                 len(
                                                                                                                     categories)))
                df = data[(data['Indent'] == indent) & (
                data['Category_id'] == category)]  # filter df for specific (indent, category_id)
                _, _, _, _, _, test = split_train_test(df,
                                                       train_test_split)  # we will get the exact train/test split as before since we split based on time
                # drop (x,y) tuples with missing y
                df_test = pd.concat([test[x_cols], test[y_col]], axis=1)
                df_test.dropna(inplace=True)
                # reshape for GRU
                X_test = df_test.iloc[:, :-1].values
                y_test = df_test.iloc[:, -1:].values
                X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
                # Get rellevent model from model tree
                model = model_tree[indent][category]
                y_hat = model.predict(X_test)
                # Append to current horizon ys and yhats

                current_score = math.sqrt(mean_squared_error(y_test.reshape(-1), y_hat.reshape(-1)))
                score_per_product['Category_id'].append(category)
                score_per_product['Horizon'].append(horizon)
                score_per_product['RMSE'].append(current_score)
                ys = np.append(ys, y_test.reshape(-1))
                yhats = np.append(yhats, y_hat.reshape(-1))
        # calculate correlation once
        if horizon == 0:
            pearson = np.round(pearsonr(ys, yhats)[0], 2)
            spearman = np.round(spearmanr(ys, yhats)[0], 2)
            print('Correltaion between Y_test and Y_hat is, Pearson {}, Spearman {}'.format(pearson, spearman))


        score = math.sqrt(mean_squared_error(ys, yhats))  # score for horizon
        score_per_horizon['Horizon'].append(horizon)
        score_per_horizon['RMSE'].append(score)
        if X_test.shape[1] > 1:  # if there is more than one X (i.e t-1, t-2 ...)
            X_test = X_test[:, 1:,
                     :]  # remove the oldest time stamp from X and set the prediction as the latest time stamp
            X_test[:, -1, :] = y_hat
        else:  # if there is only one X (which is t-1) then simply replace it with the prediction
            X_test[:, 0, :] = y_hat
    return pd.DataFrame.from_dict(score_per_product), pd.DataFrame.from_dict(score_per_horizon)
