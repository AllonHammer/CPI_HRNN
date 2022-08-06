import math
import os
import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR
from utils import split_train_test
from sklearn.metrics import mean_squared_error
from model.gru_model import GRUNet
from pytorchtools import EarlyStopping, CustomLoss

# set random seed
pd.options.mode.chained_assignment = None
np.random.seed(1)
random.seed(2)
torch.random.manual_seed(3)


def hierarchical_GRU(data, params, corr, train_test_split=0.7, top_indent=0, benchmark=False, window=None):
    """Train multiple hieratchical GRU models for each indent and category
    :param data: pandas df
    :param params: a dict of hyper-parameters for the model
    :param corr: dict. output of calc_lambda_values()
    :param train_test_split: int that represents wanted percent of rows in train
    :param top_indent: int that represents the highest level in the hierarchy
    :param benchmark: bool if True then do not utilize hierarchy of the model (alpha=0)

    :return model_tree: a nested dict {indent_level : {category_id: (model, weight i.e. importance)}}
                        remember that an item can have diffrent parents and different indents across the years,
                        that's why we store a different model for each indent+category_id tuple
    """
    model_tree = {}
    indents = sorted(data['Indent'].value_counts().index.tolist())
    # iterate hierarchies
    for indent in indents:
        model_tree[indent] = {}
        categories = data[data['Indent'] == indent]['Category_id'].value_counts().index.tolist()
        for i, category in enumerate(categories):
            print(f'Training indent {indent} category {category}. Which is {i + 1} out of {len(categories)}'
                  f' categories in the layer')
            df = data[data['Category_id'] == category]
            if indent > 0:
                pearson = corr[(category, indent)]['corr']
            else:
                pearson = 0
            parent_id = df.iloc[df.shape[0] - 1]['Parent_ID']
            X_train, y_train, X_test, y_test, _, _ = split_train_test(df, train_test_split, window=window)

            print("Catrgory", category)
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
    :param model_tree: a nested dict {indent_level : {category_id: (model, weight i.e. importance)}}
                       represents the current model tree, before adding the new model.
    :param params: a dict of hyperparameters for the model
    :param top_indent: int that represents the highest level in the hierarchy
    :param benchmark: boolean , if True then set alpha=0, essentially given us a normal (non-hierarchical GRU)

    :return model_tree: a nested dict {indent_level : {category_id: (model, weight i.e. importance)}}
                        represents the new model tree after adding the new model.
    """
    hidden_gru_units = params["hidden_gru_units"]

    number_of_inputs_back, number_of_features = X_train.shape[1], X_train.shape[2]
    lr, epochs, batch_size, verbose, num_layers, patience = params['lr'], params['epochs'], params['batch_size'], \
                                                            params['verbose'], params['num_layers'], params['patience']
    is_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if is_cuda else torch.device("cpu")

    # initial the GRU neural network
    net = GRUNet(input_size=number_of_features, hidden_size=hidden_gru_units,
                 num_layers=num_layers, output_dim=number_of_features, drop_prob=0.0)

    if top_indent == indent:
        parent_weights = None
        son_weights = None
    else:
        parent_gru_layer = find_parent_model(model_tree, indent, parent_id)
        # concat input_weights, recurrent_weights and bias
        parent_weights = concat_weights(parent_gru_layer)
        son_weights = concat_weights(layer=net)
    alpha = params['alpha']
    if benchmark:
        n = 0
    elif params['smoothing'] == 'linear':
        n = alpha * np.clip(pearson, 0, 1)
    elif top_indent == indent:
        n = 0
    else:
        n = np.exp(alpha + pearson)
    print(f"value of n is: {n}, value of pearson is: {pearson}")

    early_stopping = EarlyStopping(patience=patience, verbose=False)
    train_model_loops(net, X_train, y_train, X_test, y_test, indent, top_indent, is_cuda, device, early_stopping,
                      epochs=epochs, batch_size=batch_size, model_type="GRU", parent_weights=parent_weights,
                      son_weights=son_weights, n=n, lr=lr)
    model_tree[indent][category] = net  # add the (model,importance) tuple to the model tree
    return model_tree


def train_model_loops(net, X_train, y_train, X_test, y_test, indent, top_indent, train_on_gpu, device, early_stopping,
                      lr, epochs=10,
                      batch_size=10, model_type="GRU", parent_weights=None, son_weights=None, n=None, seq_length=4,
                      clip=5, print_every=10):
    ''' Training a network

        Arguments
        ---------

        net: CharRNN network
        data: text data to train the network
        epochs: Number of epochs to train
        batch_size: Number of mini-sequences per mini-batch, aka batch size
        seq_length: Number of character steps per mini-batch
        lr: learning rate
        clip: gradient clipping
        val_frac: Fraction of data to hold out for validation
        print_every: Number of steps for printing training and validation loss

    '''
    train_set = TabularDataset(X_train, y_train)
    valid_set = TabularDataset(X_test, y_test)
    train_set_generator = DataLoader(train_set, batch_size)
    valid_set_generator = DataLoader(valid_set, batch_size)

    # to track the training loss as the model trains
    train_losses = []
    train_losses_mse = []

    # to track the validation loss as the model trains
    valid_losses = []
    valid_losses_mse = []

    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    avg_mse_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []
    avg_mse_valid_losses = []
    # moved the parameters of the network to double
    net = net.double()
    # create the optimum for the network
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    # create the loss function depend on the indent of the node
    mse_criterion = nn.MSELoss()
    if indent == top_indent:
        criterion = nn.MSELoss()
    else:
        criterion = CustomLoss(parent_weights, son_weights, n)
    if train_on_gpu:
        net.cuda()
    counter = 0
    for e in range(epochs):
        net.train()  # reset to train mode after iterationg through validation data
        h = net.init_hidden(batch_size, device)
        for inputs, targets in train_set_generator:
            if inputs.shape[0] != batch_size:
                h = net.init_hidden(inputs.shape[0], device)
            counter += 1
            # print(inputs[1])
            # inputs, targets = torch.from_numpy(x), torch.from_numpy(y)
            if train_on_gpu:
                inputs, targets = inputs.cuda(), targets.cuda()
            if model_type == "GRU":
                h = h.data
            else:
                h = tuple([e.data for e in h])

            net.zero_grad()

            # print(inputs.shape, inputs[0])
            output, h = net(inputs, h)
            if top_indent == indent:
                loss = criterion(output, targets)
            else:
                loss = criterion(output, targets, concat_weights(layer=net))
            loss_mse = mse_criterion(output, targets)
            loss.backward()
            opt.step()
            train_losses.append(loss.item())
            train_losses_mse.append(loss_mse.item())

        val_h = net.init_hidden(batch_size, device)
        valid_losses_mse = []
        net.eval()
        for inputs, targets in valid_set_generator:
            # x, y = torch.from_numpy(x), torch.from_numpy(y)
            if inputs.shape[0] != batch_size:
                val_h = net.init_hidden(inputs.shape[0], device)
            if model_type == "GRU":
                val_h = val_h.data
            else:
                val_h = tuple([e.data for e in val_h])

            # inputs, targets = x, y
            if train_on_gpu:
                inputs, targets = inputs.cuda(), targets.cuda()

            output, val_h = net(inputs, val_h)
            if top_indent == indent:
                val_loss = criterion(output, targets)
            else:
                val_loss = criterion(output, targets, concat_weights(layer=net))
            val_mse_loss = mse_criterion(output, targets)
            valid_losses.append(val_loss.item())
            valid_losses_mse.append(val_mse_loss.item())

        avg_train_losses.append(np.average(train_losses))
        avg_mse_train_losses.append(np.average(train_losses_mse))
        avg_valid_losses.append(np.average(valid_losses))
        avg_mse_valid_losses.append(np.average(valid_losses_mse))

        print("Epoch: {}/{}...".format(e + 1, epochs),
              "Step: {}...".format(counter),
              "Loss: {:.5f}...".format(avg_train_losses[-1]),
              "Train Mse Loss: {:.5f}...".format(avg_mse_train_losses[-1]),
              "Val Loss: {:.5f}".format(avg_valid_losses[-1]),
              'Mse Val Loss: {:.5f}'.format(avg_mse_valid_losses[-1]))

        # clear lists to track next epoch
        train_losses = []
        train_losses_mse = []
        valid_losses = []
        valid_losses_mse = []
        early_stopping(avg_mse_valid_losses[-1], net)

        # if e % 5 == 0:
        #     scheduler.step()
        if early_stopping.early_stop:
            # print("schhhhm", scheduler.get_last_lr())
            print("Early stopping")
            break


def save_models(model_tree, train_name: str):
    for indent in model_tree:
        for category_id in model_tree[indent]:
            if not os.path.exists(os.path.join(os.getcwd(), 'torch_models', train_name)):
                os.mkdir(os.path.join(os.getcwd(), 'torch_models', train_name))
            torch.save(model_tree[indent][category_id].state_dict(),
                       os.path.join(os.getcwd(), 'torch_models', train_name, f"model_{indent}_{category_id}.pt"))


def load_model_by_indent_and_category(indent: int, category: int, train_name: str):
    model = GRUNet(input_size=1, hidden_size=50, output_dim=1, num_layers=2)
    model_path = os.path.join(os.getcwd(), 'torch_models', train_name, f"model_{indent}_{category}.pt")
    return model.load_state_dict(torch.load(model_path))


def find_parent_model(model_tree, indent, parent_id):
    parent_gru_layer = None
    if parent_id in model_tree[indent - 1].keys():
        parent_gru_layer = model_tree[indent - 1][parent_id]
    # If for some reason the parent does not exist in indent-1 we will start searching for it in higher hierarchies
    else:
        for j in range(1, indent):
            if parent_id in model_tree[indent - 1 - j].keys():
                parent_gru_layer = model_tree[indent - 1 - j][parent_id]
                break
    # If we still can't find no parent, then assign the parent to be the All items (8106)
    if parent_gru_layer is None:
        parent_gru_layer = model_tree[0][8106]
    return parent_gru_layer


class TabularDataset(torch.utils.data.Dataset):

    def __init__(self, features, labels):
        self.features = torch.Tensor(features).to(torch.double)
        self.labels = torch.Tensor(labels).to(torch.double)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.features)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        X = self.features[index, :]

        # Load data and get label
        y = self.labels[index]

        return X, y


def get_batches(x, y, batch_size):
    length = (x.shape[0] // batch_size)
    for n in range(length):
        yield x[n * batch_size:(n + 1) * batch_size], y[n * batch_size:(n + 1) * batch_size]


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
    first_input_kernel = layer.gru.weight_ih_l0.permute(1, 0)  # shape (D,3H)
    second_input_kernel = layer.gru.weight_ih_l1.permute(1, 0)  # shape (D,3H)
    first_recurrent_kernel = layer.gru.weight_hh_l0.permute(1, 0)  # shape (H,3H)
    second_recurrent_kernel = layer.gru.weight_hh_l1.permute(1, 0)  # shape (H,3H)
    first_bias_input = layer.gru.bias_ih_l0.view(1, -1)  # shape (1,3H)
    second_bias_input = layer.gru.bias_ih_l1.view(1, -1)  # shape (1,3H)
    return torch.cat((first_input_kernel, second_input_kernel, first_recurrent_kernel, second_recurrent_kernel,
                      first_bias_input, second_bias_input), 0)  # altogether (D+H+1,3H)
    # return torch.cat((first_input_kernel, first_recurrent_kernel, first_bias_input), 0)  # altogether (D+H+1,3H)


def hierarchical_predict_create_window(data, model_tree, train_test_split=0.7, load_save_model=False, window=None,
                                       train_name=None):
    """This function iterate through the model tree and generates the RMSE for every horizon.

    :param data: pandas df
    :param model_tree: a nested dict {indent_level :
                                                    {category_id: (model, weight i.e. importance)}}
    :param train_test_split: int that represents wanted percent of rows in train

    :return: pd.DataFrame RMSE per product
    :return: pd.DataFrame RMSE per horizon

    """
    prediction = {}
    score_per_horizon = {'Horizon': [], 'RMSE': []}
    x_cols = [col for col in data if col.startswith('Inflation t-')]
    x_cols = x_cols[::-1]
    y_cols = ['Inflation t'] + [col for col in data if col.startswith('Inflation t+')]  # order is t, t+1, t+2.. t+8
    indents = sorted(pd.unique(data.Indent))
    predicts_per_horizon = {}
    for indent in indents:
        categories = pd.unique(data[data['Indent'] == indent]['Category_id']).tolist()
        for i, category in enumerate(categories):
            h = None
            print(f'Predicting indent {indent} category {category}. {i + 1} out of {len(categories)}'
                  f' categories in the layer')
            df = data[data['Category_id'] == category]
            _, _, _, _, _, test = split_train_test(df, train_test_split, window=window)
            df_x_test = test[x_cols]
            df_x_test.dropna(inplace=True)
            for horizon in range(0, len(y_cols)):
                if f'{(indent, category)}' not in prediction:
                    prediction[f'{(indent, category)}'] = {}
                if horizon not in prediction[f'{(indent, category)}']:
                    prediction[f'{(indent, category)}'][horizon] = {'y_test': [], 'y_hat': []}
                if horizon not in predicts_per_horizon:
                    predicts_per_horizon[horizon] = {'ys': [], 'yhat': []}
                print('Currently in horizon {}'.format(horizon))
                y_col = y_cols[horizon]
                df_y_test = test[y_col]
                df_y_test.dropna(inplace=True)
                if df_y_test.shape[0] < 1:
                    break
                if horizon == 0:
                    X_test = df_x_test.values
                    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
                else:
                    X_test[:, :-1, :] = X_test[:, 1:, :]
                    X_test[:, -1, :] = y_hat.reshape(-1, 1)
                y_test = df_y_test.values
                X_test = X_test[:y_test.shape[0], :, :]
                y_hat, h = model_predication(X_test, category, indent, load_save_model, model_tree, h, y_test,
                                             train_name)
                prediction[f'{(indent, category)}'][horizon]["y_hat"] += y_hat.tolist()
                prediction[f'{(indent, category)}'][horizon]["y_test"] += y_test.tolist()
                predicts_per_horizon[horizon]['ys'] += y_test.reshape(-1).tolist()
                predicts_per_horizon[horizon]['yhat'] += y_hat.reshape(-1).tolist()
    for horizon in predicts_per_horizon:
        score = math.sqrt(
            mean_squared_error(predicts_per_horizon[horizon]['ys'], predicts_per_horizon[horizon]['yhat']))
        score_per_horizon['Horizon'].append(horizon)
        score_per_horizon['RMSE'].append(score)
    return prediction, pd.DataFrame.from_dict(score_per_horizon)


def calculate_results_to_dataframe(predication_dict: dict, num_horizons: int) -> pd.DataFrame:
    results = {'category': [], 'indent': []}
    for key in predication_dict:
        indent, category = eval(key)
        results['category'].append(category)
        results['indent'].append(indent)
        for horizon in range(num_horizons + 1):
            if f'horizon {horizon}' not in results:
                results[f'horizon {horizon}'] = []
                results[f'num samples {horizon}'] = []
            try:
                y_test = predication_dict[f'{(indent, category)}'][horizon]['y_test']
                y_hat = predication_dict[f'{(indent, category)}'][horizon]['y_hat']
                score_rmse = math.sqrt(mean_squared_error(y_test, y_hat))
                results[f'horizon {horizon}'].append(score_rmse)
                results[f'num samples {horizon}'].append(len(y_test))
            except Exception as e:
                print('error', str(e))
                results[f'horizon {horizon}'].append(None)
                results[f'num samples {horizon}'].append(None)
    return pd.DataFrame.from_dict(results)


def summary_results(predication_df: pd.DataFrame):
    horizon_cols = [col for col in predication_df.columns if col.startswith('horizon')]
    return predication_df.groupby("indent")[horizon_cols].mean(), predication_df[horizon_cols].mean(), \
           predication_df.groupby("category")[horizon_cols].mean()


def model_predication(X_test, category, indent, load_save_model, model_tree, h, y_test, train_name: str):
    if load_save_model:
        model = GRUNet(input_size=1, hidden_size=20, output_dim=1, num_layers=2).double()
        model_path = os.path.join(os.getcwd(), 'torch_models', train_name, f"model_{indent}_{category}.pt")
        model.load_state_dict(torch.load(model_path))
    else:
        model = model_tree[indent][category]
    model.eval()
    inputs, targets = torch.from_numpy(X_test), torch.from_numpy(y_test)
    if h is None:
        h = model.init_hidden(X_test.shape[0], "cpu")
    if X_test.shape[0] != h.shape[1]:
        h = h[:, :X_test.shape[0], :]
    h = h.data
    y_hat, h = model(inputs, h)
    y_hat = y_hat.detach().numpy()
    return y_hat, h
