import math
import pandas as pd
import numpy as np
import datetime
from scipy.stats import pearsonr

def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def preprocess_data(cpi_path, look_back, look_forward):
    """ Preprocess the data into time steps of inflation rate .. X(t-3), X(t-2), X(t-1), X(t), X(t+1), X(t+2) ..
    Inflation rate is 100*log(X(t)/X(t-1))

    :param cpi_path: string
        path to data
    :param look_back: int
        The time dimension for the GRU
    :param look_back: int
        The forecasting horizon
    :return: pandas df
    """

    # read cpi data and sort by category and date
    cpi_data = pd.read_csv(cpi_path)
    cpi_data.sort_values(by=['Category_id', 'Date'], axis=0, ascending=[True, True], inplace=True)
    cpi_data.reset_index(inplace=True)

    # calculate inflation and shift back "look_back" times and shift forward "look_forward" times
    cols = []
    for i in range(0, look_back + 1):
        cpi_data['Inflation t-{}'.format(i)] = \
            100 * np.log(cpi_data.groupby(['Category_id'])['Price'].shift(i) / \
                         cpi_data.groupby(['Category_id'])['Price'].shift(i + 1))
        if i == 0:
            cols.append('Inflation t')
        else:
            cols.append('Inflation t-{}'.format(i))
    for i in range(1, look_forward + 1):
        cpi_data['Inflation t+{}'.format(i)] = \
            100 * np.log(cpi_data.groupby(['Category_id'])['Price'].shift(-i) / \
                         cpi_data.groupby(['Category_id'])['Price'].shift(-i + 1))
        cols.append('Inflation t+{}'.format(i))

    # Rename, and reorder the cols
    cpi_data.rename(columns={'Inflation t-0': 'Inflation t'}, inplace=True)
    cpi_data['Year'] = cpi_data['Date'].apply(lambda x: int(x[0:4]))
    order = ['Category_id', 'Category', 'Year', 'Date', 'Price'] + cols + ['Indent', 'Weight', 'Parent', 'Parent_ID']
    cpi_data = cpi_data[order]

    return cpi_data


def fill_na(data):
    """Fill empty values of X(t-n)... X(t-1) with mean
    :param data: pandas df
    :return: pandas df
    """

    inflation_cols = [col for col in data if col.startswith('Inflation t-')]
    # Fill t-1... t-n with the mean
    data[inflation_cols] = data[inflation_cols].apply(lambda row: row.fillna(row.mean()), axis=1)
    inflation_cols.append('Inflation t')
    # Drop the remainig (i.e. items that appeared for the first time )
    data.dropna(subset=inflation_cols, inplace=True)
    return data


def split_train_test(data, split=0.7):
    """Split to train/test based on time (not just a random shuffle, we want to avoid data leak), and reshape for GRU
    :param data: pandas df
    :param split: int that represents wanted percent of rows in train
    :return X_train, X_test: numpy array
    :return y_train,y_test: numpy array
    :return train,test: pandas df (we use this as well for the horizon prediction)
    """
    # Get the diff in days between max and min dates
    max_date = datetime.datetime.strptime(data.Date.max(), "%Y-%m-%d")
    min_date = datetime.datetime.strptime(data.Date.min(), "%Y-%m-%d")
    days_diff = (max_date - min_date).days
    # train/test cutoff point by date
    cutoff_date = max_date - datetime.timedelta(days=int(days_diff * (1 - split)))
    data['cutoff'] = data['Date'].apply(lambda x: 1 if datetime.datetime.strptime(x, "%Y-%m-%d") > cutoff_date else 0)
    train = data[data['cutoff'] == 0]
    test = data[data['cutoff'] == 1]

    x_cols = [col for col in data if col.startswith('Inflation t-')]
    x_cols = x_cols[::-1]  # reverse order to have t-4, t-3, t-2, t-1
    X_train = train[x_cols].values
    y_train = train['Inflation t'].values
    X_test = test[x_cols].values
    y_test = test['Inflation t'].values
    # Reshape to GRU (#samples, #time stamps, #features)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    return X_train, y_train, X_test, y_test, train, test


def split_train_test_vec(data, split=0.7, y_category_id=8106):
    """Split to train/test based on time (not just a random shuffle, we want to avoid data leak), and reshape for Vec GRU
    :param data: pandas df
    :param split: int that represents wanted percent of rows in train
    :param y_category: int that represents the category_id we want to predict

    :return X_train, X_test: numpy array
    :return y_train,y_test: numpy array
    :return train,test: pandas df (we this as well for the horizon prediction)
    """

    train = pd.DataFrame(columns=data.columns)
    test = pd.DataFrame(columns=data.columns)
    # a list of all categories in data
    all_categories = data['Category_id'].unique().tolist()

    for category in all_categories:
        df = data[data['Category_id'] == category]
        # Get the diff in days between max and min dates
        max_date = datetime.datetime.strptime(df.Date.max(), "%Y-%m-%d")
        min_date = datetime.datetime.strptime(df.Date.min(), "%Y-%m-%d")
        days_diff = (max_date - min_date).days
        # train/test cutoff point by date
        cutoff_date = max_date - datetime.timedelta(days=int(days_diff * (1 - split)))
        df['cutoff'] = df['Date'].apply(lambda x: 1 if datetime.datetime.strptime(x, "%Y-%m-%d") > cutoff_date else 0)
        # append all categories together
        train = pd.concat([train, df[df['cutoff'] == 0]], ignore_index=True)
        test = pd.concat([test, df[df['cutoff'] == 1]], ignore_index=True)
    # Now the shape is (Samples, Timestamps), where Samples is a combination of different Categories in different months

    # Unfortunatley some categories have more samples and some have less, and when we will reshape to 3d we will get null values
    # Therefore we will cutoff some samples from categories with too many months. The cutoffpoint will be the Mode
    train_fixed = pd.DataFrame(columns=train.columns)
    test_fixed = pd.DataFrame(columns=test.columns)
    # Check how many months each category has and get it's mode
    mode_train = int(train['Category_id'].value_counts().mode().min())
    mode_test = int(test['Category_id'].value_counts().mode().min())

    # Now we iterate train and test and make sure all categories have the same amount of samples (i.e months)
    for i, data_set in enumerate([train, test]):
        data_set_categories = data_set['Category_id'].unique().tolist()
        for category in data_set_categories:
            df = data_set[data_set['Category_id'] == category]
            df.sort_values(by=['Date'], axis=0, ascending=[True], inplace=True)
            if i == 0:
                if df.shape[0] == mode_train:  # if this category has EXACTLY "MODE" months, then concat it
                    train_fixed = pd.concat([train_fixed, df], ignore_index=True)
                elif df.shape[
                    0] > mode_train:  # if this category has MORE then "MODE" months, then take the LAST "MODE" months (train)
                    train_fixed = pd.concat([train_fixed, df.tail(mode_train)], ignore_index=True)
                else:  # If this category has less than "MODE" months then ignore it, it will not enter the adjusted train set
                    continue
            elif i == 1:
                if df.shape[0] == mode_test:  # if this category has EXACTLY "MODE" months, then concat it
                    test_fixed = pd.concat([test_fixed, df], ignore_index=True)
                elif df.shape[
                    0] > mode_test:  # if this category has MORE then "MODE" months, then take the FIRST "MODE" months (test)
                    test_fixed = pd.concat([test_fixed, df.head(mode_test)], ignore_index=True)
                else:
                    continue  # If this category has less than "MODE" months then ignore it, it will not enter the adjusted test set

    # make sure the same products are in the train and test sets
    train_items = train_fixed['Category_id'].value_counts().sort_index().index.tolist()
    test_items = test_fixed['Category_id'].value_counts().sort_index().index.tolist()
    # redundent items are items that are in train and not in test or vice versa
    redundent_items = list(set(train_items) - set(test_items)) + list(set(test_items) - set(train_items))
    # remove them
    for r in redundent_items:
        if r in train_items:
            train_items.remove(r)
        if r in test_items:
            test_items.remove(r)
    print('Items removes are {}'.format(redundent_items))

    assert train_items == test_items
    train_fixed = train_fixed[train_fixed['Category_id'].isin(train_items)]
    test_fixed = test_fixed[test_fixed['Category_id'].isin(test_items)]

    # Now each category has the same number of months. Let's rank them by ascending order
    train_fixed['rnk'] = train_fixed.groupby('Category_id')['Date'].rank()
    test_fixed['rnk'] = test_fixed.groupby('Category_id')['Date'].rank()
    # Get list of month-ranks
    train_rnks = train_fixed['rnk'].unique().tolist()
    test_rnks = test_fixed['rnk'].unique().tolist()

    train_fixed.sort_values(by=['rnk', 'Category_id'], axis=0, ascending=[True, True], inplace=True)
    test_fixed.sort_values(by=['rnk', 'Category_id'], axis=0, ascending=[True, True], inplace=True)

    # The Y is in dimension (Samples, 1,1) since we train a vec GRU for each category, not for all categories together
    y_train = train_fixed[train_fixed['Category_id'] == y_category_id]['Inflation t']
    y_test = test_fixed[test_fixed['Category_id'] == y_category_id]['Inflation t']

    x_cols = [col for col in train if col.startswith('Inflation t-')][
             ::-1]  # +[col for col in train if col.startswith('month_')]

    lst = []
    # Reshape X_train to be (Samples, Time Stamps, Categories)
    for rnk in train_rnks:
        lst.append(train_fixed[train_fixed['rnk'] == rnk][x_cols].values.T)



    X_train = np.stack(lst)

    lst = []
    # Reshape X_test to be (Samples, Time Stamps, Categories)
    for rnk in test_rnks:
        lst.append(test_fixed[test_fixed['rnk'] == rnk][x_cols].values.T)
    X_test = np.stack(lst)
    return train_fixed, test_fixed, X_train, X_test, y_train, y_test



def calc_lambda_values(df):
    """ calculates stats for each node with relation to its parent such as
        variance ratio, weight ration, depth, and correlation.
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
            child_weight = df_child.iloc[0]['Weight']
            child_samples = df_child['Inflation t'].values
            df_parent = df[(df['Indent'] == indent - 1) & (df['Category_id'] == parent_id)]
            try:
                parent_weight = df_parent.iloc[0]['Weight']
                parent_samples = df_parent['Inflation t'].values
            except IndexError:  # if for some reason the parent is not in indent-1 but higher in the tree
                print('Parent higher for {}'.format((parent_id, indent - 1)))
                df_parent = df[(df['Category_id'] == parent_id)]
                parent_weight = df_parent.iloc[0]['Weight']
                parent_samples = df_parent['Inflation t'].values

            child_samples = child_samples[~np.isnan(child_samples)]
            parent_samples = parent_samples[~np.isnan(parent_samples)]
            n = np.clip(child_weight / parent_weight, 0, 1)
            v = sigmoid(np.std(child_samples) / np.std(parent_samples))
            t = min(len(child_samples), len(parent_samples))
            s = pearsonr(child_samples[0:t], parent_samples[0:t])
            d[item] = {'weight_ratio': np.round(n, 2), 'var_ratio': np.round(v, 2), 'corr': np.round(s[0], 2),
                       'depth': item[1]}
    return d

def calc_corr(df):
    categories=df['Category_id'].value_counts().sort_index().index.tolist()
    corr=pd.DataFrame(columns=categories, index=categories)
    for e,i in enumerate(categories):
        print('This is the {} out of {}'.format (e+1, len(categories)))
        for k,j in enumerate(categories):
            if i==j:
                corr.loc[i,j]=1
            elif not pd.isnull(corr.loc[j,i]):
                corr.loc[i,j]=corr.loc[j,i]
            else:
                i_samples=df[df['Category_id']==i]['Inflation t']
                j_samples=df[df['Category_id']==j]['Inflation t']
                t=min(len(i_samples), len(j_samples))
                c=pearsonr(i_samples[0:t], j_samples[0:t])
                corr.loc[i,j]=np.round(c[0],2)
    corr.fillna(0, inplace=True)
    corr = corr.abs()
    return corr


def top_5_neighbors(row):
    return (row.sort_values(ascending=False).index.tolist()[0:5])
