import mordred
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectFromModel


def wipe_off_data(path):
    """
    Clean data
    :param path: Path of the data to be cleaned
    :return: The cleaned csv file
    """
    data = pd.read_csv(path)
    smiles_pd = data.iloc[:, :4]
    column_list = [column for column in data][4:]
    data_np = data.values[:, 4:]
    m, n = data_np.shape
    print(m, n)

    # Iterate through each column. If there is a value in the column that is not 0, it is not deleted; If the value is 0, delete it
    filter_descriptors = {}
    for i in range(n):
        tag = 0
        sum = 0
        des_Name = column_list[i]
        des_value_column = data_np[:, i].tolist()
        # The two columns are True or False, converted to 0 or 1
        if des_Name == 'GhoseFilter' or des_Name == 'Lipinski':
            c = []
            for j in des_value_column:
                if j == False:
                    c.append(0)
                else:
                    c.append(1)
            des_value_column = c

        for value in des_value_column:
            if isinstance(value, str):
                tag = 1
                break
            if np.isnan(value):
                tag = 1
                break
            sum += abs(value)
        if tag == 1:
            continue
        if sum == 0:
            continue

        filter_descriptors[des_Name] = des_value_column

    filter_descriptors_pd = pd.DataFrame(filter_descriptors)
    data_pd_out = pd.concat([smiles_pd, filter_descriptors_pd], axis=1)
    data_pd_out.to_csv('xxx/xxx/xxx.csv', index=False)


def normalization_descriptors(df_descriptors_filter):
    """
    Descriptor normalization
    :param df_descriptors_filter: Descriptor to be normalized
    :return: Normalized descriptor DataFrame
    """
    column_list = [column for column in df_descriptors_filter][4:]
    smiles_pd = df_descriptors_filter.iloc[:, :4]
    data_des = df_descriptors_filter.values[:, 4:]
    n_des = preprocessing.MinMaxScaler(feature_range=(0, 1))
    data_des = n_des.fit_transform(data_des)
    filter_descriptors = {}
    for index, column in enumerate(column_list):
        filter_descriptors[column] = data_des[:, index].tolist()
    df_descriptors_filter_normalized = pd.concat([smiles_pd, pd.DataFrame(filter_descriptors)], axis=1)
    return df_descriptors_filter_normalized


def descriptors_processing(var, path):
    """
    Remove columns with small variance and columns with large correlation coefficients
    :param var: Variance threshold
    :param path: Path of the csv file to be processed
    :return: The processed csv file
    """
    df_descriptors_filter = pd.read_csv(path)
    df_descriptors_filter_normalized = normalization_descriptors(df_descriptors_filter)
    smile_pd = df_descriptors_filter_normalized.iloc[:, :4]
    target = smile_pd["value_mean"]
    df_descriptors_filter_normalized = df_descriptors_filter_normalized.iloc[:, 4:]
    for column_name, rows in df_descriptors_filter_normalized.iteritems():
        for row in rows:
            if type(row) == mordred.error.Missing:
                df_descriptors_filter_normalized.drop([column_name], axis=1, inplace=True)
                break

    df_descriptors_filter_normalized.dropna(axis=1)
    df_descriptors_filter_normalized.describe()
    var_list_columns = df_descriptors_filter_normalized.var().index.tolist()
    for column in df_descriptors_filter_normalized.columns:
        if column not in var_list_columns:
            print(column)
            df_descriptors_filter_normalized.drop([column], axis=1, inplace=True)

    # Delete columns with variance var<0.01
    for column in df_descriptors_filter_normalized.columns:
        if df_descriptors_filter_normalized[column].var() < var:
            df_descriptors_filter_normalized.drop([column], axis=1, inplace=True)
    print(df_descriptors_filter_normalized.values.shape)

    # Removes descriptors that are strongly correlated with other descriptors (|r| > 0.8) but weakly correlated with the target value
    columns_to_drop = []
    for column in tqdm(df_descriptors_filter_normalized.columns):
        correlation_with_target = df_descriptors_filter_normalized[column].corr(target)
        for other_column in df_descriptors_filter_normalized.columns:
            if column == other_column:
                continue
            correlation_with_other = df_descriptors_filter_normalized[column].corr(
                df_descriptors_filter_normalized[other_column])

            if np.abs(correlation_with_other) <= 0.8:
                continue
            other_correlation_with_target = df_descriptors_filter_normalized[other_column].corr(target)
            if np.abs(correlation_with_target) >= np.abs(other_correlation_with_target):
                columns_to_drop.append(other_column)
            else:
                columns_to_drop.append(column)
    columns_to_drop = list(set(columns_to_drop))
    df_descriptors_filter_normalized.drop(columns_to_drop, axis=1, inplace=True)
    print(df_descriptors_filter_normalized.values.shape)

    pd.concat([smile_pd, df_descriptors_filter_normalized], axis=1).to_csv(
        'xxx/xxx/xxx.csv' % var, index=False)


def ridgeCV(data):
    """
    Ridge cross validation dimension reduction
    Parameters
    ----------
    data
    Dataframe whose dimension is to be reduced
    Returns
    -------
    csv file after dimensionality reduction
    """
    smile_pd = data.iloc[:, :4]
    x, y = data.values[:, 4:], data.values[:, 3].reshape(-1, 1)
    a = np.logspace(-3, 2, num=100, endpoint=True)
    kf = KFold(n_splits=5, shuffle=True, random_state=2023)
    reg = SelectFromModel(RidgeCV(alphas=a, cv=kf, fit_intercept=True, normalize=False)).fit(x, y)
    coefs = pd.DataFrame({'coefficient': np.abs(reg.estimator_.coef_.reshape(-1, ))}, index=data.columns[4:])
    coef = coefs[coefs['coefficient'] > reg.threshold_]
    print(
        f"best alpha is {reg.estimator_.alpha_}; score is {reg.estimator_.score(x, y)}; {coef.count(axis=0).values[0]} descriptors were picked.")
    pd.concat([smile_pd, data[coef.index]], axis=1).to_csv('xxx/xxx/xxx.csv', index=False)
