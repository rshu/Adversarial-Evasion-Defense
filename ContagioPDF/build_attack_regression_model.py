import pandas as pd
import ast, sys
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklearn.svm import SVR

desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 50)


def processing(df):
    df["params"] = df["params"].map(lambda d: ast.literal_eval(d))
    dfNew = df.join(pd.DataFrame(df["params"].to_dict()).T)
    dfNew = dfNew.drop(['params', 'loss'], axis=1)

    # one-hot encoding
    batch_size_columns = ['batch_size']
    df_batch_size_values = dfNew[batch_size_columns]

    unique_batch_size = sorted(dfNew['batch_size'].astype(str).unique())
    crafted_batch_size_cols = ['batch_size_' + x for x in unique_batch_size]
    df_batch_size_values_encode = df_batch_size_values.apply(LabelEncoder().fit_transform)

    oneHotEncoder = OneHotEncoder()
    df_batch_size_values_onehot = oneHotEncoder.fit_transform(df_batch_size_values_encode)
    df_batch_size = pd.DataFrame(df_batch_size_values_onehot.toarray(), columns=crafted_batch_size_cols)

    dfNew.drop('batch_size', axis=1, inplace=True)
    dfNew = dfNew.join(df_batch_size)

    full_batch_size = ['16', '32', '64', '128']

    if len(full_batch_size) == len(unique_batch_size):
        pass
    else:
        diff_batch_size = list(set(full_batch_size) - set(unique_batch_size))
        for x in diff_batch_size:
            dfNew['batch_size_' + x] = 0.0

    hidden_layer_activation_columns = ['hidden_layer_activation']
    df_hidden_layer_activation_values = dfNew[hidden_layer_activation_columns]

    unique_hidden_layer_activation = sorted(dfNew['hidden_layer_activation'].unique())
    crafted_hidden_layer_activation_cols = ['hidden_layer_activation_' + x for x in unique_hidden_layer_activation]
    df_hidden_layer_activation_values_encode = df_hidden_layer_activation_values.apply(LabelEncoder().fit_transform)

    oneHotEncoder = OneHotEncoder()
    df_hidden_layer_activation_values_onehot = oneHotEncoder.fit_transform(df_hidden_layer_activation_values_encode)
    df_hidden_layer_activation = pd.DataFrame(df_hidden_layer_activation_values_onehot.toarray(),
                                              columns=crafted_hidden_layer_activation_cols)

    dfNew.drop('hidden_layer_activation', axis=1, inplace=True)
    dfNew = dfNew.join(df_hidden_layer_activation)

    full_hidden_layer_activation = ['elu', 'relu', 'selu', 'sigmoid', 'softmax', 'tanh', 'hard_sigmoid',
                                    'softplus', 'softsign', 'linear', 'exponential']

    if len(full_hidden_layer_activation) == len(unique_hidden_layer_activation):
        pass
    else:
        diff_hidden_layer_activation = list(set(full_hidden_layer_activation) - set(unique_hidden_layer_activation))
        for x in diff_hidden_layer_activation:
            dfNew['hidden_layer_activation_' + x] = 0.0

    output_layer_activation_columns = ['output_layer_activation']
    df_output_layer_activation_values = dfNew[output_layer_activation_columns]

    unique_output_layer_activation = sorted(dfNew['output_layer_activation'].unique())
    crafted_output_layer_activation_cols = ['output_layer_activation_' + x for x in unique_output_layer_activation]
    df_output_layer_activation_values_encode = df_output_layer_activation_values.apply(LabelEncoder().fit_transform)

    oneHotEncoder = OneHotEncoder()
    df_output_layer_activation_values_onehot = oneHotEncoder.fit_transform(df_output_layer_activation_values_encode)
    df_output_layer_activation = pd.DataFrame(df_output_layer_activation_values_onehot.toarray(),
                                              columns=crafted_output_layer_activation_cols)

    dfNew.drop('output_layer_activation', axis=1, inplace=True)
    dfNew = dfNew.join(df_output_layer_activation)

    full_output_layer_activation = ['elu', 'relu', 'selu', 'sigmoid', 'softmax', 'tanh', 'hard_sigmoid',
                                    'softplus', 'softsign', 'linear', 'exponential']

    if len(full_output_layer_activation) == len(unique_output_layer_activation):
        pass
    else:
        diff_output_layer_activation = list(set(full_output_layer_activation) - set(unique_output_layer_activation))
        for x in diff_output_layer_activation:
            dfNew['output_layer_activation_' + x] = 0.0

    optimizer_columns = ['optimizer']
    df_optimizer_values = dfNew[optimizer_columns]

    unique_optimizer = sorted(dfNew['optimizer'].unique())
    crafted_optimizer_cols = ['optimizer_' + x for x in unique_optimizer]
    df_optimizer_values_encode = df_optimizer_values.apply(LabelEncoder().fit_transform)

    oneHotEncoder = OneHotEncoder()
    df_optimizer_values_onehot = oneHotEncoder.fit_transform(df_optimizer_values_encode)
    df_optimizer = pd.DataFrame(df_optimizer_values_onehot.toarray(),
                                columns=crafted_optimizer_cols)

    dfNew.drop('optimizer', axis=1, inplace=True)
    dfNew = dfNew.join(df_optimizer)

    full_optimizer = ['Adadelta', 'Adagrad', 'Adam', 'Adamax', 'NAdam', 'RMSprop', 'SGD']

    if len(full_optimizer) == len(unique_optimizer):
        pass
    else:
        diff_optimizer = list(set(full_optimizer) - set(unique_optimizer))
        for x in diff_optimizer:
            dfNew['optimizer_' + x] = 0.0

    learning_rate_columns = ['learning_rate']
    df_learning_rate_values = dfNew[learning_rate_columns]

    unique_learning_rate = sorted(dfNew['learning_rate'].astype(str).unique())
    crafted_learning_rate_cols = ['learning_rate_' + x for x in unique_learning_rate]
    df_learning_rate_values_encode = df_learning_rate_values.apply(LabelEncoder().fit_transform)

    oneHotEncoder = OneHotEncoder()
    df_learning_rate_values_onehot = oneHotEncoder.fit_transform(df_learning_rate_values_encode)
    df_learning_rate = pd.DataFrame(df_learning_rate_values_onehot.toarray(),
                                    columns=crafted_learning_rate_cols)

    dfNew.drop('learning_rate', axis=1, inplace=True)
    dfNew = dfNew.join(df_learning_rate)

    full_learning_rate = ['0.001', '0.01', '0.1']

    if len(full_learning_rate) == len(unique_learning_rate):
        pass
    else:
        diff_learning_rate = list(set(full_learning_rate) - set(unique_learning_rate))
        for x in diff_learning_rate:
            dfNew['learning_rate_' + x] = 0.0

    return dfNew


if __name__ == "__main__":
    df = pd.read_csv('misc/distance_result_sample.csv')
    df_train = processing(df)

    df2 = pd.read_csv('distance_result.csv')
    df_test = processing(df2)

    # df_test.drop(df_test[df_test['distance'] < 0.4].index, inplace=True)
    # df_train.drop(df_train[df_train['distance'] < 0.4].index, inplace=True)

    print(df_train)
    print(df_test)

    # build regression models

    X_train = df_train.drop(['FGSM', 'BIM-A', 'BIM-B', 'deepfool', 'distance'], axis=1)
    y_fgsm_train = df_train['FGSM']
    y_bim_a_train = df_train['BIM-A']
    y_bim_b_train = df_train['BIM-B']
    y_deepfool_train = df_train['deepfool']

    X_test = df_test.drop(['FGSM', 'BIM-A', 'BIM-B', 'deepfool', 'distance'], axis=1)
    y_fgsm_test = df_test['FGSM']
    y_bim_a_test = df_test['BIM-A']
    y_bim_b_test = df_test['BIM-B']
    y_deepfool_test = df_test['deepfool']

    scaler1 = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler1.transform(X_train)
    scaler2 = preprocessing.StandardScaler().fit(X_test)
    X_test = scaler2.transform(X_test)

    model_fgsm = DecisionTreeRegressor()
    model_fgsm.fit(X_train, y_fgsm_train)
    y_fgsm_predict = model_fgsm.predict(X_test)
    print(mean_absolute_error(y_fgsm_test, y_fgsm_predict))
    # print(mean_squared_error(y_fgsm_test, y_fgsm_predict))
    # print(list(y_fgsm_predict))
    # print(list(y_fgsm_test))
    print("Prediction ", "Actual")
    for i, j in zip(list(y_fgsm_predict), list(y_fgsm_test)):
        # if j >= 0.8:
        print(i, j)

    model_bim_a = DecisionTreeRegressor()
    model_bim_a.fit(X_train, y_bim_a_train)
    y_bim_a_predict = model_bim_a.predict(X_test)
    print(mean_absolute_error(y_bim_a_test, y_bim_a_predict))
    # print(mean_squared_error(y_bim_a_test, y_bim_a_predict))
    print("")
    print("Prediction ", "Actual")
    for i, j in zip(list(y_bim_a_predict), list(y_bim_a_test)):
        # if j >= 0.8:
        print(i, j)

    model_bim_b = DecisionTreeRegressor()
    model_bim_b.fit(X_train, y_bim_b_train)
    y_bim_b_predict = model_bim_b.predict(X_test)
    print(mean_absolute_error(y_bim_b_test, y_bim_b_predict))
    # print(mean_squared_error(y_bim_a_test, y_bim_a_predict))
    print("")
    print("Prediction ", "Actual")
    for i, j in zip(list(y_bim_b_predict), list(y_bim_b_test)):
        # if j >= 0.8:
        print(i, j)

    model_deepfool = DecisionTreeRegressor()
    model_deepfool.fit(X_train, y_deepfool_train)
    y_deepfool_predict = model_deepfool.predict(X_test)
    print(mean_absolute_error(y_deepfool_test, y_deepfool_predict))
    # print(mean_squared_error(y_bim_a_test, y_bim_a_predict))
    print("")
    print("Prediction ", "Actual")
    for i, j in zip(list(y_deepfool_predict), list(y_deepfool_test)):
        # if j >= 0.8:
        print(i, j)

    # print(df_test['distance'])
    # print(y_fgsm_test - y_predict)
    # print(y_predict)
