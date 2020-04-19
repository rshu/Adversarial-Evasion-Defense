import os, sys, sklearn
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pprint as pprint

col_names = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment",
             "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted",
             "num_root", "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
             "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
             "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
             "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
             "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
             "dst_host_srv_rerror_rate", "class"]


def load_dataset(train_dataset_path, test_dataset_path):
    df_train = pd.read_csv(train_dataset_path, header=0, names=col_names)
    df_test = pd.read_csv(test_dataset_path, header=0, names=col_names)

    # print(df_test)
    print(df_train.dtypes)
    print(df_test.dtypes)

    # shape, this gives the dimensions of the dataset
    print('Dimensions of the Training set:', df_train.shape)
    print('Dimensions of the Test set:', df_test.shape)

    print(df_train.head(5))
    print(df_train.describe())

    print('Label distribution Training set:')
    print(df_train['class'].value_counts())
    print()
    print('Label distribution Test set:')
    print(df_test['class'].value_counts())

    # Identify caategoricaal features
    for col_name in df_train.columns:
        if df_train[col_name].dtype == 'object':
            unique_category = len(df_train[col_name].unique())
            print("Training Dataset: Feature '{col_name}' has {unique_cat} categories".format(col_name=col_name,
                                                                                              unique_cat=unique_category))
    # print()
    # print('Distribution of categories in service:')
    # print(df_train['service'].value_counts().sort_values(ascending=False).head())

    # feature 'service' has 64 categories in test, 70 categories in train
    print()
    for col_name in df_test.columns:
        if df_test[col_name].dtype == 'object':
            unique_category = len(df_test[col_name].unique())
            print("Testing Dataset: Feature '{col_name}' has {unique_cat} categories".format(col_name=col_name,
                                                                                             unique_cat=unique_category))

    categorical_columns = ['protocol_type', 'service', 'flag']
    df_train_categorical_values = df_train[categorical_columns]
    df_test_categorical_values = df_test[categorical_columns]
    print(df_train_categorical_values)
    print(df_test_categorical_values)

    # one-hot-encoding, create dummy column names
    unique_protocal = sorted(df_train['protocol_type'].unique())
    crafted_protocal = ['Protocol_type_' + x for x in unique_protocal]
    print(unique_protocal)
    print(crafted_protocal)
    assert len(crafted_protocal) == 3

    unique_service = sorted(df_train['service'].unique())
    crafted_service = ['service_' + x for x in unique_service]
    print(unique_service)
    print(crafted_service)
    assert len(crafted_service) == 70

    unique_flag = sorted(df_train['flag'].unique())
    crafted_flag = ['flag_' + x for x in unique_flag]
    print(unique_flag)
    print(crafted_flag)
    assert len(crafted_flag) == 11

    train_dummy_cols = unique_protocal + unique_service + unique_flag
    print(train_dummy_cols)
    assert len(train_dummy_cols) == 84

    unique_service_test = sorted(df_test['service'].unique())
    crafted_service_test = ['service_' + x for x in unique_service_test]
    test_dummy_cols = unique_protocal + unique_service_test + unique_flag
    assert len(test_dummy_cols) == 78

    df_train_categorical_value_encode = df_train_categorical_values.apply(LabelEncoder().fit_transform)
    df_test_categorical_value_encode = df_test_categorical_values.apply(LabelEncoder().fit_transform)
    print(df_train_categorical_value_encode)
    print(df_test_categorical_value_encode)

    oneHotEncoder = OneHotEncoder()
    df_train_categorical_values_onehot = oneHotEncoder.fit_transform(df_train_categorical_value_encode)
    df_train_cat_data = pd.DataFrame(df_train_categorical_values_onehot.toarray(), columns=train_dummy_cols)
    print(df_train_cat_data)

    # feature test in service miss 6 categories
    train_service = df_train['service'].tolist()
    test_service = df_test['service'].tolist()
    service_difference = list(set(train_service) - set(test_service))
    service_difference = ['service_' + x for x in service_difference]
    print(service_difference)

    df_test_categorical_values_onehot = oneHotEncoder.fit_transform(df_test_categorical_value_encode)
    df_test_cat_data = pd.DataFrame(df_test_categorical_values_onehot.toarray(), columns=test_dummy_cols)

    for col in service_difference:
        df_test_cat_data[col] = 0

    print(df_test_cat_data)

    # join and replace original dataset
    new_df_train = df_train.join(df_train_cat_data)
    new_df_train.drop('flag', axis=1, inplace=True)
    new_df_train.drop('protocol_type', axis=1, inplace=True)
    new_df_train.drop('service', axis=1, inplace=True)

    new_df_test = df_test.join(df_test_cat_data)
    new_df_test.drop('flag', axis=1, inplace=True)
    new_df_test.drop('protocol_type', axis=1, inplace=True)
    new_df_test.drop('service', axis=1, inplace=True)

    new_df_train['class'] = new_df_train['class'].map({'anomaly': 1, 'normal': 0})
    new_df_test['class'] = new_df_test['class'].map({'anomaly': 1, 'normal': 0})

    print(new_df_train['class'])
    print(new_df_test['class'])

    return new_df_train, new_df_test

if __name__ == "__main__":
    print(pd.__version__)
    print(np.__version__)
    print(sys.version)
    print(sklearn.__version__)

    train_dataset_path = os.path.join("../data/NSL-KDD", "KDDTrain+.csv")
    test_dataset_path = os.path.join("../data/NSL-KDD", "KDDTest+.csv")

    load_dataset(train_dataset_path, test_dataset_path)
