import os, sys, sklearn
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.feature_selection import SelectPercentile, f_classif
import pprint as pprint
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.layers import Input, Dense, Dropout, Activation
import pickle

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

    # print(df_train.head(5))
    # print(df_train.describe())

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


def nn_model(X_train, y_train, X_test, y_test):
    config = tf.compat.v1.ConfigProto(device_count={'GPU': 1, 'CPU': 8})
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)

    # dropout to avoid overfitting
    layers = [
        Dense(122, input_shape=(122,)),
        Activation('relu'),
        Dropout(0.5),
        Dense(64),
        Activation('relu'),
        Dropout(0.5),
        Dense(64),
        Activation('relu'),
        Dropout(0.5),
        Dense(32),
        Activation('relu'),
        Dropout(0.5),
        Dense(1),
        Activation('sigmoid')
    ]

    classifier = keras.Sequential()
    for layer in layers:
        classifier.add(layer)

    # classifier = keras.Sequential([
    #     keras.layers.Dense(122, activation=tf.nn.relu, input_shape=(122,)),
    #     keras.layers.Dense(64, activation=tf.nn.relu),
    #     keras.layers.Dense(32, activation=tf.nn.relu),
    #     keras.layers.Dense(1, activation=tf.nn.sigmoid),
    # ])

    classifier.compile(optimizer='adam',
                       loss='binary_crossentropy',
                       metrics=['accuracy'])

    classifier.fit(X_train, y_train, batch_size=16, epochs=10)

    # save the model
    nn_model_pickel_file = 'nn_model.pkl'
    pickle.dump(classifier, open(nn_model_pickel_file, 'wb'))

    # load the model from disk
    loaded_model = pickle.load(open(nn_model_pickel_file, 'rb'))

    print("evaluation...")
    print("")
    print(loaded_model.evaluate(X_test, y_test, verbose=2))


if __name__ == "__main__":
    print(pd.__version__)
    print(np.__version__)
    print(sys.version)
    print(sklearn.__version__)

    train_path = os.path.join("../data/NSL-KDD", "KDDTrain+.csv")
    test_path = os.path.join("../data/NSL-KDD", "KDDTest+.csv")

    df_train, df_test = load_dataset(train_path, test_path)
    print(df_train.shape)
    print(df_test.shape)

    X_train = df_train.drop(['class'], axis=1)
    y_train = df_train['class']
    X_test = df_test.drop(['class'], axis=1)
    y_test = df_test['class']

    # pre-processing
    scaler1 = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler1.transform(X_train)

    scaler5 = preprocessing.StandardScaler().fit(X_test)
    X_test = scaler5.transform(X_test)

    nn_model(X_train, y_train, X_test, y_test)

    # clf = RandomForestClassifier()
    # clf = SVC()
    # clf.fit(X_train, y_train)
    # y_pred = clf.predict(X_test)
    #
    # print("Accuracy is %f." % accuracy_score(y_test, y_pred))
    # print(confusion_matrix(y_test, y_pred))
    # print("Precision score is %f." % precision_score(y_test, y_pred))
    # print("Recall score is %f." % recall_score(y_test, y_pred))
    # print("F1 score is %f." % f1_score(y_test, y_pred))
    #
    # # probability=True should be set, default is false
    # probs = clf.predict_proba(X_test)
    # preds = probs[:, 1]
    # fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
    # roc_auc = metrics.auc(fpr, tpr)
    #
    # plt.title('Receiver Operating Characteristic')
    # plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    # plt.legend(loc='lower right')
    # plt.plot([0, 1], [0, 1], 'r--')
    # plt.xlim([0, 1])
    # plt.ylim([0, 1])
    # plt.ylabel('True Positive Rate')
    # plt.xlabel('False Positive Rate')
    # plt.show()
