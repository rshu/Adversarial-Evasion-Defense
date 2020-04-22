import pandas as pd
from os import path
import numpy as np
import sys, collections
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import pickle
import scipy.stats
import seaborn as sns
from correlation import plot_correlation, drop_lin_correlated
import tensorflow as tf
import keras
from keras.layers import Input, Dense, Dropout, Activation
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 50)

label_map = {'BENIGN': 0, 'SCAREWARE_FAKEAV': 1, 'SCAREWARE_FAKEAPP': 1, 'SCAREWARE_FAKEAPPAL': 1,
             'SCAREWARE_ANDROIDDEFENDER': 1, 'SCAREWARE_VIRUSSHIELD': 1, 'SCAREWARE_FAKEJOBOFFER': 1, 'MALWARE': 1,
             'SCAREWARE_PENETHO': 1, 'SCAREWARE_FAKETAOBAO': 1, 'SCAREWARE_AVPASS': 1, 'SCAREWARE_ANDROIDSPY': 1,
             'SCAREWARE_AVFORANDROID': 1, 'ADWARE_FEIWO': 1, 'ADWARE_GOOLIGAN': 1, 'ADWARE_KEMOGE': 1,
             'ADWARE_EWIND': 1, 'ADWARE_YOUMI': 1, 'ADWARE_DOWGIN': 1, 'ADWARE_SELFMITE': 1, 'ADWARE_KOODOUS': 1,
             'ADWARE_MOBIDASH': 1, 'ADWARE_SHUANET': 1, 'SMSMALWARE_FAKEMART': 1, 'SMSMALWARE_ZSONE': 1,
             'SMSMALWARE_FAKEINST': 1, 'SMSMALWARE_MAZARBOT': 1, 'SMSMALWARE_NANDROBOX': 1, 'SMSMALWARE_JIFAKE': 1,
             'SMSMALWARE_SMSSNIFFER': 1, 'SMSMALWARE_BEANBOT': 1, 'SCAREWARE': 1, 'SMSMALWARE_FAKENOTIFY': 1,
             'SMSMALWARE_PLANKTON': 1, 'SMSMALWARE_BIIGE': 1, 'RANSOMWARE_LOCKERPIN': 1, 'RANSOMWARE_CHARGER': 1,
             'RANSOMWARE_PORNDROID': 1, 'RANSOMWARE_PLETOR': 1, 'RANSOMWARE_JISUT': 1, 'RANSOMWARE_WANNALOCKER': 1,
             'RANSOMWARE_KOLER': 1, 'RANSOMWARE_RANSOMBO': 1, 'RANSOMWARE_SIMPLOCKER': 1, 'RANSOMWARE_SVPENG': 1}


def load_dataset(file_path):
    df = pd.read_csv(file_path, encoding='utf8', low_memory=False)
    # the FLOW ID column seems a combination of source and destination IP, port and protocal
    # hence useless to me
    df = df.drop([df.columns[0]], axis=1)
    # remove white space at the beginning of string in dataframe header
    df.columns = df.columns.str.lstrip()
    # print(list(df.columns))

    # # print(df.dtypes())

    print(df.shape)
    # print(df.head())
    # print(df.describe())
    # # print(list(df.Label.unique()))
    # print(df['Label'].value_counts())
    # print(df['Label'].isnull().sum())

    # Drop rows with values missing for label column (target variable)
    # df = df.dropna(subset=['Label'], inplace=True)
    df = df.dropna()
    df = df.reset_index(drop=True)
    # print(df['Label'].isnull().sum())
    print(df.shape)  # (2616566, 84)

    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    # print(df.select_dtypes(exclude=['int', 'float', 'datetime']))

    df['Packet Length Std'] = pd.to_numeric(df['Packet Length Std'], errors='coerce')
    df['CWE Flag Count'] = pd.to_numeric(df['CWE Flag Count'], errors='coerce')
    df['Active Mean'] = pd.to_numeric(df['Active Mean'], errors='coerce')
    df['Flow IAT Mean'] = pd.to_numeric(df['Flow IAT Mean'], errors='coerce')
    df['URG Flag Count'] = pd.to_numeric(df['URG Flag Count'], errors='coerce')
    df['Down/Up Ratio'] = pd.to_numeric(df['Down/Up Ratio'], errors='coerce')
    df['Fwd Avg Bytes/Bulk'] = pd.to_numeric(df['Fwd Avg Bytes/Bulk'], errors='coerce')
    df['Flow IAT Min'] = pd.to_numeric(df['Flow IAT Min'], errors='coerce')
    # print(df.select_dtypes(exclude=['int', 'float', 'datetime']))

    df['Source IP'] = df['Source IP'].astype('str')
    df['Destination IP'] = df['Destination IP'].astype('str')
    df['Label'] = df['Label'].map(label_map)

    # Drop the features which have only 1 unique value:
    nunique = df.apply(pd.Series.nunique)
    cols_to_drop = nunique[nunique == 1].index
    df.drop(cols_to_drop, axis=1, inplace=True)
    print(df.shape)

    # sample 20%
    df = df.sample(frac=.20, random_state=20)
    print(df['Label'].value_counts())
    # print(df['Source Port'].value_counts())
    # print(df['Destination Port'].value_counts())
    print(df['Timestamp'].value_counts())  # 2017-01-01 - 2017-12-31
    # time_df = df['Timestamp'].value_counts()
    # time_df.to_csv("./time_df.csv")
    x = pd.date_range('2017-01-01', '2017-12-31', freq='D').astype(str)
    time_map = dict(zip(x, np.arange(1, 366)))

    df['Timestamp'] = df['Timestamp'].astype(str).str[:10].map(time_map)
    print(df.head())
    print(df['Timestamp'].value_counts())

    # Label Encoding acc: ~53.7%
    df["Source IP"] = LabelEncoder().fit_transform(df["Source IP"])
    df["Destination IP"] = LabelEncoder().fit_transform(df["Destination IP"])

    df.to_pickle("./saved_dataframe.pkl")
    return df

    # One hot Encoding
    categorical_columns = ['Source IP', 'Destination IP']
    df_categorical_values = df[categorical_columns]
    print(df_categorical_values)

    unique_source_ip = sorted(df['Source IP'].unique())
    crafted_source_ip = ['Source_IP_' + x for x in unique_source_ip]
    print(unique_source_ip)
    print(crafted_source_ip)

    unique_destination_ip = sorted(df['Destination IP'].unique())
    crafted_destination_ip = ['Destination_IP_' + x for x in unique_destination_ip]
    print(unique_destination_ip)
    print(crafted_destination_ip)

    dummy_cols = crafted_source_ip + crafted_destination_ip
    print(dummy_cols)

    df_categorical_values_encode = df_categorical_values.apply(LabelEncoder().fit_transform)
    # print(df_categorical_values_encode)

    oneHotEncoder = OneHotEncoder()
    df_categorical_values_onehot = oneHotEncoder.fit_transform(df_categorical_values_encode)
    df_cat_data = pd.DataFrame(df_categorical_values_onehot.toarray(), columns=dummy_cols)

    # print(df_cat_data)
    df.drop('Source IP', axis=1, inplace=True)
    df.drop('Destination IP', axis=1, inplace=True)

    # memory error if join directly
    # The problem is that when you merge two dataframes,
    # you need enough memory for both of them, plus the merged one.
    print("joining dataframes...")
    # new_df = df.join(df_cat_data) # do not do this
    df.to_csv("./original_df.csv")
    df_cat_data.to_csv("./tojoin_df.csv")
    print("saving done..")

    exit(-1)
    # print("save dataframe to file...")
    # new_df.to_pickle("./saved_dataframe.pkl")
    # return new_df


def build_models(X_train, y_train, X_test, y_test):
    config = tf.compat.v1.ConfigProto(device_count={'GPU': 1, 'CPU': 8})
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)

    # dropout to avoid overfitting
    layers = [
        Dense(X_train.shape[1], input_shape=(X_train.shape[1],)),
        Activation('relu'),
        # Dropout(0.5),
        Dense(64),
        Activation('relu'),
        # Dropout(0.5),
        Dense(32),
        Activation('relu'),
        # Dropout(0.5),
        Dense(16),
        Activation('relu'),
        # Dropout(0.5),
        Dense(1),
        Activation('sigmoid')
    ]

    classifier = keras.Sequential()
    for layer in layers:
        classifier.add(layer)

    classifier.compile(optimizer='adam',
                       loss='binary_crossentropy',
                       metrics=['accuracy'])
    classifier.fit(X_train, y_train, batch_size=32, epochs=10)

    # save the model
    nn_model_pickel_file = 'saved_nn_model.pkl'
    pickle.dump(classifier, open(nn_model_pickel_file, 'wb'))

    # load the model from disk
    # loaded_model = pickle.load(open(nn_model_pickel_file, 'rb'))

    print("evaluation...")
    print("")
    print(classifier.evaluate(X_test, y_test, verbose=2))


if __name__ == "__main__":
    # print(len(col_names))
    # print(len(set(col_names)))
    # print([item for item, count in collections.Counter(col_names).items() if count > 1])

    file_path = "../data/CICAndMal2017/ConsolidateData.csv"

    if path.exists("saved_dataframe.pkl"):
        df = pd.read_pickle("./saved_dataframe.pkl")
    else:
        df = load_dataset(file_path)

    print(df.shape)
    print(df['Label'].value_counts())
    # check Anderson-Darling test results
    # print(scipy.stats.anderson(df['Label'], dist='norm'))

    # plot correlation heatmap
    # plot_correlation(df[df.columns.difference(['Label', 'Timestamp'])], 'kendall')

    # # with feature selection
    # scaler = MinMaxScaler(feature_range=[0, 1])
    # data_rescaled = scaler.fit_transform(df[df.columns.difference(['Label', 'Timestamp'])])
    # pca = PCA().fit(data_rescaled)  # Plotting the Cumulative Summation of the Explained Variance
    # plt.figure()
    # plt.plot(np.cumsum(pca.explained_variance_ratio_))
    # plt.xlabel('Number of Components', size=10)
    # plt.ylabel('Variance (%)', size=10)
    # plt.title('Dataset Explained Variance', size=10)
    # plt.rc('xtick', labelsize=10)
    # plt.rc('ytick', labelsize=10)
    # plt.show()
    #
    # train_features = df[df.columns.difference(['Label', 'Timestamp'])]
    # model = PCA(n_components=20).fit(train_features)
    # # number of components
    # n_pcs = model.components_.shape[0]
    #
    # # get the index of the most important feature on each component i.e. largest absolute value
    # most_important = [np.abs(model.components_[i]).argmax() for i in range(n_pcs)]
    # initial_feature_names = df[df.columns.difference(['Label', 'Timestamp'])].columns.values
    #
    # most_important_names = [initial_feature_names[most_important[i]] for i in range(n_pcs)]
    # # list comprehension
    # dic = {'PC{}'.format(i + 1): most_important_names[i] for i in range(n_pcs)}
    #
    # pca_df = pd.DataFrame(dic.items())
    # print(pca_df.shape)
    #

    X = df.drop(['Label'], axis=1)
    print(X)
    y = df['Label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    print(y_train.value_counts())
    print(y_test.value_counts())

    # pca = PCA(0.9899)
    # pca.fit(X_train)
    # X_train = pca.transform(X_train)
    # X_test = pca.transform(X_test)
    # print(X_train.shape)
    # print(X_test.shape)

    # sys.exit(-1)
    scaler1 = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler1.transform(X_train)

    scaler5 = preprocessing.StandardScaler().fit(X_test)
    X_test = scaler5.transform(X_test)

    build_models(X_train, y_train, X_test, y_test)

    # clf = RandomForestClassifier()
    # clf.fit(X_train, y_train)
    #
    # model_pickel_file = 'my_model.pkl'
    # pickle.dump(clf, open(model_pickel_file, 'wb'))
    #
    # # load the model from disk
    # loaded_model = pickle.load(open(model_pickel_file, 'rb'))
    #
    # y_pred = loaded_model.predict(X_test)

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

    # Accuracy is 0.734958.
    # [[212998  89554]
    #  [83821 267769]]
    # Precision score is 0.749375.
    # Recall score is 0.761594.
    # F1 score is 0.755435.
