from os import path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import random
import os

CIC_IDS_2017_label_map = {'BENIGN': 0, 'PortScan': 1, 'FTP-Patator': 1, 'SSH-Patator': 1, 'Bot': 1, 'Infiltration': 1,
                          'Web Attack � Brute Force': 1, 'Web Attack � XSS': 1, 'Web Attack � Sql Injection': 1,
                          'DDoS': 1,
                          'DoS slowloris': 1, 'DoS Slowhttptest': 1, 'DoS Hulk': 1, 'DoS GoldenEye': 1, 'Heartbleed': 1}

CICAndMal2017_label_map = {'BENIGN': 0, 'SCAREWARE_FAKEAV': 1, 'SCAREWARE_FAKEAPP': 1, 'SCAREWARE_FAKEAPPAL': 1,
                           'SCAREWARE_ANDROIDDEFENDER': 1, 'SCAREWARE_VIRUSSHIELD': 1, 'SCAREWARE_FAKEJOBOFFER': 1,
                           'MALWARE': 1,
                           'SCAREWARE_PENETHO': 1, 'SCAREWARE_FAKETAOBAO': 1, 'SCAREWARE_AVPASS': 1,
                           'SCAREWARE_ANDROIDSPY': 1,
                           'SCAREWARE_AVFORANDROID': 1, 'ADWARE_FEIWO': 1, 'ADWARE_GOOLIGAN': 1, 'ADWARE_KEMOGE': 1,
                           'ADWARE_EWIND': 1, 'ADWARE_YOUMI': 1, 'ADWARE_DOWGIN': 1, 'ADWARE_SELFMITE': 1,
                           'ADWARE_KOODOUS': 1,
                           'ADWARE_MOBIDASH': 1, 'ADWARE_SHUANET': 1, 'SMSMALWARE_FAKEMART': 1, 'SMSMALWARE_ZSONE': 1,
                           'SMSMALWARE_FAKEINST': 1, 'SMSMALWARE_MAZARBOT': 1, 'SMSMALWARE_NANDROBOX': 1,
                           'SMSMALWARE_JIFAKE': 1,
                           'SMSMALWARE_SMSSNIFFER': 1, 'SMSMALWARE_BEANBOT': 1, 'SCAREWARE': 1,
                           'SMSMALWARE_FAKENOTIFY': 1,
                           'SMSMALWARE_PLANKTON': 1, 'SMSMALWARE_BIIGE': 1, 'RANSOMWARE_LOCKERPIN': 1,
                           'RANSOMWARE_CHARGER': 1,
                           'RANSOMWARE_PORNDROID': 1, 'RANSOMWARE_PLETOR': 1, 'RANSOMWARE_JISUT': 1,
                           'RANSOMWARE_WANNALOCKER': 1,
                           'RANSOMWARE_KOLER': 1, 'RANSOMWARE_RANSOMBO': 1, 'RANSOMWARE_SIMPLOCKER': 1,
                           'RANSOMWARE_SVPENG': 1}

NSL_KDD_col_names = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment",
                     "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted",
                     "num_root", "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
                     "is_host_login",
                     "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
                     "srv_rerror_rate",
                     "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
                     "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
                     "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
                     "dst_host_rerror_rate",
                     "dst_host_srv_rerror_rate", "class"]


def load_contagio_dataset(file_path):
    df = pd.read_csv(file_path, encoding='utf8', low_memory=False)
    df = df.drop(['filename'], axis=1)
    df.columns = df.columns.str.lstrip()

    print(df.shape)  # (27205, 136)
    print(df.dtypes)
    print(df.head())
    print(df.describe())
    # print(list(df.Label.unique()))
    print(df['class'].value_counts())
    print(df['class'].isnull().sum())

    df = df.dropna()
    df = df.reset_index(drop=True)
    print(df.shape)  # (27205, 136)

    print(df.select_dtypes(exclude=['int', 'float']))

    df['class'] = df['class'].astype(str).map({'False': 0, 'True': 1})
    df['box_other_only'] = df['box_other_only'].astype(str).map({'False': 0, 'True': 1})
    df['pdfid_mismatch'] = df['pdfid_mismatch'].astype(str).map({'False': 0, 'True': 1})

    # print(df.select_dtypes(exclude=['int', 'float']))
    print(df['class'].value_counts())
    print(df['class'].dtypes)
    print(df['box_other_only'].dtypes)
    print(df['pdfid_mismatch'].dtypes)

    df = df.drop_duplicates()
    print(df.shape)  # (22525, 136)

    df.to_pickle("../saved_dataframe/contagio_saved_dataframe.pkl")
    return df


def load_CIC_IDS_2017_dataset(file_path):
    df = pd.read_csv(file_path, encoding='utf8', low_memory=False)
    df.columns = df.columns.str.lstrip()

    print(df.shape)  # (2830743, 79)
    print(df.head())
    print(df.describe())
    print(df['Label'].value_counts())

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna()
    df = df.reset_index(drop=True)
    print(df.shape)  # (2827876, 79)

    print(df.select_dtypes(exclude=['int', 'float']))
    # print(list(df.Label.unique()))
    df['Label'] = df['Label'].map(CIC_IDS_2017_label_map)
    print(df['Label'].value_counts())

    # Drop the features which have only 1 unique value:
    nunique = df.apply(pd.Series.nunique)
    cols_to_drop = nunique[nunique == 1].index
    df.drop(cols_to_drop, axis=1, inplace=True)
    print(df.shape)  # (2827876, 71)

    # sample 20%
    df = df.sample(frac=.20, random_state=20)
    print(df.shape)

    df.to_pickle("../saved_dataframe/CIC_IDS_2017_saved_dataframe.pkl")
    return df


def load_CICAndMal2017_dataset(file_path):
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
    df['Label'] = df['Label'].map(CICAndMal2017_label_map)

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

    df.to_pickle("../saved_dataframe/CICAndMal2017_saved_dataframe.pkl")
    return df

    # # One hot Encoding
    # categorical_columns = ['Source IP', 'Destination IP']
    # df_categorical_values = df[categorical_columns]
    # print(df_categorical_values)
    #
    # unique_source_ip = sorted(df['Source IP'].unique())
    # crafted_source_ip = ['Source_IP_' + x for x in unique_source_ip]
    # print(unique_source_ip)
    # print(crafted_source_ip)
    #
    # unique_destination_ip = sorted(df['Destination IP'].unique())
    # crafted_destination_ip = ['Destination_IP_' + x for x in unique_destination_ip]
    # print(unique_destination_ip)
    # print(crafted_destination_ip)
    #
    # dummy_cols = crafted_source_ip + crafted_destination_ip
    # print(dummy_cols)
    #
    # df_categorical_values_encode = df_categorical_values.apply(LabelEncoder().fit_transform)
    # # print(df_categorical_values_encode)
    #
    # oneHotEncoder = OneHotEncoder()
    # df_categorical_values_onehot = oneHotEncoder.fit_transform(df_categorical_values_encode)
    # df_cat_data = pd.DataFrame(df_categorical_values_onehot.toarray(), columns=dummy_cols)
    #
    # # print(df_cat_data)
    # df.drop('Source IP', axis=1, inplace=True)
    # df.drop('Destination IP', axis=1, inplace=True)
    #
    # # memory error if join directly
    # # The problem is that when you merge two dataframes,
    # # you need enough memory for both of them, plus the merged one.
    # print("joining dataframes...")
    # # new_df = df.join(df_cat_data) # do not do this
    # df.to_csv("./original_df.csv")
    # df_cat_data.to_csv("./tojoin_df.csv")
    # print("saving done..")
    #
    # exit(-1)
    # # print("save dataframe to file...")
    # # new_df.to_pickle("./saved_dataframe.pkl")
    # # return new_df


def load_CSE_CIC_IDS2018_dataset(file_path):
    # too big to read, even using chunks
    # (16233001, 80)
    # chunk = 1000000
    # chunks = pd.read_csv(file_path, chunksize=chunk, encoding='utf8', low_memory=False)
    # df = pd.concat(chunks)

    # sampling the csv reading
    # Count the lines
    # num_lines = sum(1 for l in open(file_path))
    num_lines = 16233001  # because I know

    # Sample size - in this case ~5%
    size = int(num_lines / 20)

    # The row indices to skip - make sure 0 is not included to keep the header!
    skip_idx = random.sample(range(1, num_lines), num_lines - size)

    df = pd.read_csv(file_path, header=0, skiprows=skip_idx, encoding='utf8', low_memory=False)
    df.columns = df.columns.str.lstrip()
    df.drop_duplicates(keep=False, inplace=True)

    print(df.shape)
    print(df.head())
    print(df['Label'].value_counts())
    print(df['Label'].isnull().sum())

    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    print(df.select_dtypes(exclude=['int', 'float', 'datetime']))

    header_list = list(df.columns.values)
    for col in header_list:
        if col != 'Timestamp' and col != 'Label':
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # second time check, only label left
    print(df.select_dtypes(exclude=['int', 'float', 'datetime']))

    # Drop the features which have only 1 unique value:
    nunique = df.apply(pd.Series.nunique)
    cols_to_drop = nunique[nunique == 1].index
    df.drop(cols_to_drop, axis=1, inplace=True)
    print(df.shape)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna()
    df = df.reset_index(drop=True)
    print(df.shape)

    label_list = list(df.Label.unique())
    label_map = {}
    for label in label_list:
        if label == 'Benign':
            label_map[label] = 0
        else:
            label_map[label] = 1

    print(label_map)
    df['Label'] = df['Label'].map(label_map)

    # third time check, empty now
    print(df.select_dtypes(exclude=['int', 'float', 'datetime']))

    print(df.head())
    print(df['Label'].value_counts())

    df.to_pickle("../saved_dataframe/CSE_CIC_IDS2018_saved_dataframe.pkl")
    return df


def load_NSL_KDD_dataset(train_dataset_path, test_dataset_path):
    df_train = pd.read_csv(train_dataset_path, header=0, names=NSL_KDD_col_names)
    df_test = pd.read_csv(test_dataset_path, header=0, names=NSL_KDD_col_names)

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

    train_dummy_cols = crafted_protocal + crafted_service + crafted_flag
    print(train_dummy_cols)
    assert len(train_dummy_cols) == 84

    unique_service_test = sorted(df_test['service'].unique())
    crafted_service_test = ['service_' + x for x in unique_service_test]
    test_dummy_cols = crafted_protocal + crafted_service_test + crafted_flag
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


def read_contagio_data():
    contagio_filepath = "../data/ContagioPDF/ConsolidateData.csv"

    if path.exists("../saved_dataframe/contagio_saved_dataframe.pkl"):
        df = pd.read_pickle("../saved_dataframe/contagio_saved_dataframe.pkl")
    else:
        df = load_contagio_dataset(contagio_filepath)

    print(df.shape)
    print(df['class'].value_counts())

    X = df.drop(['class'], axis=1)
    y = df['class']

    print(X.shape)
    print(y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    scaler1 = preprocessing.StandardScaler().fit(X_train)
    X_train_scaled = scaler1.transform(X_train)

    scaler2 = preprocessing.StandardScaler().fit(X_test)
    X_test_scaled = scaler2.transform(X_test)

    print(y_train.value_counts())
    print(y_test.value_counts())
    print(X_train_scaled.shape)
    print(X_test_scaled.shape)

    return X_train_scaled, X_test_scaled, y_train, y_test


def read_CIC_IDS_2017_data():
    CIC_IDS_2017_filepath = "../data/CIC-IDS-2017/ConsolidateData.csv"

    labels = ['BENIGN', 'PortScan', 'FTP-Patator', 'SSH-Patator', 'Bot', 'Infiltration', 'Web Attack � Brute Force',
              'Web Attack � XSS', 'Web Attack � Sql Injection', 'DDoS', 'DoS slowloris', 'DoS Slowhttptest', 'DoS Hulk',
              'DoS GoldenEye', 'Heartbleed']
    label_map = {}

    for label in labels:
        if label == 'BENIGN':
            label_map[label] = 0
        else:
            label_map[label] = 1

    # print(label_map)

    if path.exists("../saved_dataframe/CIC_IDS_2017_saved_dataframe.pkl"):
        df = pd.read_pickle("../saved_dataframe/CIC_IDS_2017_saved_dataframe.pkl")
    else:
        df = load_CIC_IDS_2017_dataset(CIC_IDS_2017_filepath)

    print(df.shape)
    print(df['Label'].value_counts())

    X = df.drop(['Label'], axis=1)
    y = df['Label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    scaler1 = preprocessing.StandardScaler().fit(X_train)
    X_train_scaled = scaler1.transform(X_train)

    scaler2 = preprocessing.StandardScaler().fit(X_test)
    X_test_scaled = scaler2.transform(X_test)

    print(y_train.value_counts())
    print(y_test.value_counts())
    print(X_train_scaled.shape)
    print(X_test_scaled.shape)

    return X_train_scaled, X_test_scaled, y_train, y_test


def read_CICAndMal_data():
    CICAndMal2017_filepath = "../data/CICAndMal2017/ConsolidateData.csv"

    if path.exists("../saved_dataframe/CICAndMal2017_saved_dataframe.pkl"):
        df = pd.read_pickle("../saved_dataframe/CICAndMal2017_saved_dataframe.pkl")
    else:
        df = load_CICAndMal2017_dataset(CICAndMal2017_filepath)

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

    scaler1 = preprocessing.StandardScaler().fit(X_train)
    X_train_scaled = scaler1.transform(X_train)

    scaler2 = preprocessing.StandardScaler().fit(X_test)
    X_test_scaled = scaler2.transform(X_test)

    print(y_train.value_counts())
    print(y_test.value_counts())
    print(X_train_scaled.shape)
    print(X_test_scaled.shape)

    return X_train_scaled, X_test_scaled, y_train, y_test


def read_CSE_CIC_IDS_2018_data():
    CSE_CIC_IDS2018_filepath = "../data/CSE-CIC-IDS2018/ConsolidateData.csv"

    if path.exists("../saved_dataframe/CSE_CIC_IDS2018_saved_dataframe.pkl"):
        df = pd.read_pickle("../saved_dataframe/CSE_CIC_IDS2018_saved_dataframe.pkl")
    else:
        df = load_CSE_CIC_IDS2018_dataset(CSE_CIC_IDS2018_filepath)

    print(df.shape)
    print(df['Label'].value_counts())

    X = df.drop(['Label', 'Timestamp'], axis=1)
    y = df['Label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    print(y_train.value_counts())
    print(y_test.value_counts())

    scaler1 = preprocessing.StandardScaler().fit(X_train)
    X_train_scaled = scaler1.transform(X_train)

    scaler2 = preprocessing.StandardScaler().fit(X_test)
    X_test_scaled = scaler2.transform(X_test)

    print(y_train.value_counts())
    print(y_test.value_counts())
    print(X_train_scaled.shape)
    print(X_test_scaled.shape)

    return X_train_scaled, X_test_scaled, y_train, y_test


def read_NSL_KDD_data():
    train_path = os.path.join("../data/NSL-KDD", "KDDTrain+.csv")
    test_path = os.path.join("../data/NSL-KDD", "KDDTest+.csv")

    df_train, df_test = load_NSL_KDD_dataset(train_path, test_path)
    print("training data shape:", df_train.shape)
    print("training data label shape: ", df_train['class'].value_counts())
    print("testing data shape:", df_test.shape)
    print("testing data label shape: ", df_test['class'].value_counts())

    X_train = df_train.drop(['class'], axis=1)
    y_train = df_train['class']
    X_test = df_test.drop(['class'], axis=1)
    y_test = df_test['class']

    # pre-processing
    scaler1 = preprocessing.StandardScaler().fit(X_train)
    X_train_scaled = scaler1.transform(X_train)

    scaler2 = preprocessing.StandardScaler().fit(X_test)
    X_test_scaled = scaler2.transform(X_test)

    print(y_train.value_counts())
    print(y_test.value_counts())
    print(X_train_scaled.shape)
    print(X_test_scaled.shape)

    return X_train_scaled, X_test_scaled, y_train, y_test
