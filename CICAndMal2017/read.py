import pandas as pd
import sys, collections
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import pickle

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
    df = df.drop([df.columns[0]], axis=1)
    # remove white space at the beginning of string in dataframe header
    df.columns = df.columns.str.lstrip()
    # print(list(df.columns))

    # # print(df.dtypes())
    print(df.shape)
    print(df.head())
    print(df.describe())
    # print(list(df.Label.unique()))
    print(df['Label'].value_counts())
    print(df['Label'].isnull().sum())

    # Drop rows with values missing for label column (target variable)
    # df = df.dropna(subset=['Label'], inplace=True)
    df = df.dropna()
    df = df.reset_index(drop=True)
    # print(df['Label'].isnull().sum())
    print(df.shape)  # (2616566, 84)

    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    print(df.select_dtypes(exclude=['int', 'float', 'datetime']))

    df['Packet Length Std'] = pd.to_numeric(df['Packet Length Std'], errors='coerce')
    df['CWE Flag Count'] = pd.to_numeric(df['CWE Flag Count'], errors='coerce')
    df['Active Mean'] = pd.to_numeric(df['Active Mean'], errors='coerce')
    df['Flow IAT Mean'] = pd.to_numeric(df['Flow IAT Mean'], errors='coerce')
    df['URG Flag Count'] = pd.to_numeric(df['URG Flag Count'], errors='coerce')
    df['Down/Up Ratio'] = pd.to_numeric(df['Down/Up Ratio'], errors='coerce')
    df['Fwd Avg Bytes/Bulk'] = pd.to_numeric(df['Fwd Avg Bytes/Bulk'], errors='coerce')
    df['Flow IAT Min'] = pd.to_numeric(df['Flow IAT Min'], errors='coerce')
    print(df.select_dtypes(exclude=['int', 'float', 'datetime']))

    df['Source IP'] = df['Source IP'].astype('str')
    df['Destination IP'] = df['Destination IP'].astype('str')
    df['Label'] = df['Label'].map(label_map)

    # encoder
    df["Source IP"] = LabelEncoder().fit_transform(df["Source IP"])
    df["Destination IP"] = LabelEncoder().fit_transform(df["Destination IP"])

    # Drop the features which have only 1 unique value:
    nunique = df.apply(pd.Series.nunique)
    cols_to_drop = nunique[nunique == 1].index
    df.drop(cols_to_drop, axis=1, inplace=True)
    print(df.shape)  # (2616566, 72)

    df.to_pickle("./df.pkl")
    return df


if __name__ == "__main__":
    # print(len(col_names))
    # print(len(set(col_names)))
    # print([item for item, count in collections.Counter(col_names).items() if count > 1])

    file_path = "../data/CICAndMal2017/ConsolidateData.csv"
    df = pd.read_pickle("./df.pkl")
    # df = load_dataset(file_path)
    X = df.drop(['Label', 'Timestamp'], axis=1)
    y = df['Label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25)
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    model_pickel_file = 'my_model.pkl'
    pickle.dump(clf, open(model_pickel_file, 'wb'))

    # load the model from disk
    loaded_model = pickle.load(open(model_pickel_file, 'rb'))

    y_pred = loaded_model.predict(X_test)

    print("Accuracy is %f." % accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print("Precision score is %f." % precision_score(y_test, y_pred))
    print("Recall score is %f." % recall_score(y_test, y_pred))
    print("F1 score is %f." % f1_score(y_test, y_pred))

    # probability=True should be set, default is false
    probs = clf.predict_proba(X_test)
    preds = probs[:, 1]
    fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
    roc_auc = metrics.auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
