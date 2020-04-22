import sys
import os
import pickle
import re
import numpy as np
import pandas as pd
import pickle
from keras.layers import Input, Dense, Dropout, Activation
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import keras
import tensorflow as tf

dir_path = '../data/Drebin/feature_vectors'
ground_truth_path = '../data/Drebin/sha256_family.csv'
benign_matrix_path = './ben_matrix.npy'
malware_matrix_path = './mal_matrix.npy'
merged_matrix_path = './merged_matrix.npy'


def extract_features(dir_path):
    f_count = 0
    feat_set = dict()

    for filename in os.listdir(dir_path):
        lines = [l for l in open(os.path.join(dir_path, filename), 'r')]
        if len(lines) > 0:  # not an empty file
            for eachLine in lines:
                eachLine = eachLine.replace('\n', '')
                if len(eachLine.split('::')) == 2:
                    features = eachLine.split('::')
                    # 545246 -> 542710 -> 535896 -> 532793 -> 532775
                    # -> 532730 -> 532559 -> 528196
                    if not re.search('[a-zA-Z0-9]+', features[1]):  # remove lines with not letter or number
                        break
                    if features[1].endswith('/'):
                        features[1] = features[1][:-1]
                    while features[1].endswith('.'):
                        features[1] = features[1][:-1]
                    if features[1].endswith(':'):
                        features[1] = features[1][:-1]

                    if features[1].startswith('.'):
                        features[1] = features[1][1:]
                    if features[1].startswith('_'):
                        features[1] = features[1][1:]
                    if features[1].startswith(':'):
                        features[1] = features[1][1:]

                    crafted_feature = features[0] + "_" + features[1]
                    crafted_feature = crafted_feature.lower()

                    if crafted_feature not in feat_set:
                        feat_set[crafted_feature] = f_count
                        f_count += 1

    print(f_count)  # 528196

    with open('./drebin_features.pkl', 'wb') as f:
        pickle.dump(feat_set, f, pickle.HIGHEST_PROTOCOL)

    print(len(feat_set))  # 528196


# create malware and benign matrix
def create_vectors(ground_truth_path):
    # remove the first line from original file, "sha256, family"
    ground_truth = [line.split(',')[0] for line in open(ground_truth_path, 'r')]
    print(ground_truth)

    count = 0
    for filename in os.listdir(dir_path):
        if filename in ground_truth:
            count = count + 1

    print(count)  # 5560

    feat_dict = pickle.load(open('./drebin_features.pkl', 'rb'))
    assert len(feat_dict) == 528196

    feature_dimension = len(feat_dict)

    malware_count = 5560
    benign_count = 129013 - 5560

    # initialize two matrix
    ben_matrix = np.zeros((benign_count, feature_dimension + 1), dtype=np.int8)  # the extra one is label
    mal_matrix = np.zeros((malware_count, feature_dimension + 1), dtype=np.int8)
    merged_matrix = np.zeros((benign_count + malware_count, feature_dimension + 1), dtype=np.int8)

    index_benign = 0
    total_index = 0

    # benign
    for filename in os.listdir(dir_path):
        if filename not in ground_truth:
            lines = [line for line in open(os.path.join(dir_path, filename), 'r')]
            if len(lines) > 0:  # not an empty file
                for eachLine in lines:
                    eachLine = eachLine.replace('\n', '')

                    if len(eachLine.split('::')) == 2:
                        features = eachLine.split('::')

                        if not re.search('[a-zA-Z0-9]+', features[1]):  # remove lines with not letter or number
                            break

                        if features[1].endswith('/'):
                            features[1] = features[1][:-1]
                        while features[1].endswith('.'):
                            features[1] = features[1][:-1]
                        if features[1].endswith(':'):
                            features[1] = features[1][:-1]

                        if features[1].startswith('.'):
                            features[1] = features[1][1:]
                        if features[1].startswith('_'):
                            features[1] = features[1][1:]
                        if features[1].startswith(':'):
                            features[1] = features[1][1:]

                        crafted_feature = features[0] + "_" + features[1]
                        crafted_feature = crafted_feature.lower()

                        # print(feat_dict[crafted_feature])
                        # ben_matrix[index_benign, feat_dict[crafted_feature]] = 1
                        # ben_matrix[index_benign, feature_dimension] = 0
                        merged_matrix[total_index, feat_dict[crafted_feature]] = 1
                        merged_matrix[total_index, feature_dimension] = 0  # label

            # index_benign += 1
            total_index += 1

    print(total_index)

    for _, filename in enumerate(ground_truth):
        lines = [line for line in open(os.path.join(dir_path, filename), 'r')]
        if len(lines) > 0:  # not an empty file
            for eachLine in lines:
                eachLine = eachLine.replace('\n', '')

                if len(eachLine.split('::')) == 2:
                    features = eachLine.split('::')

                    if not re.search('[a-zA-Z0-9]+', features[1]):  # remove lines with not letter or number
                        break

                    if features[1].endswith('/'):
                        features[1] = features[1][:-1]
                    while features[1].endswith('.'):
                        features[1] = features[1][:-1]
                    if features[1].endswith(':'):
                        features[1] = features[1][:-1]

                    if features[1].startswith('.'):
                        features[1] = features[1][1:]
                    if features[1].startswith('_'):
                        features[1] = features[1][1:]
                    if features[1].startswith(':'):
                        features[1] = features[1][1:]

                    crafted_feature = features[0] + "_" + features[1]
                    crafted_feature = crafted_feature.lower()

                    # mal_matrix[malware_index, feat_dict[crafted_feature]] = 1
                    # mal_matrix[malware_index, feature_dimension] = 1
                    merged_matrix[total_index, feat_dict[crafted_feature]] = 1
                    merged_matrix[total_index, feature_dimension] = 1
        total_index += 1

    print(total_index)

    # np.save('mal_matrix', mal_matrix)
    # np.save('ben_matrix', ben_matrix)
    np.save('merged_matrix', merged_matrix)


# convert matrix to dataframe, add label column to each dataframe and merge two dataframes
def merge_matrix(benign_matrix_path, malware_matrix_path):
    ben_data = np.load(benign_matrix_path, mmap_mode='r+')
    mal_data = np.load(malware_matrix_path, mmap_mode='r+')

    print(ben_data[:, -1])  # last column all 0
    print(ben_data.shape)
    print(mal_data[:, -1])  # last column all 1
    print(mal_data.shape)

    merged_matrix = np.zeros((ben_data.shape[0] + mal_data.shape[0], ben_data.shape[1]), dtype=np.int8)
    print(merged_matrix.shape)
    np.save('merged_matrix', merged_matrix)

    print("writing benign matrix...")
    merged = np.memmap('./merged_matrix.npy', dtype=np.int8, mode='r+',
                       shape=(ben_data.shape[0] + mal_data.shape[0], ben_data.shape[1]))

    for i in range(ben_data.shape[0]):
        merged[i] = ben_data[i]

    merged.flush()

    print("writing malware matrix...")
    for j in range(mal_data.shape[0]):
        merged[ben_data.shape[0] + j] = mal_data[j]

    merged.flush()
    print(merged.shape)

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


def create_dataframe(merged_matrix_path):
    merged_matrix = np.load(merged_matrix_path, mmap_mode='r+')
    print(merged_matrix.shape)
    print(merged_matrix[:, -1])
    print(type(merged_matrix))

    df = pd.DataFrame(merged_matrix)
    print(df.shape)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    print(X.shape)
    print(y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(y_train.value_counts())
    print(y_test.value_counts())

    # pass

    scaler1 = preprocessing.Normalizer().fit(X_train)
    X_train = scaler1.transform(X_train)

    scaler5 = preprocessing.Normalizer().fit(X_test)
    X_test = scaler5.transform(X_test)

    build_models(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    # extract_features(dir_path)
    # create_vectors(ground_truth_path)
    # merge_matrix(benign_matrix_path, malware_matrix_path)
    create_dataframe(merged_matrix_path)





sys.exit(-1)

file_path = "../data/Drebin/feature_vectors/fff1e7c6fdef70ffa3aacceb4ee43de2891cc6ff980366811bb3aaa44867c2df"

left_set = set()
left_set = {'service_receiver', 'permission', 'activity', 'real_permission', 'url', 'provider', 'call', 'api_call',
            'intent', 'feature'}
# {'service_receiver', 'permission', 'activity', 'real_permission', 'url', 'provider', 'call', 'api_call', 'intent', 'feature'}

# for filename in os.listdir(dir_path):
#     with open(os.path.join(dir_path, filename), 'r') as f:
#         for line in f:
#             if len(line.split('::')) > 1:
#                 line = line.replace('\n', '')
#                 feat = line.split('::')
#                 left_set.add(feat[0])
#     f.close()
#
# print(left_set)

service_receiver_file = open('./service_receiver.txt', 'a')
permission_file = open('./permission.txt', 'a')
activity_file = open('./activity.txt', 'a')
real_permission_file = open('./real_permission.txt', 'a')
url_file = open('./url.txt', 'a')
provider_file = open('./provider.txt', 'a')
call_file = open('./call.txt', 'a')
api_call_file = open('./api_call.txt', 'a')
intent_file = open('./intent.txt', 'a')
feature_file = open('./feature.txt', 'a')

# for left in left_set:
#     print(left)

# with open(file_path, 'r') as f:
#     for line in f:
#         line = line.replace('\n', '')
#         feat = line.split('::')
#         if feat[0] == 'service_receiver':
#             service_receiver_file.write(line + "\n")
#         elif feat[0] == 'permission':
#             permission_file.write(line + "\n")
#         elif feat[0] == 'activity':
#             activity_file.write(line + "\n")
#         elif feat[0] == 'real_permission':
#             real_permission_file.write(line + "\n")
#         elif feat[0] == 'url':
#             url_file.write(line.replace('http://', '') + "\n")
#         elif feat[0] == 'provider':
#             provider_file.write(line + "\n")
#         elif feat[0] == 'call':
#             call_file.write(line + "\n")
#         elif feat[0] == 'api_call':
#             api_call_file.write(line + "\n")
#         elif feat[0] == 'intent':
#             intent_file.write(line + "\n")
#         elif feat[0] == 'feature':
#             feature_file.write(line + "\n")
#         else:
#             pass

for filename in os.listdir(dir_path):
    with open(os.path.join(dir_path, filename), 'r') as f:
        for line in f:
            line = line.replace('\n', '')
            feat = line.split('::')
            if feat[0] == 'service_receiver':
                service_receiver_file.write(feat[1] + "\n")
            elif feat[0] == 'permission':
                permission_file.write(feat[1] + "\n")
            elif feat[0] == 'activity':
                activity_file.write(feat[1] + "\n")
            elif feat[0] == 'real_permission':
                real_permission_file.write(feat[1] + "\n")
            elif feat[0] == 'url':
                url_file.write(feat[1] + "\n")
            elif feat[0] == 'provider':
                provider_file.write(feat[1] + "\n")
            elif feat[0] == 'call':
                call_file.write(feat[1] + "\n")
            elif feat[0] == 'api_call':
                api_call_file.write(feat[1] + "\n")
            elif feat[0] == 'intent':
                intent_file.write(feat[1] + "\n")
            elif feat[0] == 'feature':
                feature_file.write(feat[1] + "\n")
            else:
                pass

service_receiver_file.close()
permission_file.close()
activity_file.close()
real_permission_file.close()
url_file.close()
provider_file.close()
call_file.close()
api_call_file.close()
intent_file.close()
feature_file.close()
