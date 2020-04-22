import sys
import os
import pickle
import re
import numpy as np

dir_path = '../data/Drebin/feature_vectors'
ground_truth_path = '../data/Drebin/sha256_family.csv'


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
    ben_matrix = np.zeros((benign_count, feature_dimension), dtype=np.int8)
    mal_matrix = np.zeros((malware_count, feature_dimension), dtype=np.int8)

    index_benign = 0

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

                        ben_matrix[index_benign, feat_dict[crafted_feature]] = 1
            index_benign += 1

    for malware_index, filename in enumerate(ground_truth):
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

                    mal_matrix[malware_index, feat_dict[crafted_feature]] = 1

    np.save('mal_matrix', mal_matrix)
    np.save('ben_matrix', ben_matrix)


# convert matrix to dataframe, add label column to each dataframe and merge two dataframes
def create_dataframe():
    pass


if __name__ == "__main__":
    # extract_features(dir_path)
    create_vectors(ground_truth_path)

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
