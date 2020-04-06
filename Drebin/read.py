"""
Reading function for each of the classification problems
read - malware vs non malware
read_multiclass - family classification

Inside each function, customize your data path
"""
import numpy as np
import sys
from Drebin import feature_extraction
from os import listdir
from os.path import isfile, join


def read(load_data=False):
    if load_data:
        print("Previous data not loaded. Attempt to read data ...")
        mypath = "../data/Drebin/feature_vectors"
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

        print("Reading csv file for ground truth ...")
        ground_truth = np.loadtxt("../data/Drebin/sha256_family.csv", delimiter=",", skiprows=1, dtype=str)
        print("ground_truth shape:", ground_truth.shape)
        families = np.unique(ground_truth[:, 1])
        print(families)
        print(len(families))

        print("Reading positive and negative texts ...")
        pos = []
        neg = []
        for virus in onlyfiles:
            if virus in ground_truth[:, 0]:
                pos.append(virus)  # 5560
            else:
                # if len(neg) < 5560:
                neg.append(virus)  # 123453

        print("Extracting features ...")
        x = []
        y = []
        for text_file in pos:
            sys.stdin = open("%s/%s" % (mypath, text_file))
            features = sys.stdin.readlines()
            sample = feature_extraction.count_feature_set(features)
            x.append(sample)
            y.append(1)

        for text_file in neg:
            sys.stdin = open("%s/%s" % (mypath, text_file))
            features = sys.stdin.readlines()
            sample = feature_extraction.count_feature_set(features)
            x.append(sample)
            y.append(0)

        print("Data is read successfully:")
        x = np.array(x)
        y = np.array(y)
        print(x.shape, y.shape)

        # store the input array in a disk file with npy extension
        print("Saving data under data_numpy directory ...")
        np.save("../data/Drebin/data_numpy/x_all.npy", x)
        np.save("../data/Drebin/data_numpy/y_all.npy", y)

        return x, y
    else:
        print("Loading previous data ...")
        x_ = np.load("../data/Drebin/data_numpy/x_all.npy")
        y_ = np.load("../data/Drebin/data_numpy/y_all.npy")
        print(x_.shape, y_.shape)
        # print x == x_, y == y_
        return x_, y_


def read_multiclass(load_data=False):
    if load_data:
        print("Previous data not loaded. Attempt to read data ...")
        mypath = "../data/Drebin/feature_vectors"
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

        print("Reading csv file for ground truth ...")
        ground_truth = np.loadtxt("../data/Drebin/sha256_family.csv", delimiter=",", skiprows=1, dtype=str)
        families = np.unique(ground_truth[:, 1])
        classes = map_family_to_category(families)
        # print families
        # print len(families)

        print("Reading positive texts ...")
        pos = []
        for virus in onlyfiles:
            if virus in ground_truth[:, 0]:
                pos.append(virus)

        print("Extracting features ...")
        x = []
        y = []
        for i in range(ground_truth.shape[0]):
            sys.stdin = open("%s/%s" % (mypath, ground_truth[i, 0]))
            features = sys.stdin.readlines()
            sample = feature_extraction.count_feature_set(features)
            x.append(sample)
            y.append(classes[ground_truth[i, 1]])

        print("Data is read successfully:")
        x = np.array(x)
        y = np.array(y)
        print(x.shape, y.shape)

        print("Saving data under data_numpy directory ...")
        np.save("../data/Drebin/data_numpy/x_multi_all.npy", x)
        np.save("../data/Drebin/data_numpy/y_multi_all.npy", y)

        return x, y
    else:
        print("Loading previous data ...")
        x_ = np.load("../data/Drebin/data_numpy/x_multi_all.npy")
        y_ = np.load("../data/Drebin/data_numpy/y_multi_all.npy")
        print(x_.shape, y_.shape)

        return x_, y_


def map_family_to_category(families):
    out = {}
    count = 1
    for family in families:
        out[family] = count
        count += 1
    return out


if __name__ == "__main__":
    # read - malware vs non malware
    # read_multiclass - family classification
    # read_multiclass(load_data=True)
    read(load_data=True)
