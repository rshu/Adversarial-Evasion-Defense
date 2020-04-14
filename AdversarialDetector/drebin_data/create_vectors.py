import pickle as p
import sys
import os
import numpy as np

from scipy.sparse import *
from scipy import *

dir_ = '../../data/Drebin/feature_vectors'
mal_files = [l.split(',')[0] for l in open('../myModels/malware_names.txt', 'r')]
feat_dict = p.load(open('./drebin_features.pkl', 'rb'))

dim = len(feat_dict)
num_total = 129013
num_mal = len(mal_files)
num_clean = num_total - num_mal

# out of memory or cannot allocate large memory issue
# in ubuntu, the default overcommit mode is 0
# cat /proc/sys/vm/overcommit_memory
# need to change to 1
# sudo su
# int root mode, echo 1 > /proc/sys/vm/overcommit_memory
# now the overcommit mode is 1
ben_matrix = np.zeros((num_clean, dim), dtype=np.int8)
mal_matrix = np.zeros((num_mal, dim), dtype=np.int8)
#
# # ben_matrix = np.asarray(np.zeros((num_clean, dim), dtype=np.int8))
# # mal_matrix = np.asarray(np.zeros((num_mal, dim), dtype=np.int8))
#
# ben_matrix = lil_matrix((num_clean, dim), dtype=np.int8)
# mal_matrix = lil_matrix((num_mal, dim), dtype=np.int8)

idx = 0
# Get clean data first
for f in os.listdir(dir_):
    if f not in mal_files:
        lines = [l for l in open(os.path.join(dir_, f), 'r')]
        for l in lines:
            if len(l.split('::')) > 1:
                feat = l.split('::')
                feature = ""
                for fe in feat:
                    feature = fe + feature
                feature = feature.replace('\n', '')
                ben_matrix[idx, feat_dict[feature]] = 1
        idx += 1

for idx, f in enumerate(mal_files):
    lines = [l for l in open(os.path.join(dir_, f), 'r')]
    for l in lines:
        if len(l.split('::')) > 1:
            feat = l.split('::')
            feature = ""
            for fe in feat:
                feature = fe + feature
            feature = feature.replace('\n', '')
            mal_matrix[idx, feat_dict[feature]] = 1

np.save('mal_matrix', mal_matrix)
np.save('ben_matrix', ben_matrix)
