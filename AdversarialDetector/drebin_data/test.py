from tensorflow.python.client import device_lib
import numpy as np
import gc, sys
from keras.utils import np_utils
from os import path

if (path.exists("./X_write.npy")):
    print("yes")
sys.exit(-1)

# print(device_lib.list_local_devices())
# X = np.concatenate((ben_data, mal_data), axis=0)
# X = np.append(ben_data,mal_data, axis=0)

ben_path = '../drebin_data/ben_matrix.npy'
mal_path = '../drebin_data/mal_matrix.npy'

ben_data = np.load(ben_path, mmap_mode='r+')
mal_data = np.load(mal_path, mmap_mode='r+')

ben_row_len = ben_data.shape[0]
mal_row_len = mal_data.shape[0]
row_len = ben_row_len + mal_row_len
col_len = ben_data.shape[1]

print("saving")
X_write = np.zeros((row_len, col_len), dtype=np.int8)
np.save('X_write', X_write)

print("loading...")
X = np.memmap('./X_write.npy', dtype=np.int8, mode='r+', shape=(row_len, col_len))
print("load done!")

for i in range(ben_row_len):
    X[i] = ben_data[i]

X.flush()
# del ben_data
# gc.collect()

print("half done...")

for j in range(mal_row_len):
    X[ben_row_len + j] = mal_data[j]

X.flush()
# del mal_data
# gc.collect()

print(X.shape[0])
print("X done...")
print("")

# sys.exit(-1)
print("=========================")
print("")

ben_lab = np.zeros((ben_data.shape[0]), dtype=np.int8)
mal_lab = np.ones((mal_data.shape[0]), dtype=np.int8)
ben_lab = np_utils.to_categorical(ben_lab, 2)
mal_lab = np_utils.to_categorical(mal_lab, 2)

print("saving")
Y_write = np.zeros((ben_lab.shape[0] + mal_lab.shape[0], ben_lab.shape[1]), dtype=np.int8)
np.save('Y_write', Y_write)

print("loading...")
Y = np.memmap('./Y_write.npy', dtype=np.int8, mode='r+', shape=(ben_lab.shape[0] + mal_lab.shape[0], ben_lab.shape[1]))
print("load done!")

for i in range(ben_lab.shape[0]):
    Y[i] = ben_lab[i]

Y.flush()
# del ben_lab
# gc.collect()

print("half done...")

for j in range(mal_lab.shape[0]):
    Y[ben_lab.shape[0] + j] = mal_lab[j]

Y.flush()
# del mal_data
# gc.collect()

print(Y.shape[0])
print("Y done...")
print("")

# a = np.array([[1, 2], [3, 4]])
# b = np.array([[5, 6]])
# c = np.concatenate((a, b), axis=0)
# print(c)
#
# z = np.zeros((a.shape[0] + b.shape[0], a.shape[1]), dtype=np.int8)
#
# for i in range(a.shape[0]):
#     z[i] = a[i]
#
# for j in range(b.shape[0]):
#     z[a.shape[0] + j] = b[j]
#
# print(np.array_equal(c, z))
# print(z)
#
#
# print("--------------")
#
# print(a.shape)
# print("sdfsf", a[0])
# print("sdfsf", a[1])
# print(b.shape)
# print("    ")
# # print(a.shape)
# # print(b.shape)
#
# d = np.vstack([a,b])
# print(d)
#
# e = np.r_[a,b]
# print("")
# print(e)
#
# print(np.array_equal(c, d))
# print(np.array_equal(c, e))
# print(np.array_equal(d, e))
#
# f = np.append(a, b, axis=0)
# print(np.array_equal(c, f))
# print("here")
