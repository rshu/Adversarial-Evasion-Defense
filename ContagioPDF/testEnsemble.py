import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from deepstack.base import KerasMember
from deepstack.ensemble import DirichletEnsemble
from sklearn.metrics import accuracy_score
from deepstack.ensemble import StackEnsemble
from sklearn.ensemble import RandomForestRegressor
import keras
import numpy as np

model_index = [1, 2, 3, 4, 5]
print(model_index)

dataset = pd.read_pickle("./saved_dataframe.pkl")

X = dataset.drop(['class'], axis=1)
y = dataset['class']

# train: 0.6, validate: 0.2, test: 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, stratify=y_train, test_size=0.25,
                                                            random_state=42)

X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()
X_validate = X_validate.to_numpy()
y_validate = y_validate.to_numpy()

y_train = y_train.reshape((-1, 1))
y_test = y_test.reshape((-1, 1))
y_validate = y_validate.reshape((-1, 1))

# pre-processing
scaler1 = preprocessing.StandardScaler().fit(X_train)
X_train = scaler1.transform(X_train)

scaler2 = preprocessing.StandardScaler().fit(X_test)
X_test = scaler2.transform(X_test)

scaler3 = preprocessing.StandardScaler().fit(X_validate)
X_validate = scaler3.transform(X_validate)

# X_train.fillna(X_train.mean(), inplace=True)
# X_validate.fillna(X_validate.mean(), inplace=True)
# X_test.fillna(X_test.mean(), inplace=True)

print(X_train)

model1 = "./pickles/bayesOpt_nn_model_" + str(model_index[0]) + ".h5"
classifier1 = keras.models.load_model(model1)
member1 = KerasMember(name='model1', keras_model=classifier1, train_batches=(X_train, y_train),
                      val_batches=(X_validate, y_validate))

model2 = "./pickles/bayesOpt_nn_model_" + str(model_index[1]) + ".h5"
classifier2 = keras.models.load_model(model2)
member2 = KerasMember(name='model2', keras_model=classifier2, train_batches=(X_train, y_train),
                      val_batches=(X_validate, y_validate))

model3 = "./pickles/bayesOpt_nn_model_" + str(model_index[2]) + ".h5"
classifier3 = keras.models.load_model(model3)
member3 = KerasMember(name='model3', keras_model=classifier3, train_batches=(X_train, y_train),
                      val_batches=(X_validate, y_validate))

model4 = "./pickles/bayesOpt_nn_model_" + str(model_index[3]) + ".h5"
classifier4 = keras.models.load_model(model4)
member4 = KerasMember(name='model4', keras_model=classifier4, train_batches=(X_train, y_train),
                      val_batches=(X_validate, y_validate))

model5 = "./pickles/bayesOpt_nn_model_" + str(model_index[4]) + ".h5"
classifier5 = keras.models.load_model(model5)
member5 = KerasMember(name='model' + str(5), keras_model=classifier5, train_batches=(X_train, y_train),
                      val_batches=(X_validate, y_validate))

stack = StackEnsemble()
stack.model = RandomForestRegressor(verbose=0, n_estimators=200,
                                    max_depth=15, n_jobs=20, min_samples_split=20)
# stack.add_members([member1, member2, member3, member4, member5])
stack.add_member(member1)
stack.add_member(member2)
stack.fit()
print("here")
stack.describe(metric=accuracy_score)
