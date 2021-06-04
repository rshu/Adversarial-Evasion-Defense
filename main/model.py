import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
import pickle
import keras


def baseline_model(X_train, X_test, y_train, y_test):
    # TensorFlow to automatically choose an existing and supported device
    # to run the operations in case the specified one doesn't exist
    # tf.config.set_soft_device_placement(True)

    classifier = tf.keras.Sequential()

    layers = [
        Dense(X_train.shape[1], input_shape=(X_train.shape[1],)),
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

    for layer in layers:
        classifier.add(layer)

    METRICS = [
        keras.metrics.BinaryAccuracy(name='accuracy'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc'),
        keras.metrics.TruePositives(name='tp'),
        keras.metrics.FalsePositives(name='fp'),
        keras.metrics.TrueNegatives(name='tn'),
        keras.metrics.FalseNegatives(name='fn')
    ]

    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=METRICS)
    classifier.fit(X_train, y_train, batch_size=64, epochs=10)

    # # save the model
    # nn_model_pickel_file = 'nn_model.pkl'
    # pickle.dump(classifier, open(nn_model_pickel_file, 'wb'))
    #
    # # load the model from disk
    # loaded_model = pickle.load(open(nn_model_pickel_file, 'rb'))
    #
    # print("evaluation...")
    # print("")
    loss, acc, precision, recall, auc, tp, fp, tn, fn = classifier.evaluate(X_test, y_test, verbose=2)
    print("loss: " + str(loss))
    print("accuracy: " + str(acc))
    print("precision: " + str(precision) + " recall: " + str(recall) + " auc: " + str(auc))
    print("tp: " + str(tp) + " fp: " + str(fp) + " tn: " + str(tn) + " fn: " + str(fn))
    # print(classifier.evaluate(X_test, y_test, verbose=2))
