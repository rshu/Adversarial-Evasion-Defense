import pandas as pd
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
# tf.executing_eagerly()
# import tensorflow.compat.v1 as tf
# tf.executing_eagerly()
import keras
from os import path
import pickle, sys
from keras.layers import Input, Dense, Dropout, Activation
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np
import pprint as pprint
from keras import backend as K


from cleverhans.attacks import FastGradientMethod
from cleverhans.utils_keras import KerasModelWrapper

desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 50)


def load_dataset(file_path):
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

    df.to_pickle("./saved_dataframe.pkl")
    return df


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


def fgm(model, x, y_train, eps=0.01, epochs=1, sign=True, clip_min=0., clip_max=1.):
    """
    Fast gradient method.
    See https://arxiv.org/abs/1412.6572 and https://arxiv.org/abs/1607.02533
    for details.  This implements the revised version since the original FGM
    has label leaking problem (https://arxiv.org/abs/1611.01236).
    :param model: A wrapper that returns the output as well as logits.
    :param x: The input placeholder.
    :param eps: The scale factor for noise.
    :param epochs: The maximum epoch to run.
    :param sign: Use gradient sign if True, otherwise use gradient value.
    :param clip_min: The minimum value in output.
    :param clip_max: The maximum value in output.
    :return: A tensor, contains adversarial samples for each input.
    """
    xadv = tf.identity(x)

    ybar = model(xadv)
    print("ybar shape", ybar.get_shape())
    yshape = ybar.get_shape().as_list()
    ydim = yshape[1]

    indices = tf.argmax(ybar, axis=1)
    target = tf.cond(
        tf.equal(ydim, 1),
        lambda: tf.nn.relu(tf.sign(ybar - 0.5)),
        lambda: tf.one_hot(indices, ydim, on_value=1.0, off_value=0.0))

    if 1 == ydim:
        loss_fn = tf.keras.losses.BinaryCrossentropy()
    # else:
    #     loss_fn = tf.nn.softmax_cross_entropy_with_logits

    if sign:
        noise_fn = tf.sign
    else:
        noise_fn = tf.identity

    eps = tf.abs(eps)

    def _cond(xadv, i):
        return tf.less(i, epochs)

    def _body(xadv, i):
        ybar = model(xadv)
        loss = loss_fn(y_train, ybar)
        dy_dx, = tf.gradients(loss, xadv)
        xadv = tf.stop_gradient(xadv + eps*noise_fn(dy_dx))
        xadv = tf.clip_by_value(xadv, clip_min, clip_max)
        return xadv, i+1

    xadv, _ = tf.while_loop(_cond, _body, (xadv, 0), back_prop=False, name='fast_gradient')
    return xadv


def adversarial_pattern(model, image, label):
    image = tf.cast(image, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(image)
        prediction = model(image)
        loss = tf.keras.losses.MSE(label, prediction)

    gradient = tape.gradient(loss, image)

    signed_grad = tf.sign(gradient)

    return signed_grad


def fgsm_pdf(X_train, y_train, eps=0.3, clip_min=0.0, clip_max=1.0):
    """Create FGSM attack points for each row of x"""

    # print(type(X_train))  # <class 'pandas.core.frame.DataFrame'>
    # print(type(y_train))  # <class 'pandas.core.series.Series'>
    # print(X_train.shape)  # (18020, 135)

    print(tf.executing_eagerly())

    model = pickle.load(open('./saved_nn_model.pkl', 'rb'))

    X_train_firstrow = X_train.iloc[0]
    y_train_firstrow = y_train.iloc[0]
    # print(type(X_train_firstrow)) # <class 'pandas.core.series.Series'>
    # print(type(y_train_firstrow)) # <class 'numpy.int64'>

    # print(X_train_firstrow)
    # print(y_train_firstrow)  # 1
    # print(X_train_firstrow.shape)  # (135,)
    # print(y_train_firstrow.shape)  # ()

    X_train_firstrow = tf.convert_to_tensor(X_train_firstrow.to_numpy())
    X_train_firstrow = tf.cast(X_train_firstrow, tf.float32)
    # print(X_train_firstrow)
    X_train_firstrow = tf.reshape(X_train_firstrow, [1, 135])
    print(X_train_firstrow)
    # print(type(X_train_firstrow))  # <class 'tensorflow.python.framework.ops.EagerTensor'>

    y_train_firstrow = tf.convert_to_tensor(y_train_firstrow)
    y_train_firstrow = tf.reshape(y_train_firstrow, [1, 1])
    print(y_train_firstrow)

    # X_train_firstrow = tf.cast(X_train_firstrow, tf.float32)
    # print(X_train_firstrow)

    loss_object = tf.keras.losses.BinaryCrossentropy()

    with tf.GradientTape() as tape:
        tape.watch(X_train_firstrow)
        prediction = model(X_train_firstrow)
        print("y pred shape:", prediction.get_shape())
        print(prediction)
        print(y_train_firstrow)
        print("y shape:", y_train_firstrow.get_shape())
        loss = loss_object(y_train_firstrow, prediction)

    print(loss)
    print(type(loss))
    print(type(X_train_firstrow))
    gradient = tape.gradient(loss, X_train_firstrow)
    signed_grad = tf.sign(gradient)

    sys.exit(-1)


    # print(X_train)
    # print(X_train.describe())

    # sess = tf.Session()
    # # keras.backend.set_session(sess)
    # model = pickle.load(open('./saved_nn_model.pkl', 'rb'))
    # wrap = KerasModelWrapper(model)
    # fgsm = FastGradientMethod(wrap)
    # fgsm_params = {'eps': 0.3,
    #                'clip_min': 0.,
    #                'clip_max': 1.}
    #
    # adv_x = fgsm.generate(X_train, **fgsm_params)
    # sys.exit(-1)

    X_train = X_train.astype('float32')
    X_train = tf.convert_to_tensor(X_train.to_numpy())
    print(X_train)
    print("X_train type", type(X_train))

    model = pickle.load(open('./saved_nn_model.pkl', 'rb'))
    print(model)

    loss_object = tf.keras.losses.BinaryCrossentropy()

    for i in range(X_train.get_shape()[0]):
        with tf.GradientTape() as tape:
            tape.watch(X_train)
            pred = model(X_train)
            print("pred shape:", pred.get_shape())
            print("x shape:", X_train.get_shape())
            loss = loss_object(X_train, pred)
        gradient = tape.gradient(loss, X_train)
        signed_grad = tf.sign(gradient)

    sys.exit(-1)



    X_train_array = X_train.to_numpy()
    X_train_tensor = tf.convert_to_tensor(X_train_array, dtype=tf.float32)

    y_train_array = y_train.to_numpy().reshape(18020, 1)
    print(y_train_array)
    print("y_train_array shape:", y_train_array.shape)
    y_train_tensor = tf.convert_to_tensor(y_train_array, dtype=tf.int8)

    model = pickle.load(open('./saved_nn_model.pkl', 'rb'))
    print(model(X_train_tensor))
    # X_train_adv = fgm(model, X_train_tensor, y_train, eps=0.01, epochs=1, sign=True, clip_min=0.0, clip_max=1.0)
    loss_object = tf.keras.losses.BinaryCrossentropy()

    with tf.GradientTape() as tape:
        tape.watch(X_train_tensor)
        pred = model(X_train_tensor)
        print(pred.get_shape())
        loss = loss_object(y_train_array, pred)
        print(loss)
    gradient = tape.gradient(loss, X_train_tensor)
    signed_grad = tf.sign(gradient)


    pass



if __name__ == "__main__":
    file_path = "../data/ContagioPDF/ConsolidateData.csv"

    if path.exists("saved_dataframe.pkl"):
        df = pd.read_pickle("./saved_dataframe.pkl")
    else:
        df = load_dataset(file_path)

    # print(df.shape)
    # print(df['class'].value_counts())

    X = df.drop(['class'], axis=1)
    y = df['class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)


    fgsm_pdf(X_train, y_train)
    sys.exit(-1)

    print(y_train.value_counts())
    print(y_test.value_counts())


    scaler1 = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler1.transform(X_train)

    scaler5 = preprocessing.StandardScaler().fit(X_test)
    X_test = scaler5.transform(X_test)

    build_models(X_train, y_train, X_test, y_test)
