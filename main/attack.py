import numpy as np
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
from model import attack_model
from art.attacks.evasion import FastGradientMethod
import tensorflow as tf
import random


# @tf.function
def adversarial_pattern(X_train_each, y_train_each, input_model):
    X_train_each = tf.cast(X_train_each, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(X_train_each)
        prediction = input_model(X_train_each)
        loss = tf.keras.losses.binary_crossentropy(y_train_each, prediction)

    gradient = tape.gradient(loss, X_train_each)
    signed_grad = tf.sign(gradient)
    return signed_grad


def FGSM(X_test, y_test):
    epsilon = 0.5
    clip_min = 0.0
    clip_max = 1.0

    X_input = X_test
    y_test = y_test.to_numpy()
    y_input = y_test.reshape((-1, 1))

    model = attack_model()

    # determine how many dataset to be perturbed
    batch_size = X_input.shape[0]

    for i in range(batch_size):
        # print("batch: ", i)
        if batch_size > 100 and i % 100 == 0:
            print("batch: ", i, " percentage: ", i / batch_size)

        # N = random.randint(0, X_train.shape[0]-1)
        X_input_each = X_input[i].reshape((1, X_input.shape[1]))
        y_input_each = y_input[i].reshape((1, 1))

        signed_grad = adversarial_pattern(X_input_each, y_input_each, model)
        X_input_each_adv = X_input_each + signed_grad * epsilon

        # clip function
        X_input_each_adv = tf.clip_by_value(X_input_each_adv, clip_min, clip_max)

        X_input[i] = X_input_each_adv

    return X_input, y_input


def BIM_A(X_test, y_test):
    epsilon = 0.5
    clip_min = 0.0
    clip_max = 1.0
    iterations = 10

    X_input = X_test
    y_test = y_test.to_numpy()
    y_input = y_test.reshape((-1, 1))

    model = attack_model()
    batch_size = X_input.shape[0]

    # iterate each line
    for i in range(batch_size):
        if batch_size > 100 and i % 100 == 0:
            print("batch: ", i, " percentage: ", i / batch_size)

        X_input_each = X_input[i].reshape((1, X_input.shape[1]))
        y_input_each = y_input[i].reshape((1, 1))

        for k in range(iterations):
            # print("row :", i, " iteration: ", k)
            signed_grad = adversarial_pattern(X_input_each, y_input_each, model)

            # the problem is that X_train_each_adv might be the same every time
            # add a small noise
            X_input_each_adv = X_input_each + signed_grad * epsilon + random.uniform(-0.5, 0.5)

            # clip function
            X_input_each_adv = tf.clip_by_value(X_input_each_adv, clip_min, clip_max)

            X_input_each_adv = X_input_each_adv.numpy()

            prediction = model.predict_classes(X_input_each_adv)
            # print("prediction: ", prediction, " actual: ", y_train_each)
            if not np.equal(prediction, y_input_each):
                break

        X_input[i] = X_input_each_adv

    return X_input, y_input


def BIM_B(X_test, y_test):
    epsilon = 0.5
    clip_min = 0.0
    clip_max = 1.0
    iterations = 10

    X_input = X_test
    y_test = y_test.to_numpy()
    y_input = y_test.reshape((-1, 1))

    model = attack_model()

    batch_size = X_input.shape[0]

    # iterate each line
    for i in range(batch_size):
        if batch_size > 100 and i % 100 == 0:
            print("batch: ", i, " percentage: ", i / batch_size)

        X_input_each = X_input[i].reshape((1, X_input.shape[1]))
        y_input_each = y_input[i].reshape((1, 1))

        for k in range(iterations):
            # print("row :", i, " iteration: ", k)
            signed_grad = adversarial_pattern(X_input_each, y_input_each, model)

            # the problem is that X_train_each_adv might be the same every time
            # add a small noise
            X_input_each_adv = X_input_each + signed_grad * epsilon + random.uniform(-0.5, 0.5)
            # clip function
            X_input_each_adv = tf.clip_by_value(X_input_each_adv, clip_min, clip_max)

            X_input_each_adv = X_input_each_adv.numpy()

            # prediction = model.predict_classes(X_train_each_adv)
            # print("prediction: ", prediction, " actual: ", y_train_each)

        X_input[i] = X_input_each_adv

    return X_input, y_input


def JSMA(X_test, y_test):
    pass


def DeepFool(X_test, y_test):
    pass


def CW(X_test, y_test):
    pass
