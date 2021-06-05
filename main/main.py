import sys, sklearn
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.client import device_lib
import logging
from datetime import datetime
from read_data import read_NSL_KDD_data, read_CIC_IDS_2017_data, read_contagio_data, read_CSE_CIC_IDS_2018_data, \
    read_CICAndMal_data
from model import baseline_model

log_file_name = datetime.now().strftime("%Y%m%d-%H%M%S")
logging.basicConfig(filename="../logs/" + log_file_name, filemode='w', format='%(name)s - %(asctime)s - %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def run(choose_dataset):
    if choose_dataset == "ContagioPDF":
        X_train_scaled, X_test_scaled, y_train, y_test = read_contagio_data()
    elif choose_dataset == "CIC-IDS-2017":
        X_train_scaled, X_test_scaled, y_train, y_test = read_CIC_IDS_2017_data()
    elif choose_dataset == "CICAndMal2017":
        X_train_scaled, X_test_scaled, y_train, y_test = read_CICAndMal_data()
    elif choose_dataset == "CSE-CIC-IDS2018":
        X_train_scaled, X_test_scaled, y_train, y_test = read_CSE_CIC_IDS_2018_data()
    elif choose_dataset == "NSL-KDD":
        X_train_scaled, X_test_scaled, y_train, y_test = read_NSL_KDD_data()
    else:
        pass

    return X_train_scaled, X_test_scaled, y_train, y_test


if __name__ == "__main__":
    logger.info('---------------INFO---------------')
    # find out which devices your operations and tensors are assigned to
    tf.debugging.set_log_device_placement(True)

    logger.info("sys.version: " + sys.version)
    logger.info("pandas version: " + pd.__version__)
    logger.info("numpy version: " + np.__version__)
    logger.info("tensorflow version: " + tf.__version__)
    logger.info("sklearn version: " + sklearn.__version__)
    logger.info("list of GPU: " + str(tf.config.list_physical_devices('GPU')))
    logger.info("Num GPUs Available: " + str(len(tf.config.list_physical_devices('GPU'))))
    logger.info(str(device_lib.list_local_devices()))

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            logger.info(str(len(gpus)) + "Physical GPUs, " + str(len(logical_gpus)) + " Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    # exit(-1)
    # ["ContagioPDF", "CIC-IDS-2017", "CICAndMal2017", "CSE-CIC-IDS2018", "NSL-KDD"]
    choose_dataset = "ContagioPDF"
    logger.info('---------------Dataset---------------')
    logger.info('current dataset: ' + str(choose_dataset))
    X_train_scaled, X_test_scaled, y_train, y_test = run(choose_dataset)
    baseline_model(X_train_scaled, X_test_scaled, y_train, y_test)
