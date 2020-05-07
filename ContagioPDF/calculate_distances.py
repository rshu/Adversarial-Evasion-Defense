import pandas as pd
import json, csv, sys
from find_two_models import dnn_model
from os import path
from sklearn.model_selection import train_test_split
from attack import load_dataset, file_path, fgsm_attack, bim_b_attack, bim_a_attack
from sklearn import preprocessing


def normalize(x, xmin, xmax):
    return (x - xmin) / (xmax - xmin)


def calculate_instance_distance(param1, param2):
    '''
    Return instance based distance
    '''
    distance = 0
    first_layer_dense_min = 30
    first_layer_dense_max = 150
    second_layer_dense_min = 30
    second_layer_dense_max = 50
    third_layer_dense_min = 10
    third_layer_dense_max = 32
    drop_out_min = 0.0
    drop_out_max = 0.5

    # add first_layer_dense distance
    distance += (normalize(param1['first_layer_dense'], first_layer_dense_min, first_layer_dense_max) - normalize(
        param2['first_layer_dense'], first_layer_dense_min, first_layer_dense_max)) ** 2

    # add second_layer_dense distance
    distance += (normalize(param1['second_layer_dense'], second_layer_dense_min, second_layer_dense_max) - normalize(
        param2['second_layer_dense'], second_layer_dense_min, second_layer_dense_max)) ** 2

    # add third_layer_dense distance
    distance += (normalize(param1['third_layer_dense'], third_layer_dense_min, third_layer_dense_max) - normalize(
        param2['third_layer_dense'], third_layer_dense_min, third_layer_dense_max)) ** 2

    # add drop_out distance
    distance += (normalize(param1['drop_out'], drop_out_min, drop_out_max) - normalize(
        param2['drop_out'], drop_out_min, drop_out_max)) ** 2

    # add hidden_layer_activation distance
    if (param1['hidden_layer_activation'] != param2['hidden_layer_activation']):
        distance += 1
    else:
        distance += 0

    # add output_layer_activation distance
    if (param1['output_layer_activation'] != param2['output_layer_activation']):
        distance += 1
    else:
        distance += 0

    # add optimizer distance
    if (param1['optimizer'] != param2['optimizer']):
        distance += 1
    else:
        distance += 0

    return distance


def calculate_instance_distance2(param1, param2):
    '''
    Return instance based distance
    '''
    distance = 0
    first_layer_dense_min = 30
    first_layer_dense_max = 150
    second_layer_dense_min = 30
    second_layer_dense_max = 50
    third_layer_dense_min = 10
    third_layer_dense_max = 32
    drop_out_min = 0.0
    drop_out_max = 0.5

    # add first_layer_dense distance
    distance += abs((normalize(param1['first_layer_dense'], first_layer_dense_min, first_layer_dense_max) - normalize(
        param2['first_layer_dense'], first_layer_dense_min, first_layer_dense_max)))

    # add second_layer_dense distance
    distance += abs(
        (normalize(param1['second_layer_dense'], second_layer_dense_min, second_layer_dense_max) - normalize(
            param2['second_layer_dense'], second_layer_dense_min, second_layer_dense_max)))

    # add third_layer_dense distance
    distance += abs((normalize(param1['third_layer_dense'], third_layer_dense_min, third_layer_dense_max) - normalize(
        param2['third_layer_dense'], third_layer_dense_min, third_layer_dense_max)))

    # add drop_out distance
    distance += abs((normalize(param1['drop_out'], drop_out_min, drop_out_max) - normalize(
        param2['drop_out'], drop_out_min, drop_out_max)))

    # add hidden_layer_activation distance
    if (param1['hidden_layer_activation'] != param2['hidden_layer_activation']):
        distance += 1
    else:
        distance += 0

    # add output_layer_activation distance
    if (param1['output_layer_activation'] != param2['output_layer_activation']):
        distance += 1
    else:
        distance += 0

    # add optimizer distance
    if (param1['optimizer'] != param2['optimizer']):
        distance += 1
    else:
        distance += 0

    return distance


if __name__ == "__main__":
    df = pd.read_csv('dl_trials_1.csv')
    # print(df.shape)

    # Sort with best scores on top and reset index for slicing
    df.sort_values('loss', ascending=True, inplace=True)
    df.reset_index(inplace=True, drop=True)
    # print(df.head())

    best_result = df.iloc[0]
    # print(best_result['params'])

    distance_result_file = 'distance_result.csv'
    distance_file = open(distance_result_file, 'w')
    writer = csv.writer(distance_file)

    writer.writerow(['loss', 'distance', 'FGSM', 'BIM-A', 'BIM-B', 'params'])

    if path.exists("saved_dataframe.pkl"):
        dataset = pd.read_pickle("./saved_dataframe.pkl")
    else:
        dataset = load_dataset(file_path)

    X = dataset.drop(['class'], axis=1)
    y = dataset['class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    y_train = y_train.reshape((-1, 1))
    y_test = y_test.reshape((-1, 1))

    # pre-processing
    scaler1 = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler1.transform(X_train)

    scaler2 = preprocessing.StandardScaler().fit(X_test)
    X_test = scaler2.transform(X_test)

    input_shape = X_train.shape[1]

    # best_model = dnn_model(eval(best_result['params']), input_shape)
    # best_model.fit(X_train, y_train, batch_size=32, epochs=1)

    # the first line of sorted file based on loss
    best_model = pd.read_pickle("./pickles/bayesOpt_nn_model_" + str(best_result['iteration']) + ".pkl")
    print("Base accuracy on original dataset:", best_model.evaluate(x=X_test, y=y_test, verbose=0))

    # define performance delta
    loss_epsilon = 0.1

    # make a copy of X_test and y_test



    print("")
    print("creating FGSM advasary data...")
    X_test_adv_fgsm, y_test_adv_fgsm = fgsm_attack(best_model, X_test, y_test, epsilon=0.2, clip_min=0.0, clip_max=1.0)

    # print(X_test_adv_fgsm)

    print("")
    print("creating BIM-A advasary data...")
    X_test_adv_bim_a, y_test_adv_bim_a = bim_a_attack(best_model, X_test, y_test, epsilon=0.2, clip_min=0.0,
                                                      clip_max=1.0, iterations=10)

    # print(X_test_adv_bim_a)

    print("")
    print("creating BIM-B advasary data...")
    X_test_adv_bim_b, y_test_adv_bim_b = bim_b_attack(best_model, X_test, y_test, epsilon=0.3, clip_min=0.0,
                                                      clip_max=1.0, iterations=10)

    # sys.exit(-1)
    print("")
    print("finding models with epsilon perf...")
    for i in range(df.shape[0]):
        if df.iloc[i]['loss'] - best_result['loss'] <= loss_epsilon:
            # print(df.iloc[i])
            # print(best_result['params'])
            # print(df.iloc[i]['params'])
            # print(calculate_instance_distance(eval(best_result['params']), eval(df.iloc[i]['params'])))

            # candidate_model = dnn_model(eval(df.iloc[i]['params']), input_shape)
            # candidate_model.fit(X_train, y_train, batch_size=32, epochs=1)

            candidate_model_index = df.iloc[i]['iteration']
            candidate_model_file = "./pickles/bayesOpt_nn_model_" + str(candidate_model_index) + ".pkl"
            candidate_model = pd.read_pickle(candidate_model_file)

            # print(candidate_model.evaluate(x=X_test_adv_fgsm, y=y_test_adv_fgsm, verbose=0))
            # print(candidate_model.evaluate(x=X_test_adv_bim_a, y=y_test_adv_bim_a, verbose=0))
            # print(candidate_model.evaluate(x=X_test_adv_bim_b, y=y_test_adv_bim_b, verbose=0))

            fgsm_result = candidate_model.evaluate(x=X_test_adv_fgsm, y=y_test_adv_fgsm, verbose=0)[1]
            bim_a_result = candidate_model.evaluate(x=X_test_adv_bim_a, y=y_test_adv_bim_a, verbose=0)[1]
            bim_b_result = candidate_model.evaluate(x=X_test_adv_bim_b, y=y_test_adv_bim_b, verbose=0)[1]

            writer.writerow([df.iloc[i]['loss'],
                             calculate_instance_distance2(eval(best_result['params']), eval(df.iloc[i]['params'])),
                             fgsm_result, bim_a_result,
                             bim_b_result, df.iloc[i]['params']])

    distance_file.close()
