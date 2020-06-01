import pandas as pd
import csv
from os import path
from sklearn.model_selection import train_test_split
from attack_backup_2 import load_dataset, file_path, fgsm_attack, bim_b_attack, bim_a_attack, deepfool_attack
from sklearn import preprocessing
import gower


def normalize(x, xmin, xmax):
    return (x - xmin) / (xmax - xmin)


def instance_distance(param1, param2):
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
    num_epochs_min = 5
    num_epochs_max = 20

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
    if param1['hidden_layer_activation'] != param2['hidden_layer_activation']:
        distance += 1
    else:
        distance += 0

    # add output_layer_activation distance
    if param1['output_layer_activation'] != param2['output_layer_activation']:
        distance += 1
    else:
        distance += 0

    # add optimizer distance
    if param1['optimizer'] != param2['optimizer']:
        distance += 1
    else:
        distance += 0

    if param1['batch_size'] != param2['batch_size']:
        distance += 1
    else:
        distance += 0

    # add num_epochs distance
    distance += (normalize(param1['num_epochs'], num_epochs_min, num_epochs_max) - normalize(
        param2['num_epochs'], num_epochs_min, num_epochs_max)) ** 2

    if param1['learning_rate'] != param2['learning_rate']:
        distance += 1
    else:
        distance += 0

    return distance


def instance_distance2(param1, param2):
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
    df = pd.read_csv('dl_trials_sample.csv')
    # print(df.shape)

    # Sort with best scores on top and reset index for slicing
    df.sort_values('loss', ascending=True, inplace=True)
    df.reset_index(inplace=True, drop=True)
    # print(df.head())

    best_result = df.iloc[0]
    # print(best_result['params'])

    distance_result_file = 'distance_result_sample.csv'
    distance_file = open(distance_result_file, 'w')
    writer = csv.writer(distance_file)

    writer.writerow(['loss', 'distance', 'FGSM', 'BIM-A', 'BIM-B', 'deepfool', 'params'])

    if path.exists("../saved_dataframe_sample.pkl"):
        dataset = pd.read_pickle("../saved_dataframe_sample.pkl")
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
    best_model = pd.read_pickle("./sample_pickles/bayesOpt_nn_model_" + str(best_result['iteration']) + ".pkl")
    print("Base accuracy on original dataset:", best_model.evaluate(x=X_test, y=y_test, verbose=0))

    # define performance delta
    loss_epsilon = 0.10

    # make a copy of X_test and y_test
    X_test_fgsm = X_test.copy()
    y_test_fgsm = y_test.copy()

    X_test_bim_a = X_test.copy()
    y_test_bim_a = y_test.copy()

    X_test_bim_b = X_test.copy()
    y_test_bim_b = y_test.copy()

    X_test_deepfool = X_test.copy()
    y_test_deepfool = y_test.copy()

    print("")
    print("creating FGSM advasary data...")
    X_test_adv_fgsm, y_test_adv_fgsm = fgsm_attack(best_model, X_test_fgsm, y_test_fgsm, epsilon=0.3, clip_min=0.0,
                                                   clip_max=1.0)

    print("")
    print("creating BIM-A advasary data...")
    X_test_adv_bim_a, y_test_adv_bim_a = bim_a_attack(best_model, X_test_bim_a, y_test_bim_a, epsilon=0.3,
                                                      clip_min=0.0,
                                                      clip_max=1.0, iterations=15)

    print("")
    print("creating BIM-B advasary data...")
    X_test_adv_bim_b, y_test_adv_bim_b = bim_b_attack(best_model, X_test_bim_b, y_test_bim_b, epsilon=0.3,
                                                      clip_min=0.0,
                                                      clip_max=1.0, iterations=15)

    print("")
    print("creating deepfool advasary data...")
    X_test_adv_deepfool = deepfool_attack(best_model, X_test_deepfool)

    # creating gower distance dataframe
    data = []

    for i in range(df.shape[0]):
        params_dict = eval(df.iloc[i]['params'])
        # print(params_dict)
        tmp_list = []
        tmp_list.append(params_dict['drop_out'])
        tmp_list.append(params_dict['first_layer_dense'])
        tmp_list.append(params_dict['hidden_layer_activation'])
        tmp_list.append(params_dict['optimizer'])
        tmp_list.append(params_dict['output_layer_activation'])
        tmp_list.append(params_dict['second_layer_dense'])
        tmp_list.append(params_dict['third_layer_dense'])
        tmp_list.append(params_dict['batch_size'])
        tmp_list.append(params_dict['num_epochs'])
        tmp_list.append(params_dict['learning_rate'])
        # print(tmp_list)
        data.append(tmp_list)

    params_dataframe = pd.DataFrame(data,
                                    columns=['drop_out', 'first_layer_dense', 'hidden_layer_activation', 'optimizer',
                                             'output_layer_activation', 'second_layer_dense', 'third_layer_dense',
                                             'batch_size', 'num_epochs', 'learning_rate'])

    # sys.exit(-1)
    print("")
    print("finding models within epsilon perf...")
    for i in range(df.shape[0]):
        if df.iloc[i]['loss'] - best_result['loss'] <= loss_epsilon:
            # print(df.iloc[i])
            # print(best_result['params'])
            # print(df.iloc[i]['params'])
            # print(calculate_instance_distance(eval(best_result['params']), eval(df.iloc[i]['params'])))

            # candidate_model = dnn_model(eval(df.iloc[i]['params']), input_shape)
            # candidate_model.fit(X_train, y_train, batch_size=32, epochs=1)

            candidate_model_index = df.iloc[i]['iteration']
            candidate_model_file = "./sample_pickles/bayesOpt_nn_model_" + str(candidate_model_index) + ".pkl"
            candidate_model = pd.read_pickle(candidate_model_file)

            # print(candidate_model.evaluate(x=X_test_adv_fgsm, y=y_test_adv_fgsm, verbose=0))
            # print(candidate_model.evaluate(x=X_test_adv_bim_a, y=y_test_adv_bim_a, verbose=0))
            # print(candidate_model.evaluate(x=X_test_adv_bim_b, y=y_test_adv_bim_b, verbose=0))

            fgsm_result = candidate_model.evaluate(x=X_test_adv_fgsm, y=y_test_adv_fgsm, verbose=0)[1]
            bim_a_result = candidate_model.evaluate(x=X_test_adv_bim_a, y=y_test_adv_bim_a, verbose=0)[1]
            bim_b_result = candidate_model.evaluate(x=X_test_adv_bim_b, y=y_test_adv_bim_b, verbose=0)[1]
            deepfool_result = candidate_model.evaluate(x=X_test_adv_deepfool, y=y_test_deepfool, verbose=0)[1]

            candidate_params = params_dataframe.iloc[i:i + 1, :]

            # fetch the first result from output matrix
            # True means categorical, False means numerical
            params_gower_distance = gower.gower_matrix(params_dataframe, candidate_params,
                                                       cat_features=[False, False, True, True, True, False, False,
                                                                     True, False, True])[0][0]

            params_instance_distance = instance_distance(eval(best_result['params']), eval(df.iloc[i]['params']))

            writer.writerow([df.iloc[i]['loss'],
                             params_gower_distance,
                             fgsm_result, bim_a_result,
                             bim_b_result, deepfool_result, df.iloc[i]['params']])

            # writer.writerow([df.iloc[i]['loss'],
            #                  calculate_instance_distance2(eval(best_result['params']), eval(df.iloc[i]['params'])),
            #                  fgsm_result, bim_a_result,
            #                  bim_b_result, df.iloc[i]['params']])

    distance_file.close()
