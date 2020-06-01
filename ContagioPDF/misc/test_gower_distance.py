import numpy as np
import pandas as pd
import gower

desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 50)

# Xd = pd.DataFrame({'age': [21, 21, 19, 30, 21, 21, 19, 30],
#                    'gender': ['M', 'M', 'M', 'M', 'F', 'F', 'F', 'F'],
#                    'civil_status': ['MARRIED', 'SINGLE', 'SINGLE', 'SINGLE', 'MARRIED', 'SINGLE', 'WIDOW', 'DIVORCED'],
#                    'salary': [3000.0, 1200.0, 32000.0, 1800.0, 2900.0, 1100.0, 10000.0, 1500.0],
#                    'has_children': [1, 0, 1, 1, 1, 0, 0, 1],
#                    'available_credit': [2200, 100, 22000, 1100, 2000, 100, 6000, 2200]})
# Yd = Xd.iloc[0:1, :]
#
# print(gower.gower_matrix(Xd, Yd, cat_features=[False, True, True, False, False, False]))

df = pd.read_csv('../../CIC-IDS-2017/dl_trials_1.csv')
# print(df.shape)

# Sort with best scores on top and reset index for slicing
df.sort_values('loss', ascending=True, inplace=True)
df.reset_index(inplace=True, drop=True)

print(df)
print(df.shape)

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
    # print(tmp_list)
    data.append(tmp_list)

print(data)

params_dataframe = pd.DataFrame(data, columns=['drop_out', 'first_layer_dense', 'hidden_layer_activation', 'optimizer',
                                               'output_layer_activation', 'second_layer_dense', 'third_layer_dense',
                                               'batch_size', 'num_epochs'])

print(params_dataframe)

loss_epsilon = 0.1
best_result = df.iloc[0]

for i in range(df.shape[0]):
    if df.iloc[i]['loss'] - best_result['loss'] <= loss_epsilon:
        # print(i)
        candidate_params = params_dataframe.iloc[i:i + 1, :]
        print(gower.gower_matrix(params_dataframe, candidate_params,
                                 cat_features=[False, False, True, True, True, False, False,
                                               True, False])[0][0])

# Yd = params_dataframe.iloc[1:2, :]
# print(gower.gower_matrix(params_dataframe, Yd, cat_features=[False, False, True, True, True, False, False]))
