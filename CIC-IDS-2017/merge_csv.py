import os, csv
import pandas as pd

dir_path = '../data/CIC-IDS-2017'

file_count = 0
file_list = []
for root, dirs, files in os.walk(dir_path, topdown=True):
    for file in files:
        file_path = os.path.join(root, file)
        file_list.append(file_path)
        # print(file_path)

        # check they have the same header
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            row1 = next(reader)
            # list.append(row1)
            print("file_count: " + str(file_count))
            print(row1)
        f.close()
        file_count += 1

print(file_count)
#

file_out = "../data/CIC-IDS-2017/ConsolidateData.csv"


def produceOneCSV(list_of_files, file_out):
    """
     Function:
      Produce a single CSV after combining all files
    """
    # Consolidate all CSV files into one object
    result_obj = pd.concat([pd.read_csv(file) for file in list_of_files])
    # Convert the above object into a csv file and export
    result_obj.to_csv(file_out, index=False, encoding="utf-8")


produceOneCSV(file_list, file_out)
