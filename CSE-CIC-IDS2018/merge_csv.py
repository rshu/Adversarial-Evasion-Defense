import os, csv
import pandas as pd
import sys, gc

dir_path = '../data/CSE-CIC-IDS2018'

file_count = 0
file_list = []
for root, dirs, files in os.walk(dir_path, topdown=True):
    for file in files:
        file_path = os.path.join(root, file)
        file_list.append(file_path)
        print(file_path)

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
print("")

# for file in file_list:
#     print(file)
#     df = pd.read_csv(file, encoding='utf8', low_memory=False)
#     print(df.shape)
#     del df
#     gc.collect()
#
# sys.exit(-1)


# # remove extra 4 columns
# df = pd.read_csv("../data/CSE-CIC-IDS2018/Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv", encoding='utf8',
#                  low_memory=False)
# print(df.shape)
#
# df.drop('Flow ID', axis=1, inplace=True)
# df.drop('Src IP', axis=1, inplace=True)
# df.drop('Src Port', axis=1, inplace=True)
# df.drop('Dst IP', axis=1, inplace=True)
#
# print(list(df.columns))
# print(df.shape)  # (7948748, 80)
#
# df.to_csv("../data/CSE-CIC-IDS2018/reshaped_Tuesday-20-02-2018.csv", index=False)
#
# sys.exit(-1)

file_out = open("../data/CSE-CIC-IDS2018/ConsolidateData.csv", 'a+')
# file_out = "../data/CSE-CIC-IDS2018/ConsolidateData.csv"

header = "Dst Port,Protocol,Timestamp,Flow Duration,Tot Fwd Pkts,Tot Bwd Pkts,TotLen Fwd Pkts,TotLen Bwd Pkts,Fwd Pkt Len Max,Fwd Pkt Len Min,Fwd Pkt Len Mean,Fwd Pkt Len Std,Bwd Pkt Len Max,Bwd Pkt Len Min,Bwd Pkt Len Mean,Bwd Pkt Len Std,Flow Byts/s,Flow Pkts/s,Flow IAT Mean,Flow IAT Std,Flow IAT Max,Flow IAT Min,Fwd IAT Tot,Fwd IAT Mean,Fwd IAT Std,Fwd IAT Max,Fwd IAT Min,Bwd IAT Tot,Bwd IAT Mean,Bwd IAT Std,Bwd IAT Max,Bwd IAT Min,Fwd PSH Flags,Bwd PSH Flags,Fwd URG Flags,Bwd URG Flags,Fwd Header Len,Bwd Header Len,Fwd Pkts/s,Bwd Pkts/s,Pkt Len Min,Pkt Len Max,Pkt Len Mean,Pkt Len Std,Pkt Len Var,FIN Flag Cnt,SYN Flag Cnt,RST Flag Cnt,PSH Flag Cnt,ACK Flag Cnt,URG Flag Cnt,CWE Flag Count,ECE Flag Cnt,Down/Up Ratio,Pkt Size Avg,Fwd Seg Size Avg,Bwd Seg Size Avg,Fwd Byts/b Avg,Fwd Pkts/b Avg,Fwd Blk Rate Avg,Bwd Byts/b Avg,Bwd Pkts/b Avg,Bwd Blk Rate Avg,Subflow Fwd Pkts,Subflow Fwd Byts,Subflow Bwd Pkts,Subflow Bwd Byts,Init Fwd Win Byts,Init Bwd Win Byts,Fwd Act Data Pkts,Fwd Seg Size Min,Active Mean,Active Std,Active Max,Active Min,Idle Mean,Idle Std,Idle Max,Idle Min,Label\n"
print(header)
file_out.write(header)

for file in file_list:
    print("")
    print(file)
    f = open(file)
    f.__next__()  # skip the header
    for line in f:
        file_out.write(line)
    f.close()
file_out.close()

# def produceOneCSV(list_of_files, file_out):
#     """
#      Function:
#       Produce a single CSV after combining all files
#     """
#     # Consolidate all CSV files into one object
#     result_obj = pd.concat([pd.read_csv(file) for file in list_of_files])
#     # Convert the above object into a csv file and export
#     result_obj.to_csv(file_out, index=False, encoding="utf-8")
#
#
# produceOneCSV(file_list, file_out)
