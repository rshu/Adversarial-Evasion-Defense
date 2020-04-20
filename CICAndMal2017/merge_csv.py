import os, csv
import pandas as pd

dir_path = '../data/CICAndMal2017'

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
assert len(file_list) == 2126

for file in file_list:
    print(file)

# file_out = open("../data/CICAndMal2017/ConsolidateData.csv", 'a+')
file_out = "../data/CICAndMal2017/ConsolidateData.csv"

# header = "FlowID,SourceIP,SourcePort,DestinationIP,DestinationPort,Protocol,Timestamp,FlowDuration,TotalFwdPackets,TotalBackwardPackets,TotalLengthofFwdPackets,TotalLengthofBwdPackets,FwdPacketLengthMax,FwdPacketLengthMin,FwdPacketLengthMean,FwdPacketLengthStd,BwdPacketLengthMax,BwdPacketLengthMin,BwdPacketLengthMean,BwdPacketLengthStd,FlowBytes/s,FlowPackets/s,FlowIATMean,FlowIATStd,FlowIATMax,FlowIATMin,FwdIATTotal,FwdIATMean,FwdIATStd,FwdIATMax,FwdIATMin,BwdIATTotal,BwdIATMean,BwdIATStd,BwdIATMax,BwdIATMin,FwdPSHFlags,BwdPSHFlags,FwdURGFlags,BwdURGFlags,FwdHeaderLength,BwdHeaderLength,FwdPackets/s,BwdPackets/s,MinPacketLength,MaxPacketLength,PacketLengthMean,PacketLengthStd,PacketLengthVariance,FINFlagCount,SYNFlagCount,RSTFlagCount,PSHFlagCount,ACKFlagCount,URGFlagCount,CWEFlagCount,ECEFlagCount,Down/UpRatio,AveragePacketSize,AvgFwdSegmentSize,AvgBwdSegmentSize,FwdHeaderLength2,FwdAvgBytes/Bulk,FwdAvgPackets/Bulk,FwdAvgBulkRate,BwdAvgBytes/Bulk,BwdAvgPackets/Bulk,BwdAvgBulkRate,SubflowFwdPackets,SubflowFwdBytes,SubflowBwdPackets,SubflowBwdBytes,Init_Win_bytes_forward,Init_Win_bytes_backward,act_data_pkt_fwd,min_seg_size_forward,ActiveMean,ActiveStd,ActiveMax,ActiveMin,IdleMean,IdleStd,IdleMax,IdleMin,Label"
# print(header)
# file_out.write(header)

# for file in file_list:
#     print("")
#     print(file)
#     f = open(file)
#     f.__next__()  # skip the header
#     for line in f:
#         file_out.write(line)
#     f.close()
# file_out.close()

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
