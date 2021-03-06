import pandas as pd
import numpy as np

labels = ['BENIGN', 'SCAREWARE_FAKEAV', 'SCAREWARE_FAKEAPP', 'SCAREWARE_FAKEAPPAL', 'SCAREWARE_ANDROIDDEFENDER',
          'SCAREWARE_VIRUSSHIELD', 'SCAREWARE_FAKEJOBOFFER', 'MALWARE', 'SCAREWARE_PENETHO', 'SCAREWARE_FAKETAOBAO',
          'SCAREWARE_AVPASS', 'SCAREWARE_ANDROIDSPY', 'SCAREWARE_AVFORANDROID', 'ADWARE_FEIWO', 'ADWARE_GOOLIGAN',
          'ADWARE_KEMOGE', 'ADWARE_EWIND', 'ADWARE_YOUMI', 'ADWARE_DOWGIN', 'ADWARE_SELFMITE', 'ADWARE_KOODOUS',
          'ADWARE_MOBIDASH', 'ADWARE_SHUANET', 'SMSMALWARE_FAKEMART', 'SMSMALWARE_ZSONE', 'SMSMALWARE_FAKEINST',
          'SMSMALWARE_MAZARBOT', 'SMSMALWARE_NANDROBOX', 'SMSMALWARE_JIFAKE', 'SMSMALWARE_SMSSNIFFER',
          'SMSMALWARE_BEANBOT', 'SCAREWARE', 'SMSMALWARE_FAKENOTIFY', 'SMSMALWARE_PLANKTON', 'SMSMALWARE_BIIGE',
          'RANSOMWARE_LOCKERPIN', 'RANSOMWARE_CHARGER', 'RANSOMWARE_PORNDROID', 'RANSOMWARE_PLETOR', 'RANSOMWARE_JISUT',
          'RANSOMWARE_WANNALOCKER', 'RANSOMWARE_KOLER', 'RANSOMWARE_RANSOMBO', 'RANSOMWARE_SIMPLOCKER',
          'RANSOMWARE_SVPENG']

label_map = {}

for label in labels:
    if label == 'BENIGN':
        label_map[label] = 0
    else:
        label_map[label] = 1

# print(label_map)

if __name__ == "__main__":
    x = pd.date_range('2017-01-01', '2017-12-31', freq='D').astype(str)
    time_map = dict(zip(x, np.arange(1, 366)))
    print(time_map)
