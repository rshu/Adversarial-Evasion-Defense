## Advesarial Lernaing Attack and Mitigation

####Notes

change to utils_tf.py 
```
# import tensorflow as tf
import tensorflow.compat.v1 as tf
```


out of memory or cannot allocate large memory issue

```
cat /proc/sys/vm/overcommit_memory
sudo su
echo 1 > /proc/sys/vm/overcommit_memory
exit
```

To download CSC-IDS-2018, first install aws-cli tool
```
sudo apt-get install awscli
```

browse the folder
```
aws s3 ls --no-sign-request "s3://cse-cic-ids2018" --recursive --human-readable --summarize
```

download a single file
```
aws s3 cp --no-sign-request "s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv" "Destination path"
```

download all csv files
```
aws s3 cp --no-sign-request --region eu-west-3 "s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/" <dest-dir> --recursive
```