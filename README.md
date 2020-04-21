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