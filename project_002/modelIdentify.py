import os
import tensorflow as tf

trainList = os.listdir("trains")
i = 0
for e in trainList:
    if e.split(".")[0] != "":
        if int(e.split(".")[0].split("_")[1]) > i:
            i = int(e.split(".")[0].split("_")[1])

model = tf.keras.models.load_model('trains/train_%d.keras' % i)

model.summary()