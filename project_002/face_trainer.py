import time
import os

import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.python.client import device_lib


print("Start loading dataset...")
batch_size = 32
img_height = 180
img_width = 180
data_dir = os.getcwd()+"/faces"
validation_dir = os.getcwd()+"/validation"

data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal",
                      input_shape=(img_height,
                                  img_width,
                                  3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
val_ds = tf.keras.utils.image_dataset_from_directory(
  validation_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
print("Finished loading dataset.")

# print("Loading data...")
# normalization_layer = layers.Rescaling(1./255)
#
# normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
# image_batch, labels_batch = next(iter(normalized_ds))
# print("Finished loading data.")

num_classes = len(train_ds.class_names)
gpus = device_lib.list_local_devices()

input("Press any key to continue")

print()

tf.debugging.set_log_device_placement(True)
if len(gpus) > 1:
    strategy = tf.distribute.MirroredStrategy([gpu.name for gpu in gpus])
    print('Running on multiple GPUs ', [gpu.name for gpu in gpus])
elif len(gpus) == 2:
    strategy = tf.distribute.get_strategy()
    print('Running on single GPU', gpus[0].name)
    print('#accelerators: ', strategy.num_replicas_in_sync)
else:
    strategy = tf.distribute.get_strategy()
    print("Running on single CPU only")

print()

input("Press any key to build model")

print("Building model...")

with strategy.scope():
    model = Sequential([
        layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    print("Finished building model.")
    model.summary()

    print()
    input("Press any key to start training")
    print()

    startTime = time.time()

    print("Training started...")

    epochs = 1
    history = model.fit(
      train_ds,
      validation_data=val_ds,
      epochs=epochs
    )

    print("Finished! Time: ", time.time()-startTime)

trainList = os.listdir("trains")
trainList.sort(reverse=True)
i = 0
for e in trainList:
    if e.split(".") != "":
        if int(e.split(".")[0].split("_")[1]) > i:
            i = int(e.split(".")[0].split("_")[1])
model.save("trains/train_%d.h5" % (i+1))
print("Model saved.")

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


