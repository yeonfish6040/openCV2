import time
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("Start loading dataset... \n")
batch_size = 16
img_height = 260
img_width = 260
data_dir = os.getcwd()+"/faces"
validation_dir = os.getcwd()+"/validation"

data_augmentation = keras.Sequential(
  [
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
  batch_size=batch_size, color_mode='grayscale')
val_ds = tf.keras.utils.image_dataset_from_directory(
  validation_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size, color_mode='grayscale')
# val_ds = val_ds.map(
#   lambda x, y: (data_augmentation(x, training=True), y)
# )


print("classes: ", train_ds.class_names)

print("\nFinished loading dataset.\n")

# print("Loading data...")
# normalization_layer = layers.Rescaling(1./255)
#
# normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
# image_batch, labels_batch = next(iter(normalized_ds))
# print("Finished loading data.")

num_classes = len(train_ds.class_names)
gpus = tf.config.list_physical_devices('GPU')
# gpus = device_lib.list_local_devices()

input("Press any key to continue")

print()

device = ""
if len(gpus) > 1:
    strategy = tf.distribute.MirroredStrategy([gpu.name for gpu in gpus])
    device = strategy.scope()
    print('Running on multiple GPUs ', [gpu.name for gpu in gpus])
elif len(gpus) == 1:
    device = tf.device("/GPU:0")
    print('Running on single GPU', gpus[0].name)
else:
    device = tf.distribute.get_strategy().scope()
    print("Running on single CPU only")

print()

input("Press any key to build model")

print("Building model...")

with device:

    model = Sequential([
        # layers.Dropout(0.5),
        # layers.Rescaling(1. / 255, input_shape=(260, 260, 3)),
        # layers.Conv2D(16, (3, 3), activation='relu'),
        # layers.MaxPooling2D(2, 2),
        # layers.Conv2D(32, (3, 3), activation='relu'),
        # layers.MaxPooling2D(2, 2),
        # layers.Conv2D(64, (3, 3), activation='relu'),
        # layers.MaxPooling2D(2, 2),
        # layers.Flatten(),
        # layers.Dense(128, activation='relu'),
        # layers.Dense(6, activation='softmax')
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.55),
        layers.Dense(256, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    print("Finished building model.")
    model.summary()

    print()
    is_continue = input("Press any key to start training (if you press y and num, it will continue training with data augmentation)")
    print()

    startTime = time.time()


    def train(dataset, dataset_val, epochs_val):
        return model.fit(
          dataset,
          validation_data=dataset_val,
          epochs=epochs_val,
        )

    acc_arr = []
    val_acc_arr = []
    loss_arr = []
    val_loss_arr = []
    def showAcc(history, epochs):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        acc_arr.extend(acc)
        val_acc_arr.extend(val_acc)
        loss_arr.extend(loss)
        val_loss_arr.extend(val_loss)

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(range(acc_arr.__len__()), acc_arr, label='Training Accuracy')
        plt.plot(range(val_acc_arr.__len__()), val_acc_arr, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(range(loss_arr.__len__()), loss_arr, label='Training Loss')
        plt.plot(range(val_loss_arr.__len__()), val_loss_arr, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()

    def setDataAugmentation():
        train_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size, color_mode='grayscale')
        return train_ds.map(
            lambda x, y: (data_augmentation(x, training=True), y)
        )

    try:
        epochs = int(input("epochs: "))
        i = 0
        while True:
            print("Training started...")


            history = train(train_ds, val_ds, epochs)

            print("Finished! Time: ", time.time() - startTime)

            showAcc(history, epochs)

            if is_continue.startswith("y"):
                if int(is_continue.replace("y", "").strip()) <= i:
                    is_continue = "n"
                else:
                    i += 1
                    train_ds = setDataAugmentation()
                    continue

            is_go = input("Do you want to continue training with data augmentation? (y/n): ")
            if is_go == "y":
                i += 1
                train_ds = setDataAugmentation()
                continue
            break

    except KeyboardInterrupt:
        pass


trainList = os.listdir("trains")
i = 0
for e in trainList:
    if e.split(".")[0] != "":
        if int(e.split(".")[0].split("_")[1]) > i:
            i = int(e.split(".")[0].split("_")[1])
model.save("trains/train_%d.keras" % (i+1))
print("Model saved.")



