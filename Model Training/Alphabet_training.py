import tensorflow as tf
import pandas as pd
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

df1 = pd.read_parquet('emnist-balanced-train.parquet')
df2 = pd.read_parquet('emnist-balanced-test.parquet')

temp1 = np.array(df1)
temp2 = np.array(df2)

# Training data
X_train = temp1[:, 1:]
for i in range(len(X_train)):
    ex = X_train[i, :].reshape((28, 28), order='F')
    ex = ex.reshape((1, 784))
    X_train[i, :] = ex
X_train = X_train.reshape(88800, 28, 28, 1)
y_train = temp1[:, 0]

# Plotting first five images from training data
for i in range(5):
    mat = np.array(X_train[i, :, :, 0], dtype='uint8')
    img = Image.fromarray(mat)
    img.show()
    print(y_train[i])

# Testing data
X_test = temp2[:, 1:]
for i in range(len(X_test)):
    ex = X_test[i, :].reshape((28, 28), order='F')
    ex = ex.reshape((1, 784))
    X_test[i, :] = ex
X_test = X_test.reshape(14800, 28, 28, 1)
y_test = temp2[:, 0]

# Plotting first five images from testing data
for i in range(5):
    mat = np.array(X_test[i, :, :, 0], dtype='uint8')
    img = Image.fromarray(mat)
    img.show()
    print(y_test[i])

train_datagen = ImageDataGenerator(rescale=1/255, rotation_range=20, height_shift_range=0.1,
                                   width_shift_range=0.1,
                                   zoom_range=0.1)
validation_datagen = ImageDataGenerator(rescale=1/255)

train_gen = train_datagen.flow(X_train, y_train, batch_size=100)
val_gen = validation_datagen.flow(X_test, y_test, batch_size=100)

DESIRED_ACCURACY = 0.97

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('acc') > DESIRED_ACCURACY:
            print('\nReached 97.0% accuracy so cancelling training!')
            self.model.stop_training = True

callbacks = myCallback()

model = tf.keras.models.Sequential(
    [tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
     tf.keras.layers.MaxPooling2D(2, 2),
     tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
     tf.keras.layers.MaxPooling2D(2, 2),
     tf.keras.layers.Flatten(),
     tf.keras.layers.Dense(142, activation='relu'),
     tf.keras.layers.Dropout(0.1),
     tf.keras.layers.Dense(37, activation='softmax')])
model.summary()

from tensorflow.keras.optimizers import RMSprop
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

# model fitting
history = model.fit(train_gen, validation_data=val_gen, epochs=20, verbose=1, callbacks=[callbacks])
model.save('Version.h5')

