import tensorflow as tf
import pandas as pd
import numpy as np
from PIL import Image

df1 = pd.read_csv('train.csv')
temp1 = np.array(df1)

X_train = temp1[:, 1:]
X_train = X_train/255
X_train = X_train.reshape(42000, 28, 28, 1)
y_train = temp1[:, 0]

for i in range(5):
    mat = np.array(X_train[i, :, :, 0], dtype='uint8')
    img = Image.fromarray(mat)
    img.show()
    print(y_train[i])

print(X_train.shape, y_train.shape)

def models():

    DESIRED_ACCURACY = 0.999

    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if logs.get('acc') > DESIRED_ACCURACY:
                print('\nReached 99.9% accuracy so cancelling training!')
                self.model.stop_training = True

    callbacks = myCallback()
    model = tf.keras.models.Sequential(
        [tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
         tf.keras.layers.MaxPooling2D(2, 2),
         tf.keras.layers.Flatten(),
         tf.keras.layers.Dense(100, activation='relu'),
         tf.keras.layers.Dense(10, activation='softmax')])

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

    # model fitting
    history = model.fit(X_train, y_train, epochs=4, verbose=1, callbacks=[callbacks])
    model.save('Digits.h5')
    return history.history['acc'][-1]

models()

