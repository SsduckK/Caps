import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import mnist
from keras.models import load_model
from emnist import extract_training_samples
from emnist import extract_test_samples
from tensorflow_datasets.image_classification import EMNIST
from keras.utils import np_utils
import sklearn.metrics as metrics
import matplotlib as mpl
import matplotlib.pyplot as plt
from functools import partial

DefaultConv2D = partial(keras.layers.Conv1D,
                        kernel_size=3, activation='relu', padding='SAME')

# (X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()
# X_train, X_valid = X_train_full[:-5000], X_train_full[-5000:]
# y_train, y_valid = y_train_full[:-5000], y_train_full[-5000:]


(X_train_full, y_train_full) = extract_training_samples('letters')
(X_test, y_test) = extract_test_samples('letters')
X_train_full = X_train_full / 255.
X_test = X_test / 255.

X_train, X_valid = X_train_full[:-20800], X_train_full[-20800:]
y_train, y_valid = y_train_full[:-20800], y_train_full[-20800:]


# X_mean = X_train.mean(axis=0, keepdims=True)
# X_std = X_train.std(axis=0, keepdims=True) + 1e-7
# X_train = (X_train - X_mean) / X_std
# X_valid = (X_valid - X_mean) / X_std
# X_test = (X_test - X_mean) / X_std
#
X_train = X_train[..., np.newaxis]
X_valid = X_valid[..., np.newaxis]
X_test = X_test[..., np.newaxis]

model = Sequential()

model.add(Conv2D(filters=64, kernel_size=(7,7), padding = 'same', activation='relu',
                 input_shape=(28, 28,1)))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Conv2D(filters=128, kernel_size=(3,3) , padding = 'same', activation='relu'))
model.add(Conv2D(filters=128, kernel_size=(3,3) , padding = 'same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=256, kernel_size=(3,3) , padding = 'same', activation='relu'))
model.add(Conv2D(filters=256, kernel_size=(3,3) , padding = 'same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(.5))
model.add(Dense(units=64, activation='relu'))
model.add(Dropout(.5))
model.add(Dense(units=27, activation='softmax'))

model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=20, batch_size=512, verbose=1,
                    validation_data=(X_valid, y_valid))
score = model.evaluate(X_test, y_test)
X_new = X_test[:10]
y_pred = model.predict(X_new)

print('acc:', score[1], 'loss:', score[0])
import random

predicted_result = model.predict(X_test)
predicted_labels = np.argmax(predicted_result, axis=1)

test_labels = y_test

wrong_result = []

for n in range(0, len(test_labels)):
    if predicted_labels[n] != test_labels[n]:
        wrong_result.append(n)

samples = random.choices(population=wrong_result, k=16)

count = 0
nrows = ncols = 4

plt.figure(figsize=(12,8))

for n in samples:
    count += 1
    plt.subplot(nrows, ncols, count)
    plt.imshow(X_test[n].reshape(28, 28), cmap='Greys', interpolation='nearest')
    tmp = "Label:" + str(test_labels[n]) + ", Prediction:" + str(predicted_labels[n])
    plt.title(tmp)

plt.tight_layout()

fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.plot(history.history['loss'], 'y', label='train loss')
loss_ax.plot(history.history['val_loss'], 'r', label='val loss')

acc_ax.plot(history.history['accuracy'], 'b', label='train acc')
acc_ax.plot(history.history['val_accuracy'], 'g', label='val acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuracy')
loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')
plt.show()
#
model.save('emnist_mlp_model.h5')
