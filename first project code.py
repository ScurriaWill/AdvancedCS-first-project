import keras
import numpy as np
from keras import layers, callbacks
from keras.datasets import mnist
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from mnist import MNIST

mn = MNIST('./mnist_data')
train_images, train_labels = mn.load_training()
test_images, test_labels = mn.load_testing()

"""x_train = np.asarray(x_train).astype(np.float32)
y_train = np.asarray(y_train).astype(np.int32)
x_test = np.asarray(x_test).astype(np.float32)
y_test = np.asarray(y_test).astype(np.int32)

# the data, split between train and Digit-Recognizer sets
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)

x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
input_shape = (28, 28, 1)

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255"""

# Reshaping data-Adding number of channels as 1 (Grayscale images)
train_images = train_images.reshape((train_images.shape[0],
                                     train_images.shape[1],
                                     train_images.shape[2], 1))

test_images = test_images.reshape((test_images.shape[0],
                                   test_images.shape[1],
                                   test_images.shape[2], 1))

# Scaling down pixel values
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

# Encoding labels to a binary class matrix
y_train = to_categorical(train_labels)
y_test = to_categorical(test_labels)

print('train_images shape:', train_images.shape)
print(train_images.shape[0], 'train samples')
print(test_images.shape[0], 'Digit-Recognizer samples')

batch_size = 128
num_classes = 10
epochs = 30

model = Sequential()

model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(10, activation="softmax"))

"""
model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
"""
model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=['accuracy'])

earlyStopping = callbacks.EarlyStopping(monitor="val_loss", mode="auto", patience=5, restore_best_weights=True)

hist = model.fit(train_images, y_train, batch_size=None, epochs=epochs, verbose=1,
                 validation_data=(test_images, y_test), callbacks=[earlyStopping])
print("The model has successfully trained")

score = model.evaluate(test_images, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print("CNN Error: %.2f%%" % (100 - score[1]*100))

model.save('saved_mnist_model')
print("Saving the model as saved_mnist_model")
