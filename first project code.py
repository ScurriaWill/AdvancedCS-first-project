import keras
import numpy as np
from keras import layers, callbacks
from keras.datasets import mnist
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
# from mnist import MNIST

# requests.packages.urllib3.disable_warnings()
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

"""mn = MNIST('./mnist_data')
train_images, train_labels = mn.load_training()
test_images, test_labels = mn.load_testing()"""

train_images = np.asarray(train_images).astype(np.float32)
train_labels = np.asarray(train_labels).astype(np.int32)
test_images = np.asarray(test_images).astype(np.float32)
test_labels = np.asarray(test_labels).astype(np.int32)

# the data, split between train and Digit-Recognizer sets
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

print(train_images.shape, train_labels.shape)

train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))
input_shape = (28, 28, 1)

# convert class vectors to binary class matrices
train_labels = keras.utils.to_categorical(train_labels, 10)
test_labels = keras.utils.to_categorical(test_labels, 10)

train_images = train_images.astype('float32')
test_images = test_images.astype('float32')
train_images /= 255
test_images /= 255

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
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

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

val_images = train_images[:10000]
partial_images = train_images[10000:]
val_labels = train_labels[:10000]
partial_labels = train_labels[10000:]

earlyStopping = callbacks.EarlyStopping(monitor="val_loss", mode="auto", patience=5, restore_best_weights=True)

hist = model.fit(partial_images, partial_labels, batch_size=None,
                 epochs=25, validation_data=(val_images, val_labels),
                 callbacks=[earlyStopping])
print("The model has successfully trained")

score = model.evaluate(test_images, test_labels, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print("CNN Error: %.2f%%" % (100 - score[1]*100))

model.save('saved_mnist_model')
print("Saving the model as saved_mnist_model")
