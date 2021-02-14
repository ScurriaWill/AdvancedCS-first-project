import ssl
from keras import layers, callbacks
from keras.datasets import mnist
from keras.models import Sequential
from keras.utils.np_utils import to_categorical

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Reshaping data-Adding number of channels as 1 (Grayscale images)
train_images = train_images.reshape((train_images.shape[0], train_images.shape[1], train_images.shape[2], 1))
test_images = test_images.reshape((test_images.shape[0], test_images.shape[1], test_images.shape[2], 1))

# Scaling down pixel values
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

# Encoding labels to a binary class matrix
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

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


model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=['accuracy'])

earlyStopping = callbacks.EarlyStopping(monitor="val_loss", mode="auto", patience=3, restore_best_weights=True)

hist = model.fit(train_images, train_labels, batch_size=None,
                 epochs=epochs, validation_data=(test_images, test_labels), callbacks=[earlyStopping])
print("The model has successfully trained")

score = model.evaluate(test_images, test_labels, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print("CNN Error: %.2f%%" % (100 - score[1]*100))

model.save('saved_mnist_model')
print("Saving the model as saved_mnist_model")
