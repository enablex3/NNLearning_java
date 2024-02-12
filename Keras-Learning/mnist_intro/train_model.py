import keras, numpy

# Get the training set and the test set (default is 60k, 10k)
(train_X, train_y), (test_X, test_y) = keras.datasets.mnist.load_data()

print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  '  + str(test_X.shape))
print('Y_test:  '  + str(test_y.shape))

# Model / Params
num_classes = 10
input_shape = (28, 28, 1)

# Scale images to the (0, 1) range
train_X = train_X.astype("float32") / 255
test_X = test_X.astype("float32") / 255

# Make sure the images have shape (input_shape)
train_X = numpy.expand_dims(train_X, -1)
test_X = numpy.expand_dims(test_X, -1)

# Convert classes 
train_y = keras.utils.to_categorical(train_y, num_classes)
test_y = keras.utils.to_categorical(test_y, num_classes)

# Build the model
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_classes, activation="softmax")
    ]
)

model.summary()

batch_size = 128
epochs = 15

# Train the model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(train_X, train_y, batch_size=batch_size, epochs=epochs, validation_split=0.1)

# Evaluate
score = model.evaluate(test_X, test_y)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# Save this model
model_file = "Trained_MNIST_Model.h5"
model.save(model_file)