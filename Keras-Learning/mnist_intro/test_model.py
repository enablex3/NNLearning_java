import keras, numpy
import matplotlib.pyplot as pyplot

# Load the model
model_file = "Trained_MNIST_Model.h5"
model = keras.models.load_model(model_file)

# Load the training & test sets
(train_X, train_y), (test_X, test_y) = keras.datasets.mnist.load_data()

"""# Model / Params
num_classes = 10
input_shape = (28, 28, 1)

# Scale images to the (0, 1) range
train_X = train_X.astype("float32") / 255
test_X = test_X.astype("float32") / 255

# Make sure the images have shape (input_shape)
train_X = numpy.expand_dims(train_X, -1)"""

# We want to test 100 images start at the 2000th image (the model is trained for 1920 images)
start_index = 2000
num_images = 10

selected_images = train_X[start_index: start_index + num_images]
image_labels = train_y[start_index: start_index + num_images]
print("Labels:", image_labels)
predictions = model.predict(selected_images)
for i in range(num_images):
    # Show the digit
    print("The model is guessing the following number...")
    pyplot.imshow(selected_images[i], cmap='gray')
    pyplot.title(f"Image {start_index + i}")
    pyplot.show(block=False)
    pyplot.pause(2)
    # they should be in the same order (label & prediction)
    # AS LONG AS WE'RE GETTING THE CORRECT LABELS!
    label = image_labels[i]
    prediction = predictions[i]
    # get the index of the value '1' from the predictions array
    predicted_index = numpy.where(prediction == 1)
    # if the predicted index is equal to the label, the model is correct
    print("Label:", label, "Prediction:", predicted_index)
    if (predicted_index == label):
        print("The model predicted correctly!")
        
"""delay = 2
for i in range(num_images):
    # Show the digiti
    print("The model is guessing the following number...")
    pyplot.imshow(selected_images[i], cmap='gray')
    pyplot.title(f"Image {start_index + i}")
    pyplot.show(block=False)
    pyplot.pause(delay)"""
