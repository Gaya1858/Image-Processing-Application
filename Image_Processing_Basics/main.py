'''
These lines import necessary libraries for the code to run.
tensorflow is the main library used for creating and training deep learning models,
numpy is used for numerical operations, matplotlib is used for visualizing the data,
and cv2 is used for image processing.
'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2 # openCV i am using this in this code

'''
This line loads the MNIST dataset using tf.keras.datasets.mnist.load_data(). 
The dataset is split into two sets - a training set of 60,000 images and a test set of 10,000 images. 
The load_data() function returns four NumPy arrays: X_train, y_train, X_test, and y_test. 
X_train and X_test contain the images, while y_train and y_test contain the corresponding labels.
'''
# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

'''
These lines reshape the input images from a 3D shape (num_samples, width, height) 
to a 4D shape (num_samples, width, height, channels) where channels is set to 1. 
This is because the input to the convolutional layer in the model expects a 4D tensor with a batch size 
as the first dimension. The values are also scaled to a range of 0 to 1 by dividing by 255.0.
'''
# Preprocess the data
X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
X_test = X_test.reshape(-1, 28, 28, 1) / 255.0

'''
These lines one-hot encode the labels using tf.keras.utils.to_categorical(). 
This is because the output layer of the model has 10 units, one for each possible digit from 0 to 9, 
and expects the output to be a probability distribution over these 10 units. 
One-hot encoding is a way to represent the labels as a vector of length 10, 
where each element in the vector is either 0 or 1, indicating whether or not the label corresponds to that digit.
'''
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

'''
This code defines the architecture of the convolutional neural network (CNN) model using tf.keras.Sequential(). 
The model has the following layers:

A Conv2D layer with 32 filters of size 3x3, using the ReLU activation function. The input_shape is set to (28, 28, 1) 
    because the input images are 28x28 grayscale images with a single channel.
A MaxPooling2D layer with a pool size of 2x2, which reduces the spatial dimensions of the feature maps output 
    by the previous layer.
A Flatten layer, which converts the 2D feature maps to a 1D vector, ready to be input to a fully connected layer.
A Dense layer with 128 units and the ReLU activation function. This is a fully connected layer that takes 
    the flattened feature vector as input and applies a linear transformation to it, followed by 
    the ReLU activation function.
A Dense layer with 10 units and the softmax activation function. This is the output layer of the model that 
    produces a probability distribution over the 10 possible digit classes.
    
The number of units in the output layer is set to 10 to match the number of possible digit classes, 
and the softmax activation function is used to ensure that the output values sum up to 1, 
forming a probability distribution.

'''

# Define the model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

'''
This line compiles the model using the Adam optimizer and the categorical cross-entropy loss function. 
The Adam optimizer is a variant of stochastic gradient descent that adapts the learning rate for each parameter 
based on the gradient and the past history of gradients. The categorical cross-entropy loss function is commonly 
used for multiclass classification problems like this one, where the goal is to minimize the difference between 
the predicted probability distribution and the true distribution of the labels.
'''
# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

'''
This line trains the model using the fit() method. The X_train and y_train arrays are used as input to 
the model during training. The batch_size parameter specifies the number of samples to use in each training batch. 
The epochs parameter specifies the number of times to iterate over the entire training set. 
The validation_data parameter is used to specify the validation set to use during training. 
The model will evaluate its performance on this set after each epoch.
'''
# Train the model
model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)

'''
Show image using matplotlib
'''
# # Plot a random image from the training dataset
# rand_idx = np.random.randint(len(X_train))
# image = X_train[rand_idx].reshape((28, 28))  # reshape the image to its original dimensions
#
#
# # Crop an image to remove a 10-pixel border on all sides
# border_size = 3
# cropped_image = image[border_size:-border_size, border_size:-border_size]
# plt.imshow(cropped_image, cmap='gray')
# plt.title(f'Label: {y_train[rand_idx]}')
# plt.show()

# Load the image as a grayscale image
image = cv2.imread('new.jpg', cv2.IMREAD_GRAYSCALE)
#resized_image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_CUBIC)
resized_image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_LANCZOS4)/255.0
# # Resize the image to 28x28 pixels
# resized_image = cv2.resize(image, (28, 28)) default is linear
#
# # Display the resized image
# # cv2.imshow('Resized Image', resized_image)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
# plt.imshow(resized_image, cmap='gray')
# #plt.title(f'Label: {y_train[rand_idx]}')
# plt.show()

image = np.expand_dims(resized_image, axis=-1)
image = np.expand_dims(resized_image, axis=0)

# Make a prediction
prediction = model.predict(image)
digit = np.argmax(prediction)
print('The predicted digit is:', digit)

image = cv2.imread('new.jpg', cv2.IMREAD_GRAYSCALE)
#resized_image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_CUBIC)
resized_image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_LANCZOS4)/255.0
# # Resize the image to 28x28 pixels
# resized_image = cv2.resize(image, (28, 28)) default is linear
#
# # Display the resized image
# # cv2.imshow('Resized Image', resized_image)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
# plt.imshow(resized_image, cmap='gray')
# #plt.title(f'Label: {y_train[rand_idx]}')
# plt.show()
image = cv2.imread('new1.jpg', cv2.IMREAD_GRAYSCALE)
#resized_image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_CUBIC)
resized_image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_LANCZOS4)/255.0
image = np.expand_dims(resized_image, axis=-1)
image = np.expand_dims(resized_image, axis=0)

# Make a prediction
prediction = model.predict(image)
digit = np.argmax(prediction)
print('The predicted digit is:', digit)

image = cv2.imread('new2.jpg', cv2.IMREAD_GRAYSCALE)
#resized_image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_CUBIC)
resized_image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_LANCZOS4)/255.0
image = np.expand_dims(resized_image, axis=-1)
image = np.expand_dims(resized_image, axis=0)

# Make a prediction
prediction = model.predict(image)
digit = np.argmax(prediction)
print('The predicted digit is:', digit)

image = cv2.imread('new3.jpg', cv2.IMREAD_GRAYSCALE)
#resized_image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_CUBIC)
resized_image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_LANCZOS4)/255.0
image = np.expand_dims(resized_image, axis=-1)
image = np.expand_dims(resized_image, axis=0)

# Make a prediction
prediction = model.predict(image)
digit = np.argmax(prediction)
print('The predicted digit is:', digit)