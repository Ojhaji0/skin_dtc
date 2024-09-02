from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau

# Define the main directory for the dataset
train_dir = '/mnt/d/Users/ayush/OneDrive/Desktop/base_dir/train_dir'
val_dir = '/mnt/d/Users/ayush/OneDrive/Desktop/base_dir/val_dir'

# Image data generator for training data with augmentation and validation data with rescaling
datagen = ImageDataGenerator(
    rescale=1./255,
     
)

# Load training images from the main directory with the defined transformations
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
      # Set as training data
)

# Load validation images from the main directory with only rescaling
validation_generator = datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
      # Set as validation data
)

# Load pre-trained VGG16 model without top layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the convolutional base
for layer in base_model.layers[:-4]:
    layer.trainable = False

# Create new model
model = Sequential([
    base_model,
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),  # Add dropout to prevent overfitting
    Dense(256, activation='relu'),
    Dropout(0.5),  # Add another dropout layer
    Dense(7, activation='softmax')  # Assuming you have 7 classes
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',  # Use categorical cross-entropy for one-hot encoded labels
              metrics=['accuracy'])

# Add a learning rate reduction on plateau callback
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)

# Fit the model
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    callbacks=[reduce_lr]
)

loss, accuracy = model.evaluate(validation_generator, steps=validation_generator.samples // validation_generator.batch_size)
print(f'Validation Accuracy: {accuracy * 100:.2f}%')

model.save('skin_disease_cnn_model98.h5')


import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
class_labels = ['akiec','bcc','bkl','df','mel','nv','vasc']
img_path = '/mnt/c/Users/ayush/OneDrive/Desktop/d.jpg'

# Load and preprocess the image
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Make prediction
prediction = model.predict(img_array)

# Get the predicted class label
predicted_class = np.argmax(prediction)

print("Predicted class:", predicted_class)
print("Prediction probabilities:", prediction)
print("disease ",class_labels[predicted_class])