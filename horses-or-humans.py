import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt
from PIL import Image

# Load data:
# _______________________________________________________________________________
PATH = "/Users/bastianhojbjerre/Documents/Skole/6. Semester/AI/horse-or-human"  # directory with the first level in our data set
train_dir = os.path.join(PATH, 'train')  # directory with our training pictures
validation_dir = os.path.join(PATH, 'validation')  # directory with our validation pictures

train_horses_dir = os.path.join(train_dir, 'horses')  # directory with our training horse pictures
train_humans_dir = os.path.join(train_dir, 'humans')  # directory with our training human pictures
validation_horses_dir = os.path.join(validation_dir, 'horses')  # directory with our validation horse pictures
validation_humans_dir = os.path.join(validation_dir, 'humans')  # directory with our validation human pictures
# _______________________________________________________________________________

# Understand the data:
print("Understand the data:")
# _______________________________________________________________________________
num_horses_tr = len(os.listdir(train_horses_dir))
num_humans_tr = len(os.listdir(train_humans_dir))

num_horses_val = len(os.listdir(validation_horses_dir))
num_humans_val = len(os.listdir(validation_humans_dir))

total_train = num_horses_tr + num_humans_tr
total_val = num_horses_val + num_humans_val

print('Total horse training images:', num_horses_tr)
print('Total human training images:', num_humans_tr)
print('Total horse validation images:', num_horses_val)
print('Total human validation images:', num_humans_val)
print("Total training images:", total_train)
print("Total validation images:", total_val)
# _______________________________________________________________________________

# Data preparation:
print("\nData preparation:")
# _______________________________________________________________________________
batch_size = 200  # Amount of images utilized in one iteration:
epochs = 5  # Amount of times the entire data set is passed forward and backward through the neural network:
IMG_HEIGHT = 300  # The images pixel-height:
IMG_WIDTH = 300  # The images pixel-width


train_image_generator = ImageDataGenerator(rescale=1. / 255)  # Generator for our training data
validation_image_generator = ImageDataGenerator(rescale=1. / 255)  # Generator for our validation data

train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary')

val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=validation_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='binary')
# _______________________________________________________________________________

# Visualize training images:
# _______________________________________________________________________________
sample_training_images, _ = next(train_data_gen)


# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5)
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.show()


plotImages(sample_training_images[:5])
# _______________________________________________________________________________

# Create the model:
# _______________________________________________________________________________
model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1)
])

# _______________________________________________________________________________

# Compile the model:
# _______________________________________________________________________________
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
# _______________________________________________________________________________

# Model summary:
print("\nModel summary:")
# _______________________________________________________________________________
model.summary()
# _______________________________________________________________________________

# Train the model:
print("\nTrain the model")
# _______________________________________________________________________________
history = model.fit(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size
)
# _______________________________________________________________________________

# Visualize training results
# _______________________________________________________________________________
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
# _______________________________________________________________________________
