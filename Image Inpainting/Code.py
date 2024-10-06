import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Load and preprocess the CIFAR-10 dataset
(x_train, _), (x_test, _) = tf.keras.datasets.cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Define a simple mask for the images
def create_masked_images(images):
    masked_images = images.copy()
    mask_size = 10
    center = 16  # Center position for 32x32 images
    masked_images[:, center-mask_size//2:center+mask_size//2, center mask_size//2:center+mask_size//2, :] = 0
    return masked_images

x_train_masked = create_masked_images(x_train)
x_test_masked = create_masked_images(x_test)

# Define a simplified autoencoder model
def build_autoencoder():
    input_img = layers.Input(shape=(32, 32, 3))

    # Encoder
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    encoded = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2, 2), padding='same')(encoded)

    # Decoder
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = models.Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

autoencoder = build_autoencoder()
autoencoder.summary()

# Train the autoencoder with fewer epochs and a smaller batch size
autoencoder.fit(x_train_masked, x_train,
                epochs=10,
                batch_size=64,
                shuffle=True,
                validation_data=(x_test_masked, x_test))

# Make predictions on the test set
decoded_imgs = autoencoder.predict(x_test_masked)

# Visualize the results
n = 10  # Number of images to display
plt.figure(figsize=(15, 5))
for i in range(n):
    # Display masked images
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(x_test_masked[i])
    plt.title("Masked")
    plt.axis('off')

    # Display original images
    ax = plt.subplot(3, n, i + n + 1)
    plt.imshow(x_test[i])
    plt.title("Original")
    plt.axis('off')

    # Display reconstructed images
    ax = plt.subplot(3, n, i + 2 * n + 1)
    plt.imshow(decoded_imgs[i])
    plt.title("Reconstructed")
    plt.axis('off')

plt.show()

