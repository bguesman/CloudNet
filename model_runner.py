import numpy as np
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

import tensorflow_addons as tfa
import cv2
import os
import math
import sys
import autoencoder

def setup_model():
    return autoencoder.AutoEncoder()

def train_batch(autoencoder, batch, epoch, batch_index):
    with tf.GradientTape() as tape:
        # Make a prediction.
        prediction = autoencoder(batch)
        # Compute the loss, which is just the difference between the
        # prediction and the original image.
        loss = autoencoder.loss(prediction, batch)
        if (epoch % 1 == 0 and batch_index % 200 == 0):
            print("LOSS, epoch " + str(epoch) + " batch " + str(batch_index) + ": " + str(loss))
        
    # Apply the gradients.
    gradients = tape.gradient(loss, autoencoder.trainable_variables)
    autoencoder.optimizer.apply_gradients(zip(gradients, autoencoder.trainable_variables))


def train_epoch(autoencoder, images, epoch, batch_size=8):
    # Shuffle
    np.random.shuffle(images)
    tensor_images = tf.convert_to_tensor(images, dtype=tf.float32)
    # tensor_images = tfa.image.rotate(tensor_images, 2 * math.pi * tf.cast(tf.random.uniform([tensor_images.shape[0]], maxval=4, dtype=tf.int32), dtype="float"), fill_mode="WRAP")

    # Split into batches.
    num_batches = int(images.shape[0] / batch_size)
    for i in range(num_batches):
        train_batch(autoencoder,
            tensor_images[i * batch_size:(i + 1) * batch_size, :, :, :], epoch, i)  

def train(autoencoder, images, out_path, epochs=1):
    # Grab the test image before we shuffle
    test_image = images[0:1,:,:,:]
    for i in range(epochs):
        train_epoch(autoencoder, images, i)
        if i % 1 == 0:
            test(autoencoder, test_image, out_path + "/test_" + str(i))
        # TODO: checkpoint

def test(autoencoder, image, path):
    # Generate a random image
    generated = autoencoder(image)
    cv2.imwrite(path + "_real.png", 255 * np.clip(tf.squeeze(image[0,:,:,0]).numpy(), 0, 1))
    cv2.imwrite(path + "_generated.png", 255 * np.clip(tf.squeeze(generated[0,:,:,0]).numpy(), 0, 1))

def load_images(path):
    images = []
    for i, file in enumerate(os.listdir(path)):
        print("Reading file " + str(i) + ": " + file)
        # Read the image as single-channel and normalize to [0, 1]
        img = cv2.imread(path + "/" + file, cv2.IMREAD_GRAYSCALE)
        img = np.float32(img) / 255.0

        images.append(img)
    
    images = np.array(images)
    # Expand so final dimension is channels
    images = np.expand_dims(images, len(images.shape))
    return images

def run(data_path, out_path):
    """
    Runs entirety of model: trains, checkpoints, tests.
    """
    # Get the data.
    images = load_images(data_path)

    # Create the model.
    autoencoder = setup_model()

    # Train the model
    k_epochs = 2000
    train(autoencoder, images, out_path, k_epochs)

    # TODO: test on test data

# Run the script
if (len(sys.argv) != 3):
    print("Usage: python preprocess.py <train data path> <output path>")
train_path = sys.argv[1]
out_path = sys.argv[2]
run(train_path, out_path)