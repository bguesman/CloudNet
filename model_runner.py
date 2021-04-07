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

def print_avg_loss(avg_loss, epoch):
    print("AVERAGE LOSS FOR EPOCH " + str(epoch) + ": " + str(avg_loss))

def train_batch(autoencoder, batch):
    with tf.GradientTape() as tape:
        # Make a prediction.
        prediction = autoencoder(batch)
        # Compute the loss, which is just the difference between the
        # prediction and the original image.
        loss = autoencoder.loss(prediction, batch)
    # Apply the gradients to the model variables.
    gradients = tape.gradient(loss, autoencoder.trainable_variables)
    autoencoder.optimizer.apply_gradients(zip(gradients, autoencoder.trainable_variables))
    return loss

def train_epoch(autoencoder, images, epoch, batch_size=8):
    # Shuffle
    np.random.shuffle(images)
    tensor_images = tf.convert_to_tensor(images, dtype=tf.float32)
    # Split into batches.
    avg_loss = 0
    num_batches = int(images.shape[0] / batch_size)
    for i in range(num_batches):
        avg_loss += train_batch(autoencoder, tensor_images[i * batch_size:(i + 1) * batch_size, :, :, :])  
    return avg_loss / num_batches

def train(autoencoder, images, out_path, epochs=1, resolution_advance_cutoff=0.001):
    # Grab the test image before we shuffle
    test_image = np.copy(images[0:1,:,:,:])

    # Create a list of average losses to use indeciding when to advance resolution
    average_loss_diffs = [-4, -3, -2, -1]
    avg_loss = 10e32

    for i in range(0, epochs):
        prev_avg_loss = avg_loss

        avg_loss = train_epoch(autoencoder, images, i)
        print_avg_loss(avg_loss, i)
        
        test(autoencoder, test_image, out_path + "/test_" + str(i))
        
        average_loss_diffs.pop(0)
        average_loss_diffs.append(avg_loss - prev_avg_loss)
        average_diff = sum(average_loss_diffs) / len(average_loss_diffs)
        print("Average loss diff: " + str(average_diff))
        if average_diff > resolution_advance_cutoff:
            autoencoder.advance_resolution()
            average_loss_diffs = [-4, -3, -2, -1]
            avg_loss = 10e32

def test(autoencoder, image, path):
    generated = autoencoder(image)
    low_res = tf.image.resize(image, (autoencoder.low_res, autoencoder.low_res))
    cv2.imwrite(path + "_low_res.png", 255 * np.clip(tf.squeeze(low_res[0,:,:,0]).numpy(), 0, 1))
    cv2.imwrite(path + "_upscaled.png", 255 * np.clip(tf.squeeze(generated[0,:,:,0]).numpy(), 0, 1))
    cv2.imwrite(path + "_real.png", 255 * np.clip(tf.squeeze(image[0,:,:,0]).numpy(), 0, 1))

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