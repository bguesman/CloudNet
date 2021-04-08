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
import discriminator
import style_generator

k_fake = 0
k_real = 1

def setup_model():
    return discriminator.Discriminator(), style_generator.StyleGenerator()

def print_avg_loss(avg_disc_loss, avg_gen_loss, epoch):
    print("AVERAGE LOSSES FOR EPOCH " + str(epoch))
    print("DISCRIMINATOR: " + str(avg_disc_loss))
    print("GENERATOR: " + str(avg_gen_loss))

def generate_random_style(batch_size, latent_dimension):
    return tf.linalg.normalize(tf.random.normal((batch_size, latent_dimension)), axis=1)[0]

def train_discriminator(discriminator, generator, batch):
    style = generate_random_style(batch.shape[0], generator.latent_dimension)
    disc_res = discriminator.get_current_resolution()
    with tf.GradientTape() as tape:
        generated = generator(style)
        real_and_fake = tf.concat([generated, tf.image.resize(batch, (disc_res, disc_res))], axis=0)
        prediction = discriminator(real_and_fake)
        truth = tf.concat([tf.fill((batch.shape[0], 1), k_fake), tf.fill((batch.shape[0], 1), k_real)], axis=0)
        loss = discriminator.loss(prediction, truth)
    # Apply gradients to the generator
    gradients = tape.gradient(loss, discriminator.trainable_variables)
    discriminator.optimizer.apply_gradients(zip(gradients, discriminator.trainable_variables))
    return loss

def train_generator(discriminator, generator, batch):
    style = generate_random_style(batch.shape[0] * 2, generator.latent_dimension)
    with tf.GradientTape() as tape:
        generated = generator(style)
        prediction = discriminator(generated)
        loss = discriminator.loss(prediction, tf.fill((batch.shape[0] * 2, 1), k_fake))
        negative_loss = -loss
    # Apply gradients to the generator
    gradients = tape.gradient(negative_loss, generator.trainable_variables)
    generator.optimizer.apply_gradients(zip(gradients, generator.trainable_variables))
    return loss

def train_batch(discriminator, generator, batch):
    discriminator_loss = train_discriminator(discriminator, generator, batch)
    generator_loss = train_generator(discriminator, generator, batch)
    return discriminator_loss, generator_loss

def train_epoch(discriminator, generator, images, epoch, batch_size=8):
    # Shuffle
    np.random.shuffle(images)
    tensor_images = tf.convert_to_tensor(images, dtype=tf.float32)
    # Split into batches.
    avg_disc_loss = 0
    avg_gen_loss = 0
    num_batches = int(images.shape[0] / batch_size)
    for i in range(num_batches):
        disc_loss, gen_loss = train_batch(discriminator, generator, tensor_images[i * batch_size:(i + 1) * batch_size, :, :, :])  
        avg_disc_loss += disc_loss 
        avg_gen_loss += gen_loss
    return avg_disc_loss / num_batches, avg_gen_loss / num_batches

# Maps image from [-1, 1] to [0, 255]
def map_to_8_bit(image):
    return np.clip((image + 1) * 0.5 * 255, 0, 255)

# Maps image from [0, 255] to [-1, 1]
def map_to_unit(image):
    return (np.float32(image) / 255.0) * 2 - 1

def train(discriminator, generator, images, out_path, epochs=1):
    # Grab the test image before we shuffle
    test_image = np.copy(images[0:1,:,:,:])
    # Also generate a latent vector to test with.
    test_latent = generate_random_style(3, generator.latent_dimension)

    for i in range(0, epochs):
        # Train the epoch and print the loss.
        avg_disc_loss, avg_gen_loss = train_epoch(discriminator, generator, images, i)
        print_avg_loss(avg_disc_loss, avg_gen_loss, i)
        # Test.
        test(generator, test_image, test_latent, out_path + "/test_" + str(i))
        if (i % (15 * (generator.current_resolution_index + 1)) == 0 and i != 0):
            generator.advance_resolution()
            discriminator.advance_resolution()
        

def test(generator, image, style, path):
    # Create a few different versions of the image.
    target_res = generator.decode_resolutions[generator.current_resolution_index]
    target_res_image = tf.image.resize(image, (target_res, target_res))
    generated = generator(style)
    cv2.imwrite(path + "_target.png", tf.squeeze(map_to_8_bit(target_res_image)[0,:,:,0]).numpy())
    cv2.imwrite(path + "_real.png", tf.squeeze(map_to_8_bit(image)[0,:,:,0]).numpy())
    cv2.imwrite(path + "_generated_0.png", tf.squeeze(map_to_8_bit(generated)[0,:,:,0]).numpy())
    cv2.imwrite(path + "_generated_1.png", tf.squeeze(map_to_8_bit(generated)[1,:,:,0]).numpy())
    cv2.imwrite(path + "_generated_2.png", tf.squeeze(map_to_8_bit(generated)[2,:,:,0]).numpy())

def load_images(path):
    images = []
    for i, file in enumerate(os.listdir(path)):
        print("Reading file " + str(i) + ": " + file)
        # Read the image as single-channel and normalize to [0, 1]
        img = cv2.imread(path + "/" + file, cv2.IMREAD_GRAYSCALE)
        images.append(map_to_unit(img))
    
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
    discriminator, generator = setup_model()

    # Train the model
    k_epochs = 2000
    train(discriminator, generator, images, out_path, k_epochs)

    # TODO: test on test data

# Run the script
if (len(sys.argv) != 3):
    print("Usage: python preprocess.py <train data path> <output path>")
train_path = sys.argv[1]
out_path = sys.argv[2]
run(train_path, out_path)