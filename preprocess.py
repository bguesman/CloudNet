import numpy as np
import cv2
import os
import sys
import random

import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

def preprocess(path, out_path):
    k_image_size = (256, 256)
    k_max_images = 20000
    images = load(path, k_image_size, k_max_images)

    # For the black parts of the image, replace them with the parts of other images
    k_epsilon = 0.05
    images = infill(images, k_epsilon)

    # Turn our list into a numpy array
    images = np.array(images)

    # Clip to remove artifacts
    images = np.clip(images, 0.15, 1)

    # Increase contrast to crush blacks back down to black
    images = np.clip(1.25 * (images - 0.5) + 0.5, 0, 1)

    # Write out images
    for i in range(images.shape[0]):
        cv2.imwrite(out_path + "/out_" + str(i) + ".png", images[i,:,:] * 255)

    # # Optionally view some of the images
    # cv2.imshow("image 0", images[0])
    # cv2.imshow("image 1", images[10])
    # cv2.imshow("image 2", images[20])
    # cv2.imshow("image 3", images[30])
    # cv2.waitKey(0)

    # Expand so final array is channels
    images = np.expand_dims(images, len(images.shape))
    
    print("Processed " + str(images.shape[0]) + " images")
    print("Final shape: " + str(images.shape))
    
    return images

# Loads images at specified path, normalizes to [0, 1], and resizes to specified size
def load(path, size, max_images=1000):
    images = []
    print("Loading " + str(min(max_images, len(os.listdir(path)))) + " images")
    # Load in each image
    for i, file in enumerate(os.listdir(path)):
        # HACK: process at most kMaxImages images
        if i > max_images:
            break
        print("Reading file " + str(i) + ": " + file)

        # Read the image as single-channel and resize it to kImageSize
        img = cv2.imread(path + "/" + file, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, size)
        
        # Normalize to [0, 1]
        img = np.float32(img)
        img = img / 255.0

        images.append(img)
    return images

# Fills in black parts of images with other images based on SSIM
# [BRAD]: there has to be a smarter and faster way to compare SSIM's without
# recomputing it for many pairs each time. Maybe there's some sort of data structure
# we can build before the infill step that groups images by their related SSIM's?
def infill(images, epsilon):
    for i, img in enumerate(images):
        zero_pixels = (img < epsilon)
        attempts = 0
        ssimMin = 0.2
        while (np.any(zero_pixels)):
            if attempts > 15:
                ssimMin = 0.15
            if attempts > 25:
                ssimMin = 0.1
            if attempts > 45:
                ssimMin = 0.0
            # Pick a random image.
            replacement = random.choice(images)
            ssim = tf.image.ssim(img[:, :, np.newaxis], replacement[:, :, np.newaxis], max_val=1.0)
            # print("SSIM: " + str(ssim))
            if (ssim > ssimMin):
                blend_factor = (np.clip(epsilon - img[zero_pixels], 0, epsilon) / epsilon)
                img[zero_pixels] = blend_factor * replacement[zero_pixels] + (1 - blend_factor) * img[zero_pixels]
                zero_pixels = (img < epsilon)
            attempts += 1
        print("infill completed for image " + str(i))
    return images

# Run
if (len(sys.argv) != 3):
    print("Usage: python preprocess.py <input path> <output path>")
path_in = sys.argv[1]
path_out = sys.argv[2]
preprocess(path_in, path_out)