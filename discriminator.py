import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

import tensorflow_addons as tfa

"""

This class is where we define the autoencoder model as an object.

"""

class Discriminator(tf.keras.Model):
    def __init__(self, learning_rate=1e-4):

        # I don't know why this is here, but I'm afraid to touch it.
        ######vvv DO NOT CHANGE vvvv##############
        super(Discriminator, self).__init__()
        ######^^^ DO NOT CHANGE ^^^##################

        # Define the optimizer we want to use to train the model.
        self.learning_rate = learning_rate
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.5)

        # Initialization standard deviation for the convolutional layers.
        self.initialize_std_dev = 0.05

        # Downsample layers. The goal here is to squish our image down to a latent code.
        self.encode_resolutions = [256, 128, 64, 32, 16, 8, 4]
        self.current_resolution_index = len(self.encode_resolutions) - 1
        self.encode_filters = [8, 16, 32, 64, 128, 256, 512]
        self.encode_convs = [tf.keras.layers.Conv2D(filters=filters, kernel_size=5, 
            strides=1, padding='same', 
            kernel_initializer=tf.keras.initializers.TruncatedNormal(0, self.initialize_std_dev), 
            bias_initializer=tf.keras.initializers.TruncatedNormal(0, self.initialize_std_dev)) for filters in self.encode_filters]
        # We'll use max pooling for the actual squishing.
        self.encode_max_pool = [tf.keras.layers.MaxPool2D((2, 2), padding="same") for _ in self.encode_resolutions]
        # Activation function is leaky relu, which has been shown to be pretty good for this kind of thing.
        self.encode_lr = [tf.keras.layers.LeakyReLU() for _ in self.encode_resolutions]
        # We'll also batch normalize the output of each convolutional layer to improve training stability.
        self.encode_b = [tf.keras.layers.BatchNormalization() for _ in self.encode_resolutions]
        # At the end of the triaged convolutions, we flatten the result and then pass it through two 
        # linear layers to get our final latent code.
        self.encode_flatten = tf.keras.layers.Flatten()
        self.encode_d0 = tf.keras.layers.Dense(512)
        self.encode_d0_lr = tf.keras.layers.LeakyReLU()
        self.encode_d0_b = tf.keras.layers.BatchNormalization()
        self.encode_d1 = tf.keras.layers.Dense(1)


    def call(self, input):
        return self.encode(input)
    
    def encode(self, input):
        # Squish the input image down to the latent state.
        encoded = input
        for i in range(self.current_resolution_index, len(self.encode_resolutions)):
            res = self.encode_resolutions[i]
            encoded = self.encode_convs[i](encoded)
            encoded = self.encode_lr[i](self.encode_b[i](encoded))
            encoded = self.encode_max_pool[i](encoded)
        encoded = self.encode_flatten(encoded)
        encoded = self.encode_d0_lr(self.encode_d0_b(self.encode_d0(encoded)))
        return self.encode_d1(encoded)

    def advance_resolution(self):
        self.current_resolution_index = max(self.current_resolution_index - 1, 0)
        # Reset the optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def get_current_resolution(self):
        return self.encode_resolutions[self.current_resolution_index]

    def loss(self, prediction, ground_truth):
        return tf.reduce_sum(tf.keras.losses.binary_crossentropy(ground_truth, prediction, from_logits=True)) / ground_truth.shape[0]
