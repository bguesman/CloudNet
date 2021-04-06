import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

import tensorflow_addons as tfa

"""

This class is where we define the autoencoder model as an object.

"""

class AutoEncoder(tf.keras.Model):
    def __init__(self, learning_rate=1e-3):

        # I don't know why this is here, but I'm afraid to touch it.
        ######vvv DO NOT CHANGE vvvv##############
        super(AutoEncoder, self).__init__()
        ######^^^ DO NOT CHANGE ^^^##################

        # Define the optimizer we want to use to train the model.
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # Initialization standard deviation for the convolutional layers.
        self.initialize_std_dev = 0.05

        # Low res map
        self.low_res = 16
        
        # Dimension of encoded latent space
        self.latent_dimension = 2048

        # Downsample layers. The goal here is to squish our image down to a latent code.
        self.encode_resolutions = [128, 64, 32, 16, 8, 4]
        self.encode_filters = [16, 16, 32, 32, 64, 128]
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
        self.encode_d0 = tf.keras.layers.Dense(self.latent_dimension)
        self.encode_d0_lr = tf.keras.layers.LeakyReLU()
        self.encode_d0_b = tf.keras.layers.BatchNormalization()
        self.encode_d1 = tf.keras.layers.Dense(self.latent_dimension)

        # Upsample layers. The goal here is to expand our latent code back out to a full image.
        self.decode_resolutions = [8, 16, 32, 64, 128, 128, 256, 256]
        self.decode_filters = [128, 64, 32, 32, 16, 16, 16, 16]
        self.decode_lr = [tf.keras.layers.LeakyReLU() for i in range(len(self.decode_resolutions))]
        self.decode_b = [tf.keras.layers.BatchNormalization() for i in range(len(self.decode_resolutions))]
        self.decode_convs = [tf.keras.layers.Conv2D(filters=filters, kernel_size=3, 
            strides=1, padding='same', 
            kernel_initializer=tf.keras.initializers.TruncatedNormal(0, self.initialize_std_dev), 
            bias_initializer=tf.keras.initializers.TruncatedNormal(0, self.initialize_std_dev)) for filters in self.decode_filters]
        # Final layer.
        self.final_conv = tf.keras.layers.Conv2D(filters=1, kernel_size=3, 
            strides=1, padding='same', 
            kernel_initializer=tf.keras.initializers.TruncatedNormal(0, self.initialize_std_dev), 
            bias_initializer=tf.keras.initializers.TruncatedNormal(0, self.initialize_std_dev))


    @tf.function
    def call(self, input):
        encoded = self.encode(input)
        return self.decode(encoded, tf.image.resize(input, (self.low_res, self.low_res)))
    
    @tf.function
    def encode(self, input):
        # Squish the input image down to the latent state.
        encoded = input
        for i, res in enumerate(self.encode_resolutions):
            encoded = self.encode_convs[i](encoded)
            encoded = self.encode_lr[i](self.encode_b[i](encoded))
            encoded = self.encode_max_pool[i](encoded)
        encoded = self.encode_flatten(encoded)
        encoded = self.encode_d0_lr(self.encode_d0_b(self.encode_d0(encoded)))
        return self.encode_d1(encoded)

    def decode(self, encoded, low_res):
        # Expand the low res image back into a high res image.
        decoded = tf.reshape(encoded, [encoded.shape[0], 4, 4, int(self.latent_dimension/(4 * 4))]) #low_res
        for i, res in enumerate(self.decode_resolutions):
            decoded = self.decode_convs[i](decoded)
            decoded = self.decode_lr[i](self.decode_b[i](decoded)) # TODO: might remove batch norm
            decoded = tf.image.resize(decoded, (res, res))
        return self.final_conv(decoded)

    def loss(self, prediction, ground_truth):
        # The loss is just the L2 norm between the prediction and the ground truth.
        # Basically, just the squared difference between the two images.
        diff = prediction - ground_truth
        return tf.reduce_sum(diff * diff)
