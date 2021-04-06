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

        # Downsample layers
        self.encode_resolutions = [128, 64, 32, 16, 8, 4]
        self.encode_filters = [16, 16, 32, 32, 64, 128]
        self.encode_lr = [tf.keras.layers.LeakyReLU() for i in range(len(self.encode_resolutions))]
        self.encode_b = [tf.keras.layers.BatchNormalization() for i in range(len(self.encode_resolutions))]
        self.encode_convs = [tf.keras.layers.Conv2D(filters=filters, kernel_size=5, 
            strides=1, padding='same', 
            kernel_initializer=tf.keras.initializers.TruncatedNormal(0, self.initialize_std_dev), 
            bias_initializer=tf.keras.initializers.TruncatedNormal(0, self.initialize_std_dev)) for filters in self.encode_filters]
        self.encode_flatten = tf.keras.layers.Flatten()
        self.encode_d0 = tf.keras.layers.Dense(self.latent_dimension)
        self.encode_d0_lr = tf.keras.layers.LeakyReLU()
        self.encode_d0_b = tf.keras.layers.BatchNormalization()
        self.encode_d1 = tf.keras.layers.Dense(self.latent_dimension)

        # Upsample layers
        self.decode_resolutions = [8, 16, 32, 64, 128, 128, 256, 256]
        self.decode_filters = [128, 64, 32, 32, 16, 16, 16, 16]
        self.decode_lr = [tf.keras.layers.LeakyReLU() for i in range(len(self.decode_resolutions))]
        self.decode_b = [tf.keras.layers.BatchNormalization() for i in range(len(self.decode_resolutions))]
        self.decode_convs = [tf.keras.layers.Conv2D(filters=filters, kernel_size=3, 
            strides=1, padding='same', 
            kernel_initializer=tf.keras.initializers.TruncatedNormal(0, self.initialize_std_dev), 
            bias_initializer=tf.keras.initializers.TruncatedNormal(0, self.initialize_std_dev)) for filters in self.decode_filters]

        # We'll use the latent code as a style vector in the decoding process
        # For each AdaIN layer, we need an instance normalization layer
        # and a learned affine transformation that will be applied to the
        # style vector.
        # self.ada_in_norm = [tfa.layers.InstanceNormalization() for _ in self.decode_filters]
        # self.ada_in_transform = [tf.Variable(tf.random.normal([filters, self.latent_dimension])) for filters in self.decode_filters]
        # self.ada_in_bias = [tf.Variable(tf.random.normal([1, filters])) for filters in self.decode_filters]


        # Final layer
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
            encoded = tf.image.resize(encoded, (res, res))
        encoded = self.encode_flatten(encoded)
        encoded = self.encode_d0_lr(self.encode_d0_b(self.encode_d0(encoded)))
        return self.encode_d1(encoded)

    def decode(self, encoded, low_res):
        # Expand the low res image back into a high res image, using the
        # latent state as a style vector.
        decoded = tf.reshape(encoded, [encoded.shape[0], 4, 4, int(self.latent_dimension/(4 * 4))]) #low_res
        for i, res in enumerate(self.decode_resolutions):
            # Concatenate feature vector as extra channels
            # encoded_tiled = tf.reshape(encoded, [encoded.shape[0], 1, 1, encoded.shape[1]])
            # encoded_tiled = tf.repeat(encoded_tiled, decoded.shape[1], axis=1)
            # encoded_tiled = tf.repeat(encoded_tiled, decoded.shape[2], axis=2)
            # decoded = tf.concat([decoded, encoded_tiled], axis=3)
            decoded = self.decode_convs[i](decoded)
            decoded = self.decode_lr[i](self.decode_b[i](decoded)) # TODO: might remove batch norm
            decoded = tf.image.resize(decoded, (res, res))
        return self.final_conv(decoded)

    # def ada_in(self, decoded, style, i):
    #     # Bias: shape is [batches, # of features]
    #     bias = tf.repeat(self.ada_in_bias[i], decoded.shape[0], axis=0)
    #     # Multiplier: shape is [batches, # of features]
    #     # Affine transform: shape is [# of features, map_dimension]
    #     # Style: shape is [batches, map_dimension]
    #     multiplier = tf.einsum("bm,fm->bf", style, self.ada_in_transform[i])
    #     # output = multiplier * output + bias, for each batch item, for 
    #     # each feature
    #     multiplier = tf.reshape(multiplier, [multiplier.shape[0], 1, 1, multiplier.shape[1]])
    #     bias = tf.reshape(bias, [bias.shape[0], 1, 1, bias.shape[1]])
    #     return multiplier * decoded + bias

    def loss(self, prediction, ground_truth):
        # The loss is just the L2 norm between the prediction and the ground truth.
        # Basically, just the squared difference between the two images.
        diff = prediction - ground_truth
        return tf.reduce_sum(diff * diff)
