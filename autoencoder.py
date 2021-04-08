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
        self.learning_rate = learning_rate
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        # Initialization standard deviation for the convolutional layers.
        self.initialize_std_dev = 0.05

        # Low res map
        self.low_res = 16
        
        # Dimension of encoded latent space
        self.latent_dimension = 512

        # Downsample layers. The goal here is to squish our image down to a latent code.
        self.encode_resolutions = [128, 64, 32, 16, 8, 4]
        self.encode_filters = [16, 32, 64, 128, 256, 512]
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

        # Loss diffs at which point to stop training particular resolutions.
        self.stop_loss_diffs = [0.001, 0.01, 1, 5]

        # Upsample layers. The goal here is to expand our latent code back out to a full image.
        self.current_resolution_index = 0
        self.decode_resolutions = [32, 64, 128, 256]
        self.decode_filters = [512, 256, 128, 64]
        self.decode_layers = [2, 2, 2, 2]

        self.decode_lr = [[tf.keras.layers.LeakyReLU() for _ in self.decode_layers] for _ in self.decode_resolutions]
        self.decode_convs = [[tf.keras.layers.Conv2D(filters=filters, kernel_size=3, 
                    strides=1, padding='same', 
                    kernel_initializer=tf.keras.initializers.TruncatedNormal(0, self.initialize_std_dev), 
                    bias_initializer=tf.keras.initializers.TruncatedNormal(0, self.initialize_std_dev)) 
                for _ in self.decode_layers] 
            for filters in self.decode_filters]
        
        # Style injection from latent code using adaptive instance normalization 
        # (https://arxiv.org/pdf/1703.06868.pdf).
        self.style_in = [[tfa.layers.InstanceNormalization(axis=3, center=False, scale=False) for _ in self.decode_layers] for _ in self.decode_resolutions]
        # Learned mappings from style to feature standard deviation and mean.
        self.style_std_dev_map = [[tf.keras.layers.Dense(filters) for _ in self.decode_layers] for filters in self.decode_filters]
        self.style_mean_map = [[tf.keras.layers.Dense(filters) for _ in self.decode_layers] for filters in self.decode_filters]

        # Output layers for each resolution.
        self.output_conv = [tf.keras.layers.Conv2D(filters=1, kernel_size=3, 
            strides=1, padding='same', 
            kernel_initializer=tf.keras.initializers.TruncatedNormal(0, self.initialize_std_dev), 
            bias_initializer=tf.keras.initializers.TruncatedNormal(0, self.initialize_std_dev)) for _ in self.decode_resolutions]


    def call(self, input):
        encoded = self.encode(input)
        return self.decode(encoded, tf.image.resize(input, (self.low_res, self.low_res)))
    
    def encode(self, input):
        # Squish the input image down to the latent state.
        encoded = input
        for i, res in enumerate(self.encode_resolutions):
            encoded = self.encode_convs[i](encoded)
            encoded = self.encode_lr[i](self.encode_b[i](encoded))
            encoded = self.encode_max_pool[i](encoded)
        encoded = self.encode_flatten(encoded)
        encoded = self.encode_d0_lr(self.encode_d0_b(self.encode_d0(encoded)))
        return tf.linalg.normalize(self.encode_d1(encoded), axis=1)[0]

    def decode(self, encoded, low_res):
        # Expand the low res image back into a high res image.
        decoded = low_res
        for i in range(self.current_resolution_index + 1):
            res = self.decode_resolutions[i]
            for j in range(self.decode_layers[i]):
                decoded = self.decode_convs[i][j](decoded)
                decoded = self.ada_in(self.decode_lr[i][j](decoded), encoded, i, j)
                decoded = tf.image.resize(decoded, (res, res))
        # Use the output layer for the current resolution we're training at.
        return self.output_conv[self.current_resolution_index](decoded)

    def ada_in(self, images, style, i, j):
        # Normalize with instance normalization layer.
        normalized = self.style_in[i][j](images)
        mapped_std_dev = self.style_std_dev_map[i][j](style)
        mapped_mean = self.style_mean_map[i][j](style)
        # Expand across spatial resolution for broadcasting.
        mapped_std_dev = tf.expand_dims(tf.expand_dims(mapped_std_dev, axis=1), axis=1)
        mapped_mean = tf.expand_dims(tf.expand_dims(mapped_mean, axis=1), axis=1)
        return mapped_std_dev * normalized + mapped_mean

    def advance_resolution(self):
        self.current_resolution_index = min(self.current_resolution_index + 1, len(self.decode_resolutions) - 1)
        print("")
        print("*****************************************************")
        print("*****************************************************")
        print("************ ADVANCING RESOLUTION TO " + str(self.decode_resolutions[self.current_resolution_index]) + " *************")
        print("*****************************************************")
        print("*****************************************************")
        print("")
        # Reset the optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        # Make sure we only train the right output layer for this resolution
        for i, layer in enumerate(self.output_conv):
            layer.trainable = (i == self.current_resolution_index)

    def get_stop_loss_diff(self):
        return self.stop_loss_diffs[self.current_resolution_index]

    def loss(self, prediction, ground_truth):
        # The loss is just the L2 norm between the prediction and the ground truth.
        # Basically, just the squared difference between the two images.
        diff = prediction - tf.image.resize(ground_truth, (prediction.shape[1], prediction.shape[2]))
        return tf.reduce_sum(diff * diff)
