import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

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
        # self.initialize_std_dev = 0.1
        
        # Batch norms and activations for convolutional layers.
        # self.lr = [tf.keras.layers.LeakyReLU() for i in range(7)]
        # self.b = [tf.keras.layers.BatchNormalization() for i in range(7)]

        # 4x4 => 256x256 means we need 6 convolutional layers, plus one final one
        # self.c0 = tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=1, padding='same', kernel_initializer=tf.keras.initializers.TruncatedNormal(0, self.initialize_std_dev), bias_initializer=tf.keras.initializers.TruncatedNormal(0, self.initialize_std_dev))

    @tf.function
    def call(self, input):
        # The stupidest possible auto-encoder... just return the input lol.
        return input
    
    def loss(self, prediction, ground_truth):
        # The loss is just the L2 norm between the prediction and the ground truth.
        # Basically, just the squared difference between the two images.
        diff = prediction - ground_truth
        return tf.reduce_sum(diff * diff)
