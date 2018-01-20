import tensorflow as tf

class DCGAN(object):
    def __init__(self, noise_dimension, image_dimension, batch_size):

        # Random noise setting for Generator
        self.z = tf.placeholder(tf.float32, shape=[batch_size, noise_dimension])

        # Real image
        self.x = tf.placeholder(tf.float32, shape=[batch_size, image_dimension])
        real_image = tf.reshape(self.x, shape=[-1, 28, 28, 1])

        # Training Phase
        self.phase_train = tf.placeholder(tf.bool, name='phase_train')

        # Fake image
        self.G_sample = self.generator(self.z)

        with tf.variable_scope("call_generator") as scope:
            D_real = self.discriminator(real_image)
            scope.reuse_variables()
            D_fake = self.discriminator(self.G_sample)

        # Loss functions
        self.D_loss = -tf.reduce_mean(self.clip_log(D_real) + self.clip_log(1.0 - D_fake))
        self.G_loss = -tf.reduce_mean(self.clip_log(D_fake))

    def generator(self, z):
        with tf.variable_scope("generator"):
            x = tf.layers.dense(inputs=z, units=7*7*128)
            x = tf.nn.relu(x)
            x = tf.layers.batch_normalization(x)
            x = tf.reshape(x, [-1, 7, 7, 128])

            x = tf.layers.conv2d_transpose(inputs=x, filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same')# [-1, 14, 14, 64]
            x = tf.nn.relu(x)
            x = tf.reshape(x, [-1, 14*14*64])
            x = tf.layers.batch_normalization(x)
            x = tf.reshape(x, [-1, 14, 14, 64])

            x = tf.layers.conv2d_transpose(inputs=x, filters=1, kernel_size=(3, 3), strides=(2, 2), padding='same')# [-1, 28, 28, 1]
            x = tf.nn.tanh(x)
        return x

    def discriminator(self, x):
        with tf.variable_scope("discriminator"):
            x = tf.layers.conv2d(inputs=x, filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same')
            x = self.leakyReLU(x)

            x = tf.layers.conv2d(inputs=x, filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same')
            x = self.leakyReLU(x)
            x = tf.reshape(x, [-1, 7*7*128])
            x = tf.layers.batch_normalization(x)
            x = tf.reshape(x, [-1, 7, 7, 128])

            x = tf.layers.dense(inputs=x, units=10)
            prob = tf.nn.sigmoid(x)
        return prob


    def leakyReLU(self, x, leak=0.2):
        return tf.maximum(x, leak*x)

    def clip_log(self, x):
        return tf.log(tf.clip_by_value(x, 1e-10, 1.0))

