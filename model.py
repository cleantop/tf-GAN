import tensorflow as tf


class SimpleGAN(object):

    def __init__(self):

        # Generator Variables
        self.G_w1 = tf.get_variable(name="G_W1", shape=[100, 128], initializer=tf.contrib.layers.xavier_initializer())
        self.G_b1 = tf.Variable(tf.constant(0.0, shape=[128], name="G_b1"))
        self.G_w2 = tf.get_variable(name="G_W2", shape=[128, 784], initializer=tf.contrib.layers.xavier_initializer())
        self.G_b2 = tf.Variable(tf.constant(0.0, shape=[784], name="G_b2"))

        self.G_vars = [self.G_w1, self.G_b1, self.G_w2, self.G_b2]

        # Discriminator Variables
        self.D_w1 = tf.get_variable(name="D_W1", shape=[784, 128], initializer=tf.contrib.layers.xavier_initializer())
        self.D_b1 = tf.Variable(tf.constant(0.0, shape=[128], name="D_b1"))
        self.D_w2 = tf.get_variable(name="D_W2", shape=[128, 1], initializer=tf.contrib.layers.xavier_initializer())
        self.D_b2 = tf.Variable(tf.constant(0.0, shape=[1], name="D_b2"))

        self.D_vars = [self.D_w1, self.D_b1, self.D_w2, self.D_b2]


        # Random noise setting for Generator
        self.z = tf.placeholder(tf.float32, shape=[None, 100])

        # Real image
        self.x = tf.placeholder(tf.float32, shape=[None, 784])

        self.G_sample = self.generator(self.z)

        D_real, D_logit_real = self.discriminator(self.x)
        D_fake, D_logit_fake = self.discriminator(self.G_sample)

        # Loss functions
        self.D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1.0 - D_fake))
        self.G_loss = -tf.reduce_mean(tf.log(D_fake))

    # Generate fake image(28*28) from noise distribution
    def generator(self, z):
        # Noise to hidden
        G_h1 = tf.nn.relu(tf.matmul(z, self.G_w1) + self.G_b1)

        # Hidden to fake image
        G_logit = tf.matmul(G_h1, self.G_w2) + self.G_b2
        G_prob = tf.nn.sigmoid(G_logit)

        return G_prob

    # Determine that input is real or fake image
    def discriminator(self, x):

        # Image to hidden
        D_h1 = tf.nn.relu(tf.matmul(x, self.D_w1) + self.D_b1)

        # Hidden to fake image
        D_logit = tf.matmul(D_h1, self.D_w2) + self.D_b2
        D_prob = tf.nn.sigmoid(D_logit)

        return D_prob, D_logit


