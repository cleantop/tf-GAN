import tensorflow as tf

class ConditionalGAN(object):

    def __init__(self, num_labels, noise_dimension, image_dimension, hidden_dimension):
        # Generator Variables
        self.G_w1 = tf.get_variable(name="G_W1", shape=[100, 128], initializer=tf.contrib.layers.xavier_initializer())
        self.G_b1 = tf.Variable(tf.constant(0.0, shape=[128], name="G_b1"))
        self.G_w2 = tf.get_variable(name="G_W2", shape=[128, 784], initializer=tf.contrib.layers.xavier_initializer())
        self.G_b2 = tf.Variable(tf.constant(0.0, shape=[784], name="G_b2"))

        self.G_C_w1 = tf.get_variable(name="G_C_W1", shape=[num_labels, 128], initializer=tf.contrib.layers.xavier_initializer())

        self.G_vars = [self.G_w1, self.G_b1, self.G_w2, self.G_b2, self.G_C_w1]

        # Discriminator Variables
        self.D_w1 = tf.get_variable(name="D_W1", shape=[784, 128], initializer=tf.contrib.layers.xavier_initializer())
        self.D_b1 = tf.Variable(tf.constant(0.0, shape=[128], name="D_b1"))
        self.D_w2 = tf.get_variable(name="D_W2", shape=[128, 1], initializer=tf.contrib.layers.xavier_initializer())
        self.D_b2 = tf.Variable(tf.constant(0.0, shape=[1], name="D_b2"))

        self.D_C_w1 = tf.get_variable(name="D_C_W1", shape=[num_labels, 128], initializer=tf.contrib.layers.xavier_initializer())

        self.D_vars = [self.D_w1, self.D_b1, self.D_w2, self.D_b2, self.D_C_w1]

        # Random noise setting for Generator
        self.z = tf.placeholder(tf.float32, shape=[None, noise_dimension])

        # Real image
        self.x = tf.placeholder(tf.float32, shape=[None, image_dimension])

        # Condition variable
        self.c = tf.placeholder(tf.float32, shape=[None, num_labels])

        self.keep_prob = tf.placeholder(tf.float32)

        # Fake image
        self.G_sample = self.generator(self.z, self.c)

        D_real, D_logit_real = self.discriminator(self.x, self.c)
        D_fake, D_logit_fake = self.discriminator(self.G_sample, self.c)

        # Loss functions
        # Loss 에서 NaN 이 나오는 경우는, loss가 확률 0~1 로 구성되어 있고 로그의 진수가 0이 되는 경우 log는 음의 무한대로 발산하기 때문
        self.D_loss = -tf.reduce_mean(self.clip_log(D_real) + self.clip_log(1.0 - D_fake))
        self.G_loss = -tf.reduce_mean(self.clip_log(D_fake))

    # Generate fake image(28*28) from noise distribution
    def generator(self, z, c):
        # Noise to hidden
        G_h1 = tf.nn.relu(tf.matmul(z, self.G_w1) + tf.matmul(c, self.G_C_w1) +self.G_b1)

        # Hidden to fake image
        G_logit = tf.matmul(G_h1, self.G_w2) + self.G_b2
        G_prob = tf.nn.sigmoid(G_logit)

        return G_prob

    # Determine that input is real or fake image
    def discriminator(self, x, c):
        # Image to hidden
        D_h1 = tf.nn.relu(tf.matmul(x, self.D_w1) + tf.matmul(c, self.D_C_w1) + self.D_b1)

        # Hidden to fake image
        D_logit = tf.matmul(D_h1, self.D_w2) + self.D_b2
        D_prob = tf.nn.sigmoid(D_logit)

        return D_prob, D_logit

    def clip_log(self, x):
        return tf.log(tf.clip_by_value(x, 1e-10, 1.0))