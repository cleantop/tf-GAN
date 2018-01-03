import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from model import SimpleGAN
from tensorflow.examples.tutorials.mnist import input_data

# Hyper-parameters
epoch = 1000000
batch_size = 100
sample_size = 10

# Load MNIST Dataset
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

config = tf.ConfigProto(allow_soft_placement=True)
with tf.Session(config=config) as sess:
    with sess.as_default():

        # GAN object
        gan = SimpleGAN()

        # Define Training procedures of Generator
        G_global_step = tf.Variable(0, name="G_global_step", trainable=True)
        G_optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        G_grads_and_vars = G_optimizer.compute_gradients(gan.G_loss, var_list=gan.G_vars)
        G_train_op = G_optimizer.apply_gradients(G_grads_and_vars, global_step=G_global_step)

        # Define Training procedures of Discriminator
        D_global_step = tf.Variable(0, name="D_global_step", trainable=True)
        D_optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        D_grads_and_vars = D_optimizer.compute_gradients(gan.D_loss, var_list=gan.D_vars)
        D_train_op = D_optimizer.apply_gradients(D_grads_and_vars, global_step=D_global_step)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def generate_noise_sample(m, n):
            return np.random.uniform(-1, 1, size=[m, n])

        # Training step
        for i in range(epoch):
            x, _ = mnist.train.next_batch(batch_size)

            # Discriminator training
            _, D_step, D_loss = sess.run([D_train_op, D_global_step, gan.D_loss],
                                          feed_dict={gan.x: x, gan.z: generate_noise_sample(batch_size, 100)})

            # Generator training
            _, G_step, G_loss = sess.run([G_train_op, G_global_step, gan.G_loss],
                                          feed_dict={gan.z: generate_noise_sample(batch_size, 100)})

            # Validating image
            if i % 50000 == 0:
                print(i)

                noise = generate_noise_sample(sample_size**2, 100)
                samples = sess.run(gan.G_sample, feed_dict={gan.z: noise})
                fig = plt.figure()
                for s in range(sample_size**2):
                    ax = fig.add_subplot(sample_size, sample_size, s + 1)
                    ax.set_axis_off()
                    ax.imshow(np.reshape(samples[s], (28, 28)))
                plt.savefig('samples/{}.png'.format(str(i)))
                plt.close(fig)