import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from model import ConditionalGAN
from tensorflow.examples.tutorials.mnist import input_data

# Hyper-parameters
epoch = 1000000
batch_size = 100
sample_size = 10

num_labels = 10
noise_dimension = 100
hidden_dimension = 128
image_dimension = 784 # 28*28

# Load MNIST Dataset
#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
mnist = input_data.read_data_sets("data/fashion", one_hot=True)

config = tf.ConfigProto(allow_soft_placement=True)
with tf.Session(config=config) as sess:
    with sess.as_default():

        # GAN object
        gan = ConditionalGAN(noise_dimension=noise_dimension,
                             hidden_dimension=hidden_dimension,
                             image_dimension=image_dimension,
                             num_labels=num_labels)

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
            x, y = mnist.train.next_batch(batch_size)

            # Discriminator training
            _, D_step, D_loss = sess.run([D_train_op, D_global_step, gan.D_loss],
                                          feed_dict={gan.x: x, gan.z: generate_noise_sample(batch_size, 100), gan.c:y, gan.keep_prob:0.5})

            # Generator training
            _, G_step, G_loss = sess.run([G_train_op, G_global_step, gan.G_loss],
                                          feed_dict={gan.z: generate_noise_sample(batch_size, 100), gan.c:y, gan.keep_prob:0.5})


            # Generate images from noise, using the generator network.
            if i % 10000 == 0:
                print(i, G_loss, D_loss)

                # generate validation condition
                row = 10
                col = 10
                conditions = []
                for r in range(row):
                    for c in range(col):
                        cond = []
                        cond.extend([0]*num_labels)
                        cond[r] = 1

                        conditions.append(cond)

                # generate validation noise
                noises = generate_noise_sample(row*col, noise_dimension)
                fig = plt.figure()
                s = 0
                for j in range(row):
                    samples = sess.run(gan.G_sample, feed_dict={gan.z: noises[j*10: (j+1)*10], gan.c: conditions[j*10: (j+1)*10], gan.keep_prob:1.0})
                    for c in range(col):
                        ax = fig.add_subplot(10, 10, s+1)
                        ax.set_axis_off()
                        ax.imshow(np.reshape(samples[c], (28, 28)))
                        s += 1
                plt.savefig('samples/{}.png'.format(int(i)))
                plt.close(fig)