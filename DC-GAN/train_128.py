import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import scipy.misc

from model_128 import DCGAN
from utils import CelebData

# Hyper-parameters
epoch = 2000000
batch_size = 100
sample_size = 10

num_labels = 10
noise_dimension = 100
hidden_dimension = 128
image_dimension = 128*128*3

# Load MNIST Dataset
ds = CelebData(path="D:\\yeachan\\information\\dataset\\CelebA")

config = tf.ConfigProto(allow_soft_placement=True)
with tf.Session(config=config) as sess:
    with sess.as_default():

        # GAN object
        gan = DCGAN(noise_dimension=noise_dimension, image_dimension=image_dimension, batch_size=batch_size)

        # Define Training procedures of Generator
        G_global_step = tf.Variable(0, name="G_global_step", trainable=True)
        G_optimizer = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5, beta2=0.999)
        G_grads_and_vars = G_optimizer.compute_gradients(gan.G_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator'))
        G_train_op = G_optimizer.apply_gradients(G_grads_and_vars, global_step=G_global_step)

        # Define Training procedures of Discriminator
        D_global_step = tf.Variable(0, name="D_global_step", trainable=True)
        D_optimizer = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5, beta2=0.999)
        D_grads_and_vars = D_optimizer.compute_gradients(gan.D_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='call_generator/discriminator'))
        D_train_op = D_optimizer.apply_gradients(D_grads_and_vars, global_step=D_global_step)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def generate_noise_sample(m, n):
            return np.random.normal(size=[m, n])


        # Training step
        for i in range(epoch):
            print(i)
            total_batch = int(len(ds.imgs)/batch_size)
            for t in range(total_batch):
                x = ds.imgs[t*batch_size:(t+1)*batch_size]
                z = generate_noise_sample(batch_size, 100)

                # Discriminator training
                _, D_step, D_loss = sess.run([D_train_op, D_global_step, gan.D_loss],
                                              feed_dict={gan.x: x, gan.z: z})

                # Generator training
                _, G_step, G_loss = sess.run([G_train_op, G_global_step, gan.G_loss],
                                              feed_dict={gan.z: z})

            # Generate images from noise, using the generator network.
            print(i, G_loss, D_loss)
            noises = generate_noise_sample(3*3, 100)
            samples = sess.run(gan.G_sample, feed_dict={gan.z: noises})
            fig = plt.figure()
            s = 0
            for row in range(1, 4):
                for col in range(1, 4):
                    ax = fig.add_subplot(3, 3, s+1)
                    ax.set_axis_off()
                    ax.imshow(scipy.misc.toimage(samples[s]))
                    s += 1
            plt.savefig('samples/{}.png'.format(int(i*len(ds.imgs))))
            plt.close(fig)