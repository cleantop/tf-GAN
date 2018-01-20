# DC-GAN
I implemented DCGAN (Deep Convolutional Generative Adversarial Network) using tensorflow framework. For confirming the results, I used MNIST datset and CelebA dataset. For MNIST dataset, I used 2 layers convolutional layers as these images has 28*28 size. In CelebA datset which is the celebrity face dataset, I used 4 layers convolutional layers since they have relatively large size images than previous one. 

## Dataset
1. MNIST dataset (http://yann.lecun.com/exdb/mnist/) 
2. CelebA dataset (http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

## Dependencies
1. Python==3.5
2. Tensorflow==1.0.0
3. Scipy==1.0.0
4. Matplotlib==2.1.1

## Usage

Before running the code, you first locate your dataset-information in the train codes.

1. For MNIST,
<pre><code>python ./train_28.py</code></pre>

2. For MNIST,
<pre><code>python ./train_128.py</code></pre>

## Results
1. MNIST
![Alt text](/img/mnist.png)

2. CelebA - with small training epochs
![Alt text](/img/celeba.png)

## Reference
1. Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks (https://arxiv.org/abs/1511.06434)