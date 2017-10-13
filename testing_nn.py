import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
batch_size = 100
x, y = mnist.train.next_batch(1)
print (x)
print (y)