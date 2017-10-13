import numpy as np
import tensorflow as tf
import csv




sess = tf.Session()

W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b

init = tf.global_variables_initializer()
sess.run(init)


y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)


fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
sess.run(init) # reset values to incorrect defaults.

for i in range(1000):
  sess.run(train, {x:[2.359,2.319, 2.273, 2.231, 3.531, 3.54], y:[0.013000000000000001,0.011,0.009000000000000001,0.007, 0.20800000000000002, 0.212]})
print(sess.run(loss, {x:[2.359,2.319, 2.273, 2.231, 3.531, 3.54], y:[0.013000000000000001,0.011,0.009000000000000001,0.007, 0.20800000000000002, 0.212]}))

print(sess.run([W, b]))
