'''
input > weight > hidden layer 1(activation function) > weights > output layer

compare output to predicted output > cost function (cross entropy)
optimazation function (optimizer)  > mimize cost (AdamOptimizer, SGD, AdaGrad...)

backpropagation

feed forwar + backprogaation = epoch 

'''


import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell_impl

import reading_from_file
reading_from_file.read_from_file()
learning_rate = 0.001


n_nodes_input = 2
n_nodes_output = 1
batch_size = 10
hm_epochs = 200
chunk_size = 20
n_chunks = 20
rnn_size = 200


x = tf.placeholder('float', [None, 2])
y = tf.placeholder('float')

def recurrent_neural_network_model(x):

    #(input_data * weights) + biases


    layer = {'weights': tf.Variable(tf.random_normal([], 0, 0.1)),
                      'biases': tf.Variable(tf.random_normal([rnn_size, 1]))}

    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, chunk_size])
    x = tf.split(0, n_chunks, x)

    #output = tf.matmul(???, layer['weights']) + layer['biases']

    return output


def train_neural_network(x):
    prediction = neural_network_model(x)

    #Maybe change?
    cost = tf.reduce_mean(tf.squared_difference(prediction, y))

    #   Learning rate = 0.001
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    #Cycles feed forward + backprop


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range (hm_epochs):
            epoch_loss = 0
            total_batch = int(reading_from_file.get_batch_size()/ batch_size)
            for _ in range(total_batch):
                epoch_x, epoch_y = reading_from_file.get_next_element(batch_size)
                _,c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            reading_from_file.reset_counter()

            print ('Epoch', epoch, 'completed out of' , hm_epochs, 'average loss', epoch_loss/total_batch)

        correct = tf.abs(tf.subtract(prediction, y))
        accuracy = tf.reduce_mean(correct)


        print ('Avrage deviation', accuracy.eval({x:reading_from_file.get_test_input(), y:reading_from_file.get_test_output()}))
        print('Prediction', prediction.eval({x: [[2.359, 251.774]]}), "Expected 0.013", )

        #print ("Prediction of 251.774, 2.359: ", sess.run(neural_network_model, feed_dict={x: [251.774, 2.359]}))

train_neural_network(x)
