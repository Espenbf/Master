'''
input > weight > hidden layer 1(activation function) > weights > output layer

compare output to predicted output > cost function (cross entropy)
optimazation function (optimizer)  > mimize cost (AdamOptimizer, SGD, AdaGrad...)

backpropagation

feed forwar + backprogaation = epoch 

'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import reading_from_file
from tensorflow.contrib.rnn import core_rnn_cell


chunk_size = 5
n_chunks = 9
rnn_size = 128


class NN:
    def __init__(self, nr_inp, nr_out):
        self.n_nodes_input = nr_inp
        self.n_nodes_output = nr_out
        self.x = tf.placeholder('float', [None, nr_inp])
        self.y = tf.placeholder('float', [None, nr_out])
        reading_from_file.read_from_file_time_series(self.n_nodes_input, self.n_nodes_output)
        self.batch_size = reading_from_file.get_test_data_size_time_series()

    learning_rate = 0.01
    n_nodes_hl1 = 5000
    n_nodes_hl2 = 5000
    n_nodes_hl3 = 500
    rnn_size = 128

    hm_epochs = 30

    def recurrent_network_model(self, x):

        layer = {'weights': tf.Variable(tf.random_normal([self.rnn_size, self.n_nodes_output], 0, 0.1)),
                          'biases': tf.Variable(tf.random_normal([self.n_nodes_output], 0, 0.1))}

        #x = tf.transpose(x, [1, 0, 2])
        #x = tf.reshape(x, [-1, chunk_size])
        #x = tf.split(0, n_chunks, x)

        lstm_cell = core_rnn_cell.BasicLSTMCell(self.rnn_size)
        outputs, states = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32)


        output = tf.matmul(outputs[-1], layer['weights']) + layer['biases']
        return output


    def train_neural_network(self, x):
        prediction = self.recurrent_network_model(x)

        #Maybe change?
        cost = tf.reduce_mean(tf.squared_difference(prediction, self.y))

        #   Learning rate = 0.001
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)

        #Cycles feed forward + backprop


        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for epoch in range (self.hm_epochs):
                epoch_loss = 0
                total_batch = int(reading_from_file.get_batch_size_time_series()/ self.batch_size)
                for _ in range(total_batch):
                    epoch_x, epoch_y = reading_from_file.get_next_element_time_series(self.batch_size)
                    _,c = sess.run([optimizer, cost], feed_dict={x: epoch_x, self.y: epoch_y})
                    epoch_loss += c
                reading_from_file.reset_counter()

                print ('Epoch', epoch, 'completed out of' , self.hm_epochs, 'average loss', epoch_loss/total_batch)
                #if (epoch_loss/total_batch) < 0.4:
                #    break

            print (self.n_nodes_output)
            correct = tf.reduce_mean(tf.squared_difference(prediction, self.y), 0)
            #correct = tf.metrics.mean(tf.transpose(tf.squared_difference(prediction, self.y)))

            print ("correct: " , correct)
            result = correct.eval({x: reading_from_file.get_test_input_time_series(), self.y: reading_from_file.get_test_output_time_series()})
            print(' Mean Squared Deviation: ', result)
        return result

    def run(self):
       result = self.train_neural_network(self.x)
       return result