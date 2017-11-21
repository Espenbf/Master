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


class NN:
    def __init__(self, nr_inp, nr_out):
        self.n_nodes_input = nr_inp
        self.n_nodes_output = nr_out
        self.x = tf.placeholder('float', [None, nr_inp])
        self.y = tf.placeholder('float', [None, nr_out])
        reading_from_file.read_from_file_time_series_norwegian(self.n_nodes_input, self.n_nodes_output)
        reading_from_file.normalize_time_series()

        self.batch_size = reading_from_file.get_test_data_size_time_series()
        self.batch_size = 1000

    learning_rate = 0.01
    n_nodes_hl1 = 2000
    n_nodes_hl2 = 2000
    n_nodes_hl3 = 2000


    hm_epochs = 5




    def neural_network_model(self, data):

        hidden_1_layer = {'weights': tf.Variable(tf.random_normal([self.n_nodes_input, self.n_nodes_hl1], 0, 0.1)),
                          'biases': tf.Variable(tf.random_normal([self.n_nodes_hl1], 0, 0.1))}

        hidden_2_layer = {'weights': tf.Variable(tf.random_normal([self.n_nodes_hl1, self.n_nodes_hl2], 0, 0.1)),
                          'biases': tf.Variable(tf.random_normal([self.n_nodes_hl2], 0, 0.1))}

        hidden_3_layer = {'weights': tf.Variable(tf.random_normal([self.n_nodes_hl2, self.n_nodes_hl2], 0, 0.1)),
                          'biases': tf.Variable(tf.random_normal([self.n_nodes_hl2], 0, 0.1))}

        output_layer = {'weights': tf.Variable(tf.random_normal([self.n_nodes_hl1, self.n_nodes_output], 0, 0.1)),
                          'biases': tf.Variable(tf.random_normal([self.n_nodes_output], 0, 0.1))}

        l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
        l1 = tf.nn.relu(l1)

        l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
        l2 = tf.nn.relu(l2)

        #l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
        #l3 = tf.nn.relu(l3)

        output = tf.matmul(l2, output_layer['weights']) + output_layer['biases']

        return output

    def train_neural_network(self, x):
        prediction = self.neural_network_model(x)

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

            print (self.n_nodes_output)
            correct = tf.reduce_mean(tf.squared_difference(prediction, self.y), 0)
            #correct = tf.metrics.mean(tf.transpose(tf.squared_difference(prediction, self.y)))

            print ("correct: " , correct)
            result = correct.eval({x: reading_from_file.get_test_input_time_series(), self.y: reading_from_file.get_test_output_time_series()})
            print(' Mean Squared Deviation: ', result)
            #print ('Batch Size: ', batch_size,' Mean Squared Deviation', correct.eval({self.x:reading_from_file.get_test_input(), self.y:reading_from_file.get_test_output()}))
        return result

    def run(self):
       result = self.train_neural_network(self.x)
       return result