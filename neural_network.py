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
reading_from_file.read_from_file()
learning_rate = 0.0001
n_nodes_hl1 = 10
n_nodes_hl2 = 100
n_nodes_hl3 = 500

n_nodes_input = 2
n_nodes_output = 1
batch_size = 1000
hm_epochs = 10

x = tf.placeholder('float', [None, 2])
y = tf.placeholder('float')

def neural_network_model(data):

    #(input_data * weights) + biases

    #Maybe change activation function

    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_input, n_nodes_hl1], 0, 0.1)),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl1], 0, 0.1))}

    #hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2], 0, 0.1)),
    #                  'biases': tf.Variable(tf.random_normal([n_nodes_hl2], 0, 0.1))}

    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_output], 0, 0.1)),
                      'biases': tf.Variable(tf.random_normal([n_nodes_output], 0, 0.1))}

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    #l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    #l2 = tf.nn.relu(l2)

    output = tf.matmul(l1, output_layer['weights']) + output_layer['biases']

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

        correct = tf.reduce_mean(tf.squared_difference(prediction, y))
        accuracy = tf.reduce_mean(correct)

        print ('Batch Size: ', batch_size,' Mean Squared Deviation', correct.eval({x:reading_from_file.get_test_input(), y:reading_from_file.get_test_output()}))
        #print('Prediction', prediction.eval({x: [[2.359, 251.774]]}), "Expected 0.013", )

        #print ("Prediction of 251.774, 2.359: ", sess.run(neural_network_model, feed_dict={x: [251.774, 2.359]}))

batch_size_array = [1000]
for i in batch_size_array:
    global batch_size
    batch_size = i
    train_neural_network(x)
