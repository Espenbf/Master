import reading_from_file
from time_series import persistence_test, neural_network, lstm_rnn, recurrent_neural_network_simple
from time_series import svr, knn_improved, lstm_keras
import csv



def test_persistnece_time():
    reading_from_file.read_from_file_time_series_norwegian(5, 5)
    reading_from_file.normalize_time_series()
    test_output = reading_from_file.get_test_output_time_series()
    test_input = reading_from_file.get_test_input_time_series()
    result = persistence_test.persistence_test(test_input, test_output)
    print("Result:" + str(result))
    with open('output_files/raggovidda_persistence_test_time_norwegian.csv', 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=';',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)

        spamwriter.writerow(result)


def test_nn_time():
    nn = neural_network.NN(5, 5)
    result = nn.run()

    with open('output_files/raggovidda_nn_time_norwegian.csv', 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=';',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(result)


def test_rnn_simple():
    rnn = recurrent_neural_network_simple.RNN(5, 5)
    result = rnn.run()
    with open('output_files/aasen_rnn_simple_norwegian.csv', 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=';',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(result)

def run_rnn_time():
    rnn = lstm_rnn.NN(5,5)
    result = rnn.run()


def run_svr():
    #reading_from_file.read_from_file_time_series(5, 5)
    reading_from_file.read_from_file_time_series_norwegian(5, 5)
    reading_from_file.normalize_time_series()

    test_input = reading_from_file.get_test_input_time_series()
    test_output = reading_from_file.get_test_output_time_series()
    train_input = reading_from_file.get_train_input_time_series()
    train_output = reading_from_file.get_train_output_time_series()
    result = svr.predict_stuff(train_input, train_output, test_input, test_output)

    with open('output_files/raggovidda_svr_norwegian.csv', 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=';',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(result)


def test_knn_time_new():
    #reading_from_file.read_from_file_time_series(5, 5)
    reading_from_file.read_from_file_time_series_norwegian(5, 5)
    reading_from_file.normalize_time_series()

    test_input = reading_from_file.get_test_input_time_series()
    test_output = reading_from_file.get_test_output_time_series()

    train_input = reading_from_file.get_train_input_time_series()
    train_output = reading_from_file.get_train_output_time_series()

    with open('output_files/raggovidda_knn_norwegian.csv', 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=';',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for k in range(1, 30):
            result = knn_improved.knn_new(train_input, train_output, test_input, test_output, k)
            print("Result:" + str(result) +  " For K = " + str(k))
            spamwriter.writerow(result)


def test_lstm_keras():
    result = lstm_keras.lstm_keras(5, 5)
    with open('output_files/reggovidda_lstm_.csv', 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=';',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(result)

#test_knn_time_new()
#test_persistnece_time()
#test_nn_time()
#run_svr()
test_rnn_simple()
#test_lstm_keras()