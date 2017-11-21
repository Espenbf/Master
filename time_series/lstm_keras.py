from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from time_series import lstm
import time
import reading_from_file
import numpy as np
from sklearn.metrics import mean_squared_error

def lstm_keras(nr_input, nr_output):
    reading_from_file.read_from_file_time_series_norwegian(nr_input, nr_output)
    reading_from_file.normalize_time_series()

    train_inp = reading_from_file.get_train_input_time_series()
    train_out = reading_from_file.get_train_output_time_series()


    x_train, y_train, x_test, y_test = lstm.load_data('test_files/Raggovidda_2.csv', 5, True)
    #print("X:", x_test)
    #print("Y:", y_test)

    print(len(y_test))
    print(len(x_test))

    train_inp = np.asarray(train_inp)
    len_inp = len(train_inp)
    train_inp = train_inp.reshape(len_inp, 5, 1)


    columns = list(zip(*train_out))
    train_out = columns[0]
    train_out = np.reshape(train_out, len(train_out))



    test_inp = reading_from_file.get_test_input_time_series()
    test_out = reading_from_file.get_test_output_time_series()

    columns = list(zip(*test_out))
    test_out = columns[0]
    test_out = np.reshape(test_out, len(test_out))

    len_inp = len(test_inp)
    test_inp = test_inp.reshape(len_inp, 5, 1)

    #test_out = np.asarray(test_out)
    #test_out = test_out.reshape(len_inp, 5, 1)

    model = Sequential()

    model.add(LSTM(
        input_dim = 1,
        output_dim = 5,
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        100,
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(
        output_dim=1))
    model.add(Activation('linear'))

    start = time.time()
    model.compile(loss='mse', optimizer='rmsprop')
    print('compilation time: ', time.time() - start)

    model.fit(
        x_train,
        y_train,
        batch_size=2000,
        validation_split=0.05,
        epochs=5)



    predictions = lstm.predict_sequences_multiple(model, x_test, 5, 5)
    pred = np.asarray(predictions)
    y_val = np.asarray(y_test)
    y_val = y_val.reshape(len(pred), 5)

    #print ("Y_val: ",y_val)
    #print ("Pred: ", pred)

    print(len(y_val))
    print(len(pred))

    result = mean_squared_error(pred, y_val, multioutput='raw_values')

    print (result)
    return result
