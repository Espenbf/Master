import csv

ws = []
deg = []
power = []

ws_te = []
deg_te = []
power_te = []

time_series_power_input_train = []
time_series_power_output_train = []

time_series_power_input_test = []
time_series_power_output_test = []

test_data_part = 10
counter_batch = 0


def read_from_file2(file_name):
    global ws, deg, power, ws_te, deg_te, power_te, test_data_part
    with open(file_name) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        next(spamreader)
        next(spamreader)
        next(spamreader)
        next(spamreader)

        counter = 0
        for row in spamreader:
            if counter != test_data_part-1:
                ws.append(float(row[7]))
                deg.append(float(row[6]))
                power.append(float(row[5]))
                counter += 1
            else:
                counter = 0
                ws_te.append(float(row[7]))
                deg_te.append(float(row[6]))
                power_te.append(float(row[5]))


def read_from_file_time_series2(file_name, window_size_input, window_size_output):
    global time_series_power_input_train, time_series_power_output_train, time_series_power_input_test, time_series_power_output_test, test_data_part
    with open(file_name) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        next(spamreader)
        next(spamreader)
        next(spamreader)
        next(spamreader)

        initial_counter = 0
        counter = 0

        previous_values = []
        for row in spamreader:
            if initial_counter < window_size_input+window_size_output:
                initial_counter += 1
                previous_values.append(float(row[5]))

            else:
                if counter != test_data_part-1:
                    counter += 1

                    time_series_power_input_train.append(previous_values[:window_size_input].copy())
                    time_series_power_output_train.append(previous_values[window_size_input:window_size_input+window_size_output].copy())

                else:
                    time_series_power_input_test.append(previous_values[:window_size_input].copy())
                    time_series_power_output_test.append(previous_values[window_size_input:window_size_input+window_size_output].copy())

                    counter = 0
                previous_values.pop(0)
                previous_values.append(float(row[5]))





def read_from_file_time_series3(file_name, window_size_input, window_size_output):
    global time_series_power_input_train, time_series_power_output_train, time_series_power_input_test, time_series_power_output_test, test_data_part
    with open(file_name) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        next(spamreader)
        next(spamreader)
        next(spamreader)
        next(spamreader)

        initial_counter = 0
        counter = 0

        previous_values = []
        for row in spamreader:
            if initial_counter < window_size_input+window_size_output:
                initial_counter += 1
                previous_values.append(float(row[3]))

            else:
                if counter != test_data_part-1:
                    counter += 1

                    time_series_power_input_train.append(previous_values[:window_size_input].copy())
                    time_series_power_output_train.append(previous_values[window_size_input:window_size_input+window_size_output].copy())

                else:
                    time_series_power_input_test.append(previous_values[:window_size_input].copy())
                    time_series_power_output_test.append(previous_values[window_size_input:window_size_input+window_size_output].copy())

                    counter = 0
                previous_values.pop(0)
                previous_values.append(float(row[3]))



def read_from_file():
    read_from_file2('test_files/61559-2007.csv')
    read_from_file2('test_files/61559-2008.csv')
    read_from_file2('test_files/61559-2009.csv')
    #read_from_file2('test_files/61559-2010.csv')


def read_from_file_time_series(window_size_input, window_size_output):
    read_from_file_time_series2('test_files/61559-2007.csv', window_size_input, window_size_output)
    #read_from_file_time_series2('test_files/61559-2008.csv', window_size_input, window_size_output)
    #read_from_file_time_series2('test_files/61559-2009.csv', window_size_input, window_size_output)
    #read_from_file_time_series2('test_files/61559-2010.csv', window_size_input, window_size_output)


def read_from_file_time_series_norwegian(window_size_input, window_size_output):
    read_from_file_time_series3('test_files/Aasen_II.csv', window_size_input, window_size_output)

ws2 = ws[::2].copy()
deg2 = deg[::2].copy()
power2 = power[::2].copy()


ws_te2 = ws_te[::4].copy()
deg_te2 = deg_te[::4].copy()
power_te2 = power_te[::4].copy()


def get_next_element(batch_size):
    global counter_batch
    x = []
    y = []
    for i in range(0, batch_size):
        x.append([ws[counter_batch+i], deg[counter_batch+i]])
        y.append(power[counter_batch+i])
    counter_batch += batch_size
    return x, y


def get_next_element_time_series(batch_size):
    global counter_batch
    x = []
    y = []
    for i in range(0, batch_size):
        x.append(time_series_power_input_train[counter_batch+i])
        y.append(time_series_power_output_train[counter_batch+i])
    counter_batch += batch_size
    return x, y


def get_next_element_time_series_rnn(batch_size):
    x = []
    for i in range(0, batch_size):
        pass


def reset_counter():
    global counter_batch
    counter_batch = 0


def get_batch_size():
    return len(ws)


def get_batch_size_time_series():
    return len(time_series_power_input_train)


def get_test_input():
    x = []
    for i in range(0, len(ws_te)):
        x.append([ws_te[i], deg_te[i]])
    return x


def get_test_output():
    return power_te


def get_test_input_time_series():
    return time_series_power_input_test


def get_test_output_time_series():
    return time_series_power_output_test


def get_train_input_time_series():
    return time_series_power_input_train


def get_train_output_time_series():
    return time_series_power_output_train


def get_test_data_size_time_series():
    return len(time_series_power_output_test)


def get_test_input2():
    return ws_te, deg_te



def get_full_training_data_set():
    return ws, deg, power

#ws_te = ws[3::4].copy()
#deg_te = deg[3::4].copy()
#power_te = power[3::4].copy()






