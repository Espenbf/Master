import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

def predict_stuff(input_values, output_values, test_input, test_output):

    columns = list(zip(*output_values))
    output_values = columns[0]

    output_values = np.reshape(output_values, len(output_values))



    svr_rbf = SVR(kernel= 'rbf', C=1e3)
    svr_rbf.fit(input_values, output_values)

    test_out_column = list(zip(*test_output))




    prediction = svr_rbf.predict(test_input)
    results = [prediction]
    results_mse = [mean_squared_error(prediction, test_out_column[0], multioutput='raw_values')[0]]
    test_input_temp = test_input


    for i in range(0, len(test_out_column) - 1):
        columns = list(zip(*test_input_temp))
        columns.append(prediction)
        columns.pop(0)

        test_input_temp = list(zip(*columns))
        prediction = svr_rbf.predict(test_input_temp)
        results.append(prediction)
        results_mse.append(mean_squared_error(prediction, test_out_column[i+1], multioutput='raw_values')[0])
    print (results_mse)

    return results_mse

