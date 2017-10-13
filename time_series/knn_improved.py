import numpy as np
from sklearn import neighbors
from sklearn.metrics import mean_squared_error


# #############################################################################
# Fit regression model

def knn_new(input_data, output_data, test_inp, test_out, n):
    for i, weights in enumerate(['uniform', 'distance']):
        knn = neighbors.KNeighborsRegressor(n, weights=weights)
        y_= knn.fit(input_data, output_data).predict(test_inp)
        result = mean_squared_error(y_, test_out, multioutput='raw_values')
        print(result)
        return result


