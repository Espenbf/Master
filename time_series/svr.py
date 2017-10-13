import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

def predict_stuff(input_values, output_values, test_input, test_output):


    max_value = np.amax(input_values)
    output_values = np.reshape(output_values, len(output_values))
    output_values = output_values / max_value
    input_values = input_values / max_value
    #print (input_values)
    #print (output_values)



    svr_rbf = SVR(kernel= 'rbf', C=1e3)
    #svr_rbf.fit(input_values, output_values)

    #svr_poly = SVR(kernel='poly', C=1e3)
    #svr_poly.fit(input_values, output_values)

    svr_lin = SVR(kernel='linear', C=1e3)
    svr_lin.fit(input_values, output_values)

    test_input = test_input / max_value
    test_output = test_output / max_value

    #prediction1 = svr_rbf.predict(test_input)
    #prediction2 = svr_poly.predict(test_input)
    prediction3 = svr_lin.predict(test_input)
    
    #result_rbf = mean_squared_error(prediction1, test_output, multioutput='raw_values')
    #result2_rbf = max_value*result_rbf
    
    #result_poly = mean_squared_error(prediction2, test_output, multioutput='raw_values')
    #result2_poly = max_value*result_poly

    result_lin = mean_squared_error(prediction3, test_output, multioutput='raw_values')
    result2_lin = max_value * result_lin

    #print(result2_rbf)
    #print(result2_poly)
    print(result2_lin)
    return result2_lin

