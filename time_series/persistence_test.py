#Testdata ([input, ouput], [input, ouput] .....)
import numpy
def persistence_test(test_input, test_output):
    deviation_list = []
    deviation_output = []
    len_input = len(test_input[0])
    len_output = len(test_output[0])
    for i in range(len_output):
        deviation_list.append([])

    for i in range(len(test_input)):
        expected_value = test_input[i][len_input-1]
        for j in range(len_output):
            deviation_list[j].append(numpy.square(expected_value - test_output[i][j]))
    for i in range(len(deviation_list)):
        deviation_output.append(numpy.mean(deviation_list[i]))

    return deviation_output