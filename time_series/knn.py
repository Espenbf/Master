import math
import numpy

def euclidean_distance(train_inp, test_inp):

    result = 0
    for i in range(len(train_inp)):
        result += pow(train_inp[i]-test_inp[i], 2)
    return math.sqrt(result)


def k_nn(k, train_data_inp, test_data_inp, train_data_out, test_data_out):
    len_out = len(test_data_out[0])
    deviation_list = []
    for z in range(len_out):
        deviation_list.append([])

    for i in range(len(test_data_inp)):
        k_nearest = []

        for j in range(len(train_data_inp)):
            dist = euclidean_distance(train_data_inp[j], test_data_inp[i])

            if len(k_nearest) < k:
                k_nearest.append([dist, train_data_out[j]])
                k_nearest.sort(key=lambda x: x[0], reverse=True)
            else:
                if dist < k_nearest[0][0]:
                    k_nearest[0] = [dist, train_data_out[j]]
                    k_nearest.sort(key=lambda x: x[0], reverse=True)

        sub_sum = []
        for z in range(len_out):
            sub_sum.append(0)
            for y in range(0, k):
                sub_sum[z] += k_nearest[y][1][z]*(1/k)


        for z in range(len_out):

            deviation_list[z].append(abs(sub_sum[z] - test_data_out[i][z]))

    deviation_output = []

    for i in range(len(deviation_list)):
        deviation_output.append(numpy.mean(deviation_list[i]))


    #print ("K: " + str(k) + "Average deviation: " + str(deviation_output))
    return deviation_output



#for k in range(2, 7):
#    k_nn(k, [ws2, deg2, power2], [ws_te2, deg_te2, power_te2])


