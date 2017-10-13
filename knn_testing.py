import csv
import math
import reading_from_file as rf


def euclidean_distance(a0, a1, b0, b1):
    result = math.sqrt(pow((a0 - b0),2) + pow((a1 - b1), 2))
    return result


def k_nn(k, training_data, test_data):
    ws_tra = training_data[0]
    deg_tra = training_data[1]
    pow_tra = training_data[2]

    ws_test = test_data[0]
    deg_test = test_data[1]
    pow_test = test_data[2]

    deviation_list = []

    for i in range(len(test_data[0])):
        k_nearest = []


        for j in range(len(training_data[0])):
            dist = euclidean_distance(ws_test[i], deg_test[i], ws_tra[j], deg_tra[j])
            if len(k_nearest) < k:
                k_nearest.append([dist, pow_tra[j]])
                k_nearest.sort(key=lambda x: x[0], reverse=True)
            else:
                if dist < k_nearest[0][0]:
                    k_nearest[0] = [dist, pow_tra[j]]

                    k_nearest.sort(key=lambda x: x[0], reverse=True)
        sum = 0.
        for y in range(0, k):
            sum += k_nearest[y][1]
        avrage = sum/k
        deviation = abs(pow_test[i] - avrage)
        deviation_list.append(deviation)

    avrage_deviation = 0.
    for i in range(len(deviation_list)):
        avrage_deviation += deviation_list[i]
    avrage_deviation = avrage_deviation / len(deviation_list)
    print ("K: " + str(k) + "Average deviation: " + str(avrage_deviation))




#for k in range(2, 7):
#    k_nn(k, [ws2, deg2, power2], [ws_te2, deg_te2, power_te2])

rf.read_from_file()
x, z, y = rf.get_full_training_data_set()
a = rf.get_test_input2()

x_te = a[0]
z_te = a[1]
y_te = rf.get_test_output()


for k in range(2, 7):
    k_nn(k, [x, z, y], [x_te, z_te, y_te])

