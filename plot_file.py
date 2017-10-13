import matplotlib.pyplot as plt

def plot_knn_short():
    k1 = [0.0745251524118,0.268681145942,0.575561913234,0.945889435449,1.3634995989]
    k2 = [0.0604136763153,0.209032019741,0.440880100205,0.727152504257,1.03818705173]
    k3 = [0.057935082115,0.195630985602,0.405082035941,0.661969489952,0.945854556909]
    k4 = [0.0570067137047,0.188029884752,0.383823064022,0.625625446651,0.889275966963]
    k5 = [0.0567941118219,0.183257629786,0.369836493867,0.601694309561,0.855137985615]
    k6 = [0.057656260637,0.184566346391,0.370068320973,0.599192143419,0.848595016721]
    k7 = [0.057286059844,0.181980853881,0.364232709263,0.589651641796,0.83477753954]
    k8 = [0.057647257452,0.182088664142,0.362678391777,0.584659516602,0.826765040202]
    k9 = [0.057375397089,0.181140633562,0.360725494259,0.580108492614,0.820101669578]
    k10 = [0.0582889424784,0.182499022958,0.361431824717,0.579679302202,0.819273322013]
    k11 = [0.0587023979003,0.182861513712,0.361618915056,0.579815602276,0.818952174052]
    k12 = [0.059138383982,0.183309883891,0.360649508631,0.576468267122,0.813553958696]
    k13 = [0.0593563069791,0.182418431157,0.359174221537,0.574737549753,0.810528195714]
    k14 = [0.0597971045858,0.182559256901,0.358613953358,0.572920616931,0.807256320594]



    values = [1, 2, 3, 4, 5]
    plt.ylabel('Mean Squared Error')
    plt.xlabel('Prediction')
    plt.plot(values, k3, color='red', label='K3')
    plt.plot(values, k4, color='blue', label='K4')
    plt.plot(values, k5, color='green', label='K5')
    plt.plot(values, k6, color='tomato', label='K6')
    plt.plot(values, k7, color='teal', label='K7')
    plt.plot(values, k8, color='pink', label='K8')
    plt.plot(values, k9, color='yellow', label='K9')
    plt.plot(values, k14, color='peru', label='K14')

    plt.title('KNN-Regression')
    plt.legend()
    plt.show()




def plot_short_time_series_all():
    k7 = [0.057286059844,0.181980853881,0.364232709263,0.589651641796,0.83477753954]
    k14 = [0.0597971045858,0.182559256901,0.358613953358,0.572920616931,0.807256320594]
    nn = [0.0610407, 0.241432, 0.364084, 0.593817, 1.08541]
    per = [0.0806643891162,0.257594906098,0.48758173856,0.747954009324,1.02758717201]

    values = [1, 2, 3, 4, 5]
    plt.ylabel('Mean Squared Error')
    plt.xlabel('Prediction')

    plt.plot(values, k7, color='red', label='K7')
    plt.plot(values, k14, color='blue', label='K14')
    plt.plot(values, nn, color='yellow', label='NN')
    plt.plot(values, per, color='green', label='Per')
    plt.title('Compare')
    plt.legend()
    plt.show()


def long_time_series_all():
    nn = [0.061692, 0.212462, 0.405219, 0.632888, 0.871834]
    per = [0.0857597456474,0.28361303375,0.526635370564,0.798896314004,1.07945766954]
    k9 = [0.0547852590079, 0.195611670346, 0.386172365675, 0.616426796375, 0.864786746519]
    k14 = [0.0555893360662,0.195246657743,0.384464906002,0.609682036247,0.849917550943]

    values = [1, 2, 3, 4, 5]
    plt.ylabel('Mean Squared Error')
    plt.xlabel('Prediction')

    plt.plot(values, k9, color='red', label='K7')
    plt.plot(values, k14, color='blue', label='K14')
    plt.plot(values, nn, color='yellow', label='NN')
    plt.plot(values, per, color='green', label='Per')
    plt.title('Compare')
    plt.legend()
    plt.show()


def norwegian_time_series_all():
    nn = [10321.2, 17822.3, 24816.7, 31249.1, 38061.8]
    per = [10931.7215416,19383.6466301,27697.115728,35523.3862101,43906.0999986]
    k14 = [10711.4933709,18283.1805697,25373.8469133,32103.7323014,38998.2850586]
    k5 = [11881.3536744,20338.1043711,28100.4679799,35952.4499161,43657.2129868]

    values = [1, 2, 3, 4, 5]
    plt.ylabel('Mean Squared Error')
    plt.xlabel('Prediction')

    plt.plot(values, k5, color='red', label='K5')
    plt.plot(values, k14, color='blue', label='K14')
    plt.plot(values, nn, color='yellow', label='NN')
    plt.plot(values, per, color='green', label='Per')
    plt.title('Compare')
    plt.legend()
    plt.show()




#plot_knn_short()
#plot_short_time_series_all()
#long_time_series_all()
norwegian_time_series_all()


