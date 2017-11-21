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


def norwegian_time_series_all_aasen():
    nn = [0.015614,0.0268241,0.0371039,0.0467627,0.0566584]
    per = [0.0166616697785,0.0295437381955,0.0422147778205,0.0541432498248,0.0669198292921]
    svr = [0.0176347231328,0.0304374112519,0.0425360860365,0.0537606034365,0.0653104991418]

    k14 = [0.0162918091859,0.0277749635449,0.0384906456659,0.0486985391246,0.0592049584812]
    k5 = [0.0176420104467,0.030185381289,0.0418498548196,0.0533686068244,0.0648820883534]
    k30 = [0.0158593889316,0.0271024119702,0.0376306029718,0.0473603160895,0.057452947102]

    rnn_simple = [0.0176336904287,0.0282336420251,0.0395729133751,0.0494449058468,0.0583766876792]
    lstm = [0.0255262803064,0.0343665593354,0.0462806256831,0.0581061246562,0.0706336012979]


    values = [1, 2, 3, 4, 5]
    plt.ylabel('Mean Squared Error')
    plt.xlabel('Prediction')

    plt.plot(values, k5, color='red', label='K5')
    plt.plot(values, k14, color='blue', label='K14')
    plt.plot(values, nn, color='yellow', label='NN')
    plt.plot(values, per, color='green', label='Per')
    plt.plot(values, k30, color='black', label='k30')
    plt.plot(values, rnn_simple, label='rnn_simple')
    plt.plot(values, lstm, color='cyan', label='lstm')
    plt.plot(values, svr, color='purple', label='svr')


    plt.title('Compare')
    plt.legend()
    plt.show()


def norwegian_time_series_all_Raggovidda():
    nn = [0.0180995,0.0274104,0.0420119,0.0507118,0.0601649]
    per = [0.0187659909234,0.0298562656185,0.0450754674824,0.0561400756569,0.0693630069015]
    k14 = [0.0185138208668,0.0294095273901,0.0423355415422,0.0523102748262,0.0628235541952]
    k5 = [0.0206022892895,0.0329989349885,0.0470744421019,0.0581387617386,0.0697775713606]
    k30 = [0.0179174487004,0.0282054000672,0.040773773449,0.0502449703872,0.0605966122087]
    rnn_simple = [0.0179169781295,0.0282373685315,0.0413144283817,0.0512363353834,0.0619029512078]
    lstm = [0.0299569390828,0.0383666087955,0.0486957026094,0.0605777152601,0.0702483113742]

    svr = []

    values = [1, 2, 3, 4, 5]
    plt.ylabel('Mean Squared Error')
    plt.xlabel('Prediction')

    plt.plot(values, k5, color='red', label='K5')
    plt.plot(values, k14, color='blue', label='K14')
    plt.plot(values, nn, color='yellow', label='NN')
    plt.plot(values, per, color='green', label='Per')
    plt.plot(values, k30, color='black', label='k30')
    plt.plot(values, rnn_simple, label='rnn_simple')
    plt.plot(values, lstm, color='cyan', label='lstm')
    #plt.plot(values, svr, color='purple', label='svr')


    plt.title('Compare')
    plt.legend()
    plt.show()


def norwegian_time_series_all_bessaker():
    nn = [0.0230361,0.0350711,0.0476716,0.059241,0.0642264]
    per = [0.0248538088135,0.0387700016339,0.0535038640175,0.0652755267272,0.0757174687063]
    k14 = [0.0237855993243,0.0359645420403,0.048227781452,0.0574092730725,0.0654647846258]
    k5 = [0.026219753839,0.0402692008365,0.0537665855481,0.0643775683211,0.0733777163545]
    k30 = [0.02296803321,0.0346057174893,0.046596847268,0.0552013968203,0.062866293122]
    rnn_simple = [0.0227137146576,0.0344287472003,0.0467064087269,0.0561929689005,0.0649657673822]
    lstm = [0.0312083961306,0.0381241212951,0.0504936855967,0.0650280420784,0.0743490079624]
    svr = [0.0227628488427,0.0347349128561,0.0475516251118,0.0583552741015,0.0687677181348]

    values = [1, 2, 3, 4, 5]
    plt.ylabel('Mean Squared Error')
    plt.xlabel('Prediction')

    plt.plot(values, k5, color='red', label='K5')
    plt.plot(values, k14, color='blue', label='K14')
    plt.plot(values, nn, color='yellow', label='NN')
    plt.plot(values, per, color='green', label='Per')
    plt.plot(values, k30, color='black', label='k30')
    plt.plot(values, rnn_simple, label='rnn_simple')
    plt.plot(values, lstm, color='cyan', label='lstm')
    plt.plot(values, svr, color='purple', label='svr')

    plt.title('Compare')
    plt.legend()
    plt.show()


#plot_knn_short()
#plot_short_time_series_all()
#long_time_series_all()
norwegian_time_series_all_aasen()
#norwegian_time_series_all_Raggovidda()
#norwegian_time_series_all_bessaker()


