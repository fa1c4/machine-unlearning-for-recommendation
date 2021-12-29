import pandas as pd
import time
import matplotlib.pyplot as plt
from KNNbaseline import KNNbase_Unlearning
from surprise import dump
from KnnPred import knnpred

if __name__ == "__main__":

    KNN_Unl = KNNbase_Unlearning(shuffle=False)
    KNN_Unl.data_readin('~/.surprise_data/ml-100k/ml-100k/u.data')
    # algorithm = KNN_Unl.train_model()
    # dump.dump("../model/fulltrained_model.m", algo=algorithm)
    # KNN_Unl.get_similar_users_recommendations(algorithm, '7', 10)
    KNN_Unl.data_df.to_csv("../../data/u-unlearning0.csv", sep="\t", header=None, index=False)

    accuracys = []
    time_start = time.time()
    next_index = KNN_Unl.recommendation_unlearning(0, True)
    for i in range(500):
        validate_flag = False
        if i % 10 == 9:
            validate_flag = True
        print("---- trainning rounds {} ----".format(i))
        print('----- time elapsed {} s -----'.format(time.time() - time_start))
        next_index = KNN_Unl.recommendation_unlearning(next_index, validate_flag)
        if i % 50 == 49:
            Pred = knnpred(KNN_Unl.alg) # transmit into class knnpred and output accuracy
            accuracys.append(Pred.calculate_accuracy())

    x = range(0, 500, 50)
    plt.plot(x, accuracys, label='acc', linewidth=1, color='r', marker='o')
    # markerfacecolor='blue',markersize=12)
    # plt.plot(x, KNN_Unl.MAE, label='MAE', linewidth=1, color='b', marker='*')
    plt.xlabel('unlearning data points')
    plt.ylabel('accuracy')
    plt.title('accuracy of full retrain unlearning')
    plt.legend()
    plt.show()

    res = {'accuracy': accuracys}
    res_data = pd.DataFrame(res)
    res_data.to_csv('../results/fullretrain_res.cvs')

'''

    res = {'RMSE': KNN_Unl.RMSE, 'MAE': KNN_Unl.MAE}
    res_data = pd.DataFrame(res)
    res_data.to_csv('../results/RMSE_MAE_res.cvs')

    x = range(0, 60, 10)
    plt.plot(x, KNN_Unl.RMSE, label='RMSE', linewidth=1, color='r', marker='o')
    # markerfacecolor='blue',markersize=12)
    plt.plot(x, KNN_Unl.MAE, label='MAE', linewidth=1, color='b', marker='*')
    plt.xlabel('unlearning data points')
    plt.ylabel('error')
    plt.title('RMSE and MAE\nfor machine unlearning')
    plt.legend()
    plt.show()
    
'''
