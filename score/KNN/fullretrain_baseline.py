import pandas as pd
import time
import matplotlib.pyplot as plt
from Kbatch_KNNunlearning import KNNbase_Unlearning
# from KNNbaseline import KNNbase_Unlearning
from surprise import dump
from KnnPred import knnpred

if __name__ == "__main__":
    shards = 5
    shuffle = True
    shuffled_ordered_str = 'shuffled' if shuffle else 'ordered'
    batchsize = 50
    sharding_batchsize = batchsize // shards
    total_unlearning_num = 30
    acc_num = batchsize * total_unlearning_num

    KNN_Unl = KNNbase_Unlearning(shuffle=shuffle)
    KNN_Unl.data_readin('../../data/train_data_{}.csv'.format(shuffled_ordered_str))
    KNN_Unl.data_df.to_csv("../../data/u-unlearning0.csv", sep="\t", header=None, index=False)

    accuracys = []
    time_start = time.time()
    next_index = KNN_Unl.recommendation_unlearning(0, batchsize)
    for i in range(total_unlearning_num):
        print("---- trainning rounds {} ----".format(i))
        print('----- time elapsed {} s -----'.format(time.time() - time_start))
        next_index = KNN_Unl.recommendation_unlearning(next_index, batchsize)
        Pred = knnpred(KNN_Unl.alg) # transmit into class knnpred and output accuracy
        acc_temp, _ = Pred.calculate_accuracy()
        accuracys.append(acc_temp)

    x = range(0, acc_num, batchsize)
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
    res_data.to_csv('../../results/fullretrain_res.csv')

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
