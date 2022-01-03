import pandas as pd
import time
import matplotlib.pyplot as plt
from Kbatch_KNNunlearning import KNNbase_Unlearning
from KnnPred import knnpred
import os


class Fullretrain():

    def __init__(self, shuffle, batchsize, total_unlearning_num):
        self.shuffle = shuffle
        self.shuffled_ordered_str = 'shuffled' if shuffle else 'ordered'
        self.batchsize = batchsize
        self.total_unlearning_num = total_unlearning_num
        self.acc_num = batchsize * total_unlearning_num

    def unlearning(self):
        KNN_Unl = KNNbase_Unlearning(shuffle=self.shuffle)
        KNN_Unl.data_readin('../../data/train_data_{}.csv'.format(self.shuffled_ordered_str))
        KNN_Unl.data_df.to_csv("../../data/u-unlearning0.csv", sep="\t", header=None, index=False)

        accuracys = []
        unlearning_times = []
        time_start = time.time()
        next_index = KNN_Unl.recommendation_unlearning(0, self.batchsize)
        for i in range(self.total_unlearning_num):
            print("---- full retraining rounds {} ----".format(i))
            elapsed_time = time.time() - time_start
            unlearning_times.append(elapsed_time)
            print('----- full retrain time elapsed {} s -----'.format(elapsed_time))
            next_index = KNN_Unl.recommendation_unlearning(next_index, self.batchsize)
            Pred = knnpred(KNN_Unl.alg)  # transmit into class knnpred and output accuracy
            acc_temp, _ = Pred.calculate_accuracy()
            accuracys.append(acc_temp)

        last_file_path = "../../data/u-unlearning{}.csv".format(next_index)
        os.remove(last_file_path)
        res = {'accuracys': accuracys, 'unlearning time': unlearning_times}
        res_data = pd.DataFrame(res)
        res_data.to_csv('../../results/fullretrain_res.csv')


if __name__ == '__main__':
    test_cls_fullretrain = Fullretrain(True, 50, 10)
    test_cls_fullretrain.unlearning()
    res = pd.read_csv('../../results/fullretrain_res.csv')
    accuracys = res['accuracys']
    unlearning_times = res['unlearning time']

    x = range(0, test_cls_fullretrain.acc_num, test_cls_fullretrain.batchsize)
    plt.plot(x, accuracys, label='acc', linewidth=1, color='r', marker='o')
    # plt.plot(x, KNN_Unl.MAE, label='MAE', linewidth=1, color='b', marker='*')
    plt.xlabel('unlearning data points')
    plt.ylabel('accuracy')
    plt.title('accuracy of full retrain unlearning')
    plt.legend()
    plt.show()