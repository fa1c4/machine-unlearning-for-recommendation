import pandas as pd
import time
import matplotlib.pyplot as plt
from Kbatch_KNNunlearning import KNNbase_Unlearning
from KnnPred import knnpred
import os


class S_fraction():

    def __init__(self, shards, shuffle, batchsize, total_unlearning_num):
        self.shards = shards
        self.shuffle = shuffle
        self.shuffled_ordered_str = 'shuffled' if shuffle else 'ordered'
        self.batchsize = batchsize
        self.total_unlearning_num = total_unlearning_num
        self.acc_num = batchsize * total_unlearning_num

    def unlearning(self):
        KNN_Unl = KNNbase_Unlearning(shuffle=self.shuffle, shards=self.shards)
        KNN_Unl.data_readin('../../data/shards_{}_{}/dataset_sharded0.csv'.format(self.shards, self.shuffled_ordered_str))
        KNN_Unl.data_df.to_csv("../../data/shards_{}_{}/u-unlearning0.csv".format(self.shards, self.shuffled_ordered_str),
                               sep="\t", header=None, index=False)
        accuracys = []
        unlearning_times = []
        time_start = time.time()
        next_index = KNN_Unl.recommendation_unlearning(0, self.batchsize)
        for i in range(self.total_unlearning_num):
            print("---- trainning rounds {} ----".format(i))
            elapsed_time = time.time() - time_start
            unlearning_times.append(elapsed_time)
            print('----- time elapsed {} s -----'.format(elapsed_time))
            # test_time = time.time()
            next_index = KNN_Unl.recommendation_unlearning(next_index, self.batchsize)
            # print('unlearning step time: {}'.format(time.time() - test_time))
            Pred = knnpred(KNN_Unl.alg)  # transmit into class knnpred and output accuracy
            acc_temp, _ = Pred.calculate_accuracy()
            accuracys.append(acc_temp)
            print('accuracy: {}'.format(acc_temp))

        last_file_path = os.path.expanduser('../../data/shards_{}_{}/u-unlearning{}.csv'.format(self.shards, self.shuffled_ordered_str, next_index))
        os.remove(last_file_path)
        res = {'accuracys': accuracys, 'unlearning time': unlearning_times}
        res_data = pd.DataFrame(res)
        res_data.to_csv('../../results/1_{}_fraction_retrain_res.csv'.format(self.shards))


if __name__ == '__main__':
    test_s_fraction = S_fraction(10, True, 50, 3)
    test_s_fraction.unlearning()
    res = pd.read_csv('../../results/1_{}_fraction_retrain_res.csv'.format(test_s_fraction.shards))
    accuracys = res['accuracys']
    unlearning_times = res['unlearning time']

    x = range(0, test_s_fraction.acc_num, test_s_fraction.batchsize)
    plt.plot(x, accuracys, label='acc', linewidth=1, color='g', marker='o')
    plt.xlabel('unlearning data points')
    plt.ylabel('accuracy')
    plt.title('accuracy of 1 / {} fraction unlearning'.format(test_s_fraction.shards))
    plt.legend()
    plt.show()

