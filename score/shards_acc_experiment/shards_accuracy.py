import pandas as pd
import time
import matplotlib.pyplot as plt
from bagging_KNNunlearning import KNNbase_Unlearning
from KnnPred import knnpred
import random
import os

class Knnbagging():

    def __init__(self, shards, shuffle, batchsize, total_unlearning_num):
        self.shards = shards
        self.shuffle = shuffle
        self.shuffled_ordered_str = 'shuffled' if shuffle else 'ordered'
        self.batchsize = batchsize
        self.sharding_batchsize = batchsize // shards
        self.total_unlearning_num = total_unlearning_num
        self.acc_num = batchsize * total_unlearning_num

    def bagging(self, models):
        Pred = knnpred(algo=models[0].alg, shuffle=self.shuffle)
        testlist = Pred.get_testlist()
        preds = [_ for _ in range(self.shards)]
        for s in range(self.shards):
            Pred = knnpred(algo=models[s].alg, shuffle=self.shuffle)  # transmit into class knnpred and output accuracy
            acc, preds[s] = Pred.calculate_accuracy()
            # print('accuracy of shard{} shuffled: {}'.format(s, acc))
        bagginglist = []
        half_cnt = self.shards // 2
        for i in range(len(testlist)):
            cnt = 0 # static 1s number
            for j in range(self.shards):
                if preds[j][i] == 1: cnt += 1
            if cnt >= half_cnt: bagginglist.append(1)
            else: bagginglist.append(0)

        hits = 0
        for i in range(len(testlist)):
            if bagginglist[i] == testlist[i]:
                hits += 1
        acc = hits / len(testlist)
        # print('bagging model accuracy: {}'.format(acc))
        return acc

    # sharding unlearning implementation
    # batch unlearning in one shard to save more time
    def unlearning(self):
        models = []
        for s in range(self.shards):
            models.append(KNNbase_Unlearning(shuffle=self.shuffle, shards=self.shards, sharding_idx=s))
            models[s].data_readin('../../data/shards_{}_{}/dataset_sharded{}.csv'.format(self.shards, self.shuffled_ordered_str, s))
            models[s].data_df.to_csv("../../data/shards_{}_{}/shard{}-unlearning0.csv".format(self.shards, self.shuffled_ordered_str, s),
                                   sep="\t", header=None, index=False)

        accuracys, unlearning_times, next_indexs = [], [], []
        time_start = time.time()
        for s in range(self.shards):
            next_indexs.append(models[s].recommendation_unlearning(0, self.sharding_batchsize))

        bagging_time = 0.0
        for i in range(self.total_unlearning_num):
            print("---- trainning rounds {} ----".format(i))
            elpased_time = time.time() - time_start - bagging_time
            unlearning_times.append(elpased_time)
            print('----- time elapsed {} s -----'.format(elpased_time))
            batch_flag = random.randint(0, self.shards - 1)
            next_indexs[batch_flag] = models[batch_flag].recommendation_unlearning(next_indexs[batch_flag], self.batchsize)
            bagging_start = time.time()
            acc_temp = self.bagging(models)
            accuracys.append(acc_temp)
            print('shards {}\'s accuracy: {}'.format(self.shards, acc_temp))
            bagging_time += time.time() - bagging_start

        for s in range(self.shards):
            last_file_path = os.path.expanduser("../../data/shards_{}_{}/shard{}-unlearning{}.csv".format(self.shards, self.shuffled_ordered_str, s, next_indexs[s]))
            os.remove(last_file_path)

        res = {'accuracys': accuracys, 'unlearning time': unlearning_times}
        res_data = pd.DataFrame(res)
        res_data.to_csv('../../results/shards{}_unlearning_res.csv'.format(self.shards))


if __name__ == '__main__':

    # shuffled
    shuffle = True
    shuffled_acc = []
    shards = 5
    test_knnbagging = Knnbagging(shards, shuffle, 50, 10)
    test_knnbagging.unlearning()
    res = pd.read_csv('../../results/shards{}_unlearning_res.csv'.format(test_knnbagging.shards))
    accuracys = res['accuracys']
    # shuffled_acc.append(accuracys[0])

    x = range(0, 500, 50)
    # plt.plot(x, shuffled_acc, label='shuffled', linewidth=1, color='y', marker='o')
    plt.plot(x, accuracys, label='acc', linewidth=1, color='g', marker='o')
    plt.xlabel('unlearning datapoints number')
    plt.ylabel('accuracy')
    plt.title('accuracy of 5 shards unlearning')
    plt.legend()
    plt.show()

    # for s in range(2, 18, 2):
    #     test_knnbagging = Knnbagging(s, shuffle, 50, 1)
    #     test_knnbagging.unlearning()
    #     res = pd.read_csv('../../results/shards{}_unlearning_res.csv'.format(test_knnbagging.shards))
    #     accuracys = res['accuracys']
    #     shuffled_acc.append(accuracys[0])
    #
    # # ordered
    # shuffle = False
    # ordered_acc = []
    # for s in range(2, 18, 2):
    #     test_knnbagging = Knnbagging(s, shuffle, 50, 1)
    #     test_knnbagging.unlearning()
    #     res = pd.read_csv('../../results/shards{}_unlearning_res.csv'.format(test_knnbagging.shards))
    #     accuracys = res['accuracys']
    #     ordered_acc.append(accuracys[0])
    #
    #
    # x = range(2, 18, 2)
    # plt.plot(x, shuffled_acc, label='shuffled', linewidth=1, color='y', marker='o')
    # plt.plot(x, ordered_acc, label='ordered', linewidth=1, color='g', marker='x')
    # plt.xlabel('shards')
    # plt.ylabel('accuracy')
    # plt.title('accuracy of dataset ordered/shuffled unlearning')
    # plt.legend()
    # plt.show()

