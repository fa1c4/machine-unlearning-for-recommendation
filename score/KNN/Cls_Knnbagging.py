import pandas as pd
import time
import matplotlib.pyplot as plt
from bagging_KNNunlearning import KNNbase_Unlearning
from KnnPred import knnpred


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
        for i in range(len(testlist)):
            cnt = 0 # static 1s number
            for j in range(5):
                if preds[j][i] == 1: cnt += 1
            if cnt >= 3: bagginglist.append(1)
            else: bagginglist.append(0)

        hits = 0
        for i in range(len(testlist)):
            if bagginglist[i] == testlist[i]:
                hits += 1
        acc = hits / len(testlist)
        # print('bagging model accuracy: {}'.format(acc))
        return acc

    # sharding unlearning implementation
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

        for i in range(self.total_unlearning_num):
            print("---- trainning rounds {} ----".format(i))
            elpased_time = time.time() - time_start
            unlearning_times.append(elpased_time)
            print('----- time elapsed {} s -----'.format(elpased_time))
            for s in range(self.shards):
                next_indexs[s] = models[s].recommendation_unlearning(next_indexs[s], self.sharding_batchsize)
            acc_temp = self.bagging(models)
            accuracys.append(acc_temp)
            print('shards {}\'s accuracy: {}'.format(self.shards, acc_temp))

        res = {'accuracys': accuracys, 'unlearning time': unlearning_times}
        res_data = pd.DataFrame(res)
        res_data.to_csv('../../results/shards{}_unlearning_res.csv'.format(self.shards))


if __name__ == '__main__':
    test_knnbagging = Knnbagging(5, True, 50, 5)
    test_knnbagging.unlearning()
    res = pd.read_csv('../../results/shards{}_unlearning_res.csv'.format(test_knnbagging.shards))
    accuracys = res['accuracys']
    unlearning_times = res['unlearning time']

    x = range(0, test_knnbagging.acc_num, test_knnbagging.batchsize)
    plt.plot(x, accuracys, label='acc', linewidth=1, color='y', marker='o')
    plt.xlabel('unlearning data points')
    plt.ylabel('accuracy')
    plt.title('accuracy of shards {} unlearning'.format(test_knnbagging.shards))
    plt.legend()
    plt.show()

