import pandas as pd
import time
import matplotlib.pyplot as plt
# from KNNbaseline import KNNbase_Unlearning
from Kbatch_KNNunlearning import KNNbase_Unlearning
from surprise import dump
from KnnPred import knnpred


if __name__ == "__main__":

    shards = 5
    shuffle = True
    shuffled_ordered_str = 'shuffled' if shuffle else 'ordered'
    batchsize = 50
    total_unlearning_num = 5
    acc_num = batchsize * total_unlearning_num

    KNN_Unl = KNNbase_Unlearning(shuffle=shuffle, shards=shards)
    KNN_Unl.data_readin('../../data/shards_{}_{}/dataset_sharded0.csv'.format(shards, shuffled_ordered_str))
    KNN_Unl.data_df.to_csv("../../data/shards_{}_{}/u-unlearning0.csv".format(shards, shuffled_ordered_str),
                           sep="\t", header=None, index=False)

    accuracys = []
    time_start = time.time()
    next_index = KNN_Unl.recommendation_unlearning(0, batchsize)
    for i in range(total_unlearning_num):
        print("---- trainning rounds {} ----".format(i))
        print('----- time elapsed {} s -----'.format(time.time() - time_start))
        next_index = KNN_Unl.recommendation_unlearning(next_index, batchsize)

        Pred = knnpred(KNN_Unl.alg)  # transmit into class knnpred and output accuracy
        acc_temp, _ = Pred.calculate_accuracy()
        accuracys.append(acc_temp)
        print('accuracy: {}'.format(acc_temp))

    x = range(0, acc_num, batchsize)
    plt.plot(x, accuracys, label='acc', linewidth=1, color='b', marker='o')
    plt.xlabel('unlearning data points')
    plt.ylabel('accuracy')
    plt.title('accuracy of 1 / {} fraction unlearning'.format(shards))
    plt.legend()
    plt.show()

    res = {'accuracy': accuracys}
    res_data = pd.DataFrame(res)
    res_data.to_csv('../../results/S_fraction_retrain_res.csv')