import pandas as pd
import time
import matplotlib.pyplot as plt
from bagging_KNNunlearning import KNNbase_Unlearning
from surprise import dump
from KnnPred import knnpred


def bagging(models, shuffle, shards):
    Pred = knnpred(algo=models[0].alg, shuffle=shuffle)
    testlist = Pred.get_testlist()
    preds = [_ for _ in range(shards)]
    for s in range(shards):
        Pred = knnpred(algo=models[s].alg, shuffle=shuffle)  # transmit into class knnpred and output accuracy
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


if __name__ == "__main__":

    shards = 5
    shuffle = True
    shuffled_ordered_str = 'shuffled' if shuffle else 'ordered'
    batchsize = 50
    sharding_batchsize = batchsize // shards
    total_unlearning_num = 5
    acc_num = batchsize * total_unlearning_num

    # sharding unlearning implementation
    models = []
    for s in range(shards):
        models.append(KNNbase_Unlearning(shuffle=shuffle, shards=shards, sharding_idx=s))
        models[s].data_readin('../../data/shards_{}_{}/dataset_sharded{}.csv'.format(shards, shuffled_ordered_str, s))
        models[s].data_df.to_csv("../../data/shards_{}_{}/shard{}-unlearning0.csv".format(shards, shuffled_ordered_str, s),
                               sep="\t", header=None, index=False)

    accuracys, next_indexs = [], []
    time_start = time.time()
    for s in range(shards):
        next_indexs.append(models[s].recommendation_unlearning(0, sharding_batchsize))

    for i in range(total_unlearning_num):
        print("---- trainning rounds {} ----".format(i))
        print('----- time elapsed {} s -----'.format(time.time() - time_start))
        for s in range(shards):
            next_indexs[s] = models[s].recommendation_unlearning(next_indexs[s], sharding_batchsize)
        acc_temp = bagging(models, shuffle, shards)
        accuracys.append(acc_temp)
        print('shards {}\'s accuracy: {}'.format(shards, acc_temp))

    x = range(0, acc_num, batchsize)
    plt.plot(x, accuracys, label='acc', linewidth=1, color='b', marker='o')
    plt.xlabel('unlearning data points')
    plt.ylabel('accuracy')
    plt.title('accuracy of shards {} unlearning'.format(shards))
    plt.legend()
    plt.show()

    res = {'accuracy': accuracys}
    res_data = pd.DataFrame(res)
    res_data.to_csv('../../results/shards{}_unlearning_res.csv'.format(shards))



'''
    batchsize = 500
    accuracys = []
    time_start = time.time()
    next_index = KNN_Unl.recommendation_unlearning(0, True)
    for i in range(100):
        print("---- retrainning rounds {} ----".format(i))
        print('----- time elapsed {} s -----'.format(time.time() - time_start))
        next_index = KNN_Unl.recommendation_unlearning(next_index, batchsize)
        Pred = knnpred(KNN_Unl.alg)  # transmit into class knnpred and output accuracy
        accuracys.append(Pred.calculate_accuracy())

    x = range(0, 50000, 500)
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

'''
    for s in range(shards):
        KNN_Unl = KNNbase_Unlearning(shuffle=shuffle, shards=shards)
        KNN_Unl.data_readin('../../data/shards_{}_{}/dataset_sharded{}.csv'.format(shards, shuffled_ordered_str, s))
        algorithm = KNN_Unl.train_model()
        dump.dump("../../model/shards_{}_{}/sharded_trained_model{}.m".format(shards, shuffled_ordered_str, s), algo=algorithm)
        KNN_Unl.data_df.to_csv("../../data/shards_{}_{}/sharded{}-unlearning0.csv".format(shards, shuffled_ordered_str, s),
                               sep="\t", header=None, index=False)

    models = [_ for _ in range(shards)]
    for s in range(shards):
        _, models[s] = dump.load("../../model/shards_{}_{}/sharded_trained_model{}.m".format(shards, shuffled_ordered_str, s))

    acc0 = bagging(models, shuffle, shards)
'''
