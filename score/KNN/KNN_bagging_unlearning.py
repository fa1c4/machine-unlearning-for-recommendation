import pandas as pd
import time
import matplotlib.pyplot as plt
from Kbatch_KNNunlearning import KNNbase_Unlearning
from surprise import dump
from KnnPred import knnpred

if __name__ == "__main__":

    shards = 5
    shuffle = False
    shuflled_ordered_str = 'shuffled' if shuffle else 'ordered'

    for s in range(shards):
        KNN_Unl = KNNbase_Unlearning(shuffle=False, shards=shards)
        KNN_Unl.data_readin('../../data/shards_{}_{}/dataset_sharded{}.csv'.format(shards, shuflled_ordered_str, s))
        algorithm = KNN_Unl.train_model()
        dump.dump("../../model/shards_{}_{}/sharded_trained_model{}.m".format(shards, shuflled_ordered_str, s), algo=algorithm)
        KNN_Unl.data_df.to_csv("../../data/shards_{}_{}/sharded{}-unlearning0.csv".format(shards, shuflled_ordered_str, s),
                               sep="\t", header=None, index=False)

    models = [_ for _ in range(shards)]
    for s in range(shards):
        _, models[s] = dump.load("../../model/shards_{}_{}/sharded_trained_model{}.m".format(shards, shuflled_ordered_str, s))

    Pred = knnpred(algo=models[0], shuffle=shuffle)
    testlist = Pred.get_testlist()
    preds = [_ for _ in range(shards)]
    for s in range(shards):
        Pred = knnpred(algo=models[s], shuffle=shuffle)  # transmit into class knnpred and output accuracy
        acc, preds[s] = Pred.calculate_accuracy()
        print('accuracy of shard{} shuffled: {}'.format(s, acc))

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
    print('bagging model accuracy: {}'.format(acc))

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
