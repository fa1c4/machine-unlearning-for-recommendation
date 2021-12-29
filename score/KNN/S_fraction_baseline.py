import pandas as pd
import time
import matplotlib.pyplot as plt
from KNNbaseline import KNNbase_Unlearning
from surprise import dump
from KnnPred import knnpred


if __name__ == "__main__":

    KNN_Unl = KNNbase_Unlearning(shuffle=False, shards=5)
    KNN_Unl.data_readin('../../data/shards_5_ordered/dataset_sharded0.csv')
    # algorithm = KNN_Unl.train_model()
    # dump.dump("../model/fulltrained_model.m", algo=algorithm)
    KNN_Unl.data_df.to_csv("../../data/shards_5_ordered/u-unlearning0.csv", sep="\t", header=None, index=False)

    accuracys = []
    time_start = time.time()
    next_index = KNN_Unl.recommendation_unlearning(0, True)
    for i in range(500):
        validate_flag = False
        print("---- trainning rounds {} ----".format(i))
        print('----- time elapsed {} s -----'.format(time.time() - time_start))
        next_index = KNN_Unl.recommendation_unlearning(next_index, validate_flag)
        if i % 50 == 49:
            Pred = knnpred(KNN_Unl.alg)  # transmit into class knnpred and output accuracy
            acc_temp = Pred.calculate_accuracy()
            accuracys.append(acc_temp)
            print(acc_temp)

    x = range(0, 500, 50)
    plt.plot(x, accuracys, label='acc', linewidth=1, color='b', marker='o')
    # markerfacecolor='blue',markersize=12)
    # plt.plot(x, KNN_Unl.MAE, label='MAE', linewidth=1, color='b', marker='*')
    plt.xlabel('unlearning data points')
    plt.ylabel('accuracy')
    plt.title('accuracy of 1 / S fraction unlearning')
    plt.legend()
    plt.show()

    res = {'accuracy': accuracys}
    res_data = pd.DataFrame(res)
    res_data.to_csv('../../results/S_fraction_retrain_res.csv')