import pandas as pd
import time
import matplotlib.pyplot as plt
# from KNNbaseline import KNNbase_Unlearning
from surprise import dump
from Kbatch_KNNunlearning import KNNbase_Unlearning


if __name__ == "__main__":
    shuffle = True
    shuffled_ordered_str = 'shuffled' if shuffle else 'ordered'
    RMSE, MAE = [], []

    for s in range(2, 18, 2):
        shards = s
        KNN_Unl = KNNbase_Unlearning(shuffle=shuffle, shards=shards)
        KNN_Unl.data_readin('../data/shards_{}_{}/dataset_sharded0.csv'.format(shards, shuffled_ordered_str))
        _, rmse, mae = KNN_Unl.train_unlearning_model_crossvalidate()
        RMSE.append(rmse)
        MAE.append(mae)

    res = {'RMSE': RMSE, 'MAE': MAE}
    res_data = pd.DataFrame(res)
    res_data.to_csv('../results/shards_RMSE_MAE_res.csv')

    x = range(2, 18, 2)
    plt.plot(x, RMSE, label='RMSE', linewidth=1, color='r', marker='o')
    plt.plot(x, MAE, label='MAE', linewidth=1, color='b', marker='*')
    plt.xlabel('shards')
    plt.ylabel('error')
    plt.title('RMSE and MAE\n depends on shards')
    plt.legend()
    plt.show()

    # KNN_Unl = KNNbase_Unlearning(shuffle=False)
    # KNN_Unl.data_readin('~/.surprise_data/ml-100k/ml-100k/u.data')
    # algorithm = KNN_Unl.train_model()
    # dump.dump("../model/fulltrained_model.m", algo=algorithm)
    # KNN_Unl.get_similar_users_recommendations(algorithm, '7', 10)
    # KNN_Unl.data_df.to_csv("../data/u-unlearning0.csv", sep="\t", header=None, index=False)
    #
    #
    # time_start = time.time()
    # next_index = KNN_Unl.recommendation_unlearning(0, True)
    # for i in range(50):
    #     validate_flag = False
    #     if i % 10 == 9:
    #         validate_flag = True
    #     print("---- trainning rounds {} ----".format(i))
    #     print('----- time elapsed {} s -----'.format(time.time() - time_start))
    #     next_index = KNN_Unl.recommendation_unlearning(next_index, validate_flag)
    #
    # res = {'RMSE': KNN_Unl.RMSE, 'MAE': KNN_Unl.MAE}
    # res_data = pd.DataFrame(res)
    # res_data.to_csv('../results/RMSE_MAE_res.csv')
    #
    # x = range(0, 60, 10)
    # plt.plot(x, KNN_Unl.RMSE, label='RMSE', linewidth=1, color='r', marker='o')
    # # markerfacecolor='blue',markersize=12)
    # plt.plot(x, KNN_Unl.MAE, label='MAE', linewidth=1, color='b', marker='*')
    # plt.xlabel('unlearning data points')
    # plt.ylabel('error')
    # plt.title('RMSE and MAE\nfor machine unlearning')
    # plt.legend()
    # plt.show()

