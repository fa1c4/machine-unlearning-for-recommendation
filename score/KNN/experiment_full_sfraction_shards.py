import pandas as pd
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from Cls_Fullretrain import Fullretrain
from Cls_Sfraction import S_fraction
from Cls_Knnbagging import Knnbagging


if __name__ == '__main__':

    shuffle = True
    batchsize = 500
    total_unlearning_num = 10
    unlearning_data_pointes = batchsize * total_unlearning_num

    # full retrain
    # fullretrain = Fullretrain(shuffle, batchsize, total_unlearning_num)
    # fullretrain.unlearning()
    res = pd.read_csv('../../results/fullretrain_res.csv')
    acc_fullretrain = res['accuracys']
    unlearningtimes_fullretrain = res['unlearning time']

    # 1/s fraction
    shards = 5
    # s_fraction = S_fraction(shards, shuffle, batchsize, total_unlearning_num)
    # s_fraction.unlearning()
    res = pd.read_csv('../../results/1_{}_fraction_retrain_res.csv'.format(shards))
    acc_5fraction = res['accuracys']
    unlearningtimes_5fraction = res['unlearning time']

    shards = 10
    # s_fraction = S_fraction(shards, shuffle, batchsize, total_unlearning_num)
    # s_fraction.unlearning()
    res = pd.read_csv('../../results/1_{}_fraction_retrain_res.csv'.format(shards))
    acc_10fraction = res['accuracys']
    unlearningtimes_10fraction = res['unlearning time']

    shards = 15
    # s_fraction = S_fraction(shards, shuffle, batchsize, total_unlearning_num)
    # s_fraction.unlearning()
    res = pd.read_csv('../../results/1_{}_fraction_retrain_res.csv'.format(shards))
    acc_15fraction = res['accuracys']
    unlearningtimes_15fraction = res['unlearning time']

    shards = 2
    # s_fraction = S_fraction(shards, shuffle, batchsize, total_unlearning_num)
    # s_fraction.unlearning()
    res = pd.read_csv('../../results/1_{}_fraction_retrain_res.csv'.format(shards))
    acc_2fraction = res['accuracys']
    unlearningtimes_2fraction = res['unlearning time']

    shards = 4
    # s_fraction = S_fraction(shards, shuffle, batchsize, total_unlearning_num)
    # s_fraction.unlearning()
    res = pd.read_csv('../../results/1_{}_fraction_retrain_res.csv'.format(shards))
    acc_4fraction = res['accuracys']
    unlearningtimes_4fraction = res['unlearning time']

    shards = 8
    # s_fraction = S_fraction(shards, shuffle, batchsize, total_unlearning_num)
    # s_fraction.unlearning()
    res = pd.read_csv('../../results/1_{}_fraction_retrain_res.csv'.format(shards))
    acc_8fraction = res['accuracys']
    unlearningtimes_8fraction = res['unlearning time']

    shards = 16
    # s_fraction = S_fraction(shards, shuffle, batchsize, total_unlearning_num)
    # s_fraction.unlearning()
    res = pd.read_csv('../../results/1_{}_fraction_retrain_res.csv'.format(shards))
    acc_16fraction = res['accuracys']
    unlearningtimes_16fraction = res['unlearning time']

    # bagging shards
    shards = 5
    # knnbagging = Knnbagging(shards, shuffle, batchsize, total_unlearning_num)
    # knnbagging.unlearning()
    res = pd.read_csv('../../results/shards{}_unlearning_res.csv'.format(shards))
    acc_5shards = res['accuracys']
    unlearningtimes_5shards = res['unlearning time']

    shards = 10
    # knnbagging = Knnbagging(shards, shuffle, batchsize, total_unlearning_num)
    # knnbagging.unlearning()
    res = pd.read_csv('../../results/shards{}_unlearning_res.csv'.format(shards))
    acc_10shards = res['accuracys']
    unlearningtimes_10shards = res['unlearning time']

    shards = 15
    # knnbagging = Knnbagging(shards, shuffle, batchsize, total_unlearning_num)
    # knnbagging.unlearning()
    res = pd.read_csv('../../results/shards{}_unlearning_res.csv'.format(shards))
    acc_15shards = res['accuracys']
    unlearningtimes_15shards = res['unlearning time']

    shards = 2
    # knnbagging = Knnbagging(shards, shuffle, batchsize, total_unlearning_num)
    # knnbagging.unlearning()
    res = pd.read_csv('../../results/shards{}_unlearning_res.csv'.format(shards))
    acc_2shards = res['accuracys']
    unlearningtimes_2shards = res['unlearning time']

    shards = 4
    # knnbagging = Knnbagging(shards, shuffle, batchsize, total_unlearning_num)
    # knnbagging.unlearning()
    res = pd.read_csv('../../results/shards{}_unlearning_res.csv'.format(shards))
    acc_4shards = res['accuracys']
    unlearningtimes_4shards = res['unlearning time']

    shards = 8
    # knnbagging = Knnbagging(shards, shuffle, batchsize, total_unlearning_num)
    # knnbagging.unlearning()
    res = pd.read_csv('../../results/shards{}_unlearning_res.csv'.format(shards))
    acc_8shards = res['accuracys']
    unlearningtimes_8shards = res['unlearning time']

    shards = 16
    # knnbagging = Knnbagging(shards, shuffle, batchsize, total_unlearning_num)
    # knnbagging.unlearning()
    res = pd.read_csv('../../results/shards{}_unlearning_res.csv'.format(shards))
    acc_16shards = res['accuracys']
    unlearningtimes_16shards = res['unlearning time']

    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    x = range(0, unlearning_data_pointes, batchsize)
    plt.title('accuracy - unlearning data points - unlearning time')
    ax.set_xlabel('unlearning data points')
    ax.set_ylabel('unlearning time')
    ax.set_zlabel('accuracy')

    plt.plot(x, unlearningtimes_fullretrain, acc_fullretrain, label='full retrain', linewidth=1, color='r', marker='o')
    plt.plot(x, unlearningtimes_2shards, acc_2shards, label='2 shards', linewidth=1, color='b', marker='x')
    plt.plot(x, unlearningtimes_4shards, acc_4shards, label='4 shards', linewidth=1, color='g', marker='x')
    plt.plot(x, unlearningtimes_8shards, acc_8shards, label='8 shards', linewidth=1, color='y', marker='x')
    plt.plot(x, unlearningtimes_16shards, acc_16shards, label='16 shards', linewidth=1, color='pink', marker='x')
    # plt.plot(x, unlearningtimes_2fraction, acc_2fraction, label='1/2 fraction', linewidth=1, color='grey', marker='*')
    # plt.plot(x, unlearningtimes_4fraction, acc_4fraction, label='1/4 fraction', linewidth=1, color='purple', marker='*')
    # plt.plot(x, unlearningtimes_8fraction, acc_8fraction, label='1/8 fraction', linewidth=1, color='orange', marker='*')
    # plt.plot(x, unlearningtimes_16fraction, acc_16fraction, label='1/16 fraction', linewidth=1, color='cyan', marker='*')
    #
    # plt.legend(bbox_to_anchor=(1, 0), loc=3, borderaxespad=0)
    # plt.show()

    # fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    # x = range(0, unlearning_data_pointes, batchsize)
    # plt.title('accuracy - unlearning data points - unlearning time')
    # ax.set_xlabel('unlearning data points')
    # ax.set_ylabel('unlearning time')
    # ax.set_zlabel('accuracy')

    # plt.plot(x, unlearningtimes_fullretrain, acc_fullretrain, label='full retrain', linewidth=1, color='r', marker='o')
    # plt.plot(x, unlearningtimes_5shards, acc_5shards, label='5 shards', linewidth=1, color='b', marker='x')
    # plt.plot(x, unlearningtimes_10shards, acc_10shards, label='10 shards', linewidth=1, color='g', marker='x')
    # plt.plot(x, unlearningtimes_15shards, acc_15shards, label='15 shards', linewidth=1, color='y', marker='x')
    # plt.plot(x, unlearningtimes_5fraction, acc_5fraction, label='1/5 fraction', linewidth=1, color='grey', marker='*')
    # plt.plot(x, unlearningtimes_10fraction, acc_10fraction, label='1/10 fraction', linewidth=1, color='purple',
    #          marker='*')
    # plt.plot(x, unlearningtimes_15fraction, acc_15fraction, label='1/15 fraction', linewidth=1, color='orange',
    #          marker='*')

    plt.legend(bbox_to_anchor=(1, 0), loc=3, borderaxespad=0)
    plt.show()


    # unlearningtimes_fullretrain = [random.randint(0, 10) for _ in range(10)]
    # acc_fullretrain = [random.randint(0, 10) for _ in range(10)]
    # unlearningtimes_5shards = [random.randint(0, 10) for _ in range(10)]
    # acc_5shards = [random.randint(0, 10) for _ in range(10)]
    # unlearningtimes_10shards = [random.randint(0, 10) for _ in range(10)]
    # acc_10shards = [random.randint(0, 10) for _ in range(10)]
    # unlearningtimes_15shards = [random.randint(0, 10) for _ in range(10)]
    # acc_15shards = [random.randint(0, 10) for _ in range(10)]
    # unlearningtimes_5fraction = [random.randint(0, 10) for _ in range(10)]
    # acc_5fraction = [random.randint(0, 10) for _ in range(10)]
    # unlearningtimes_10fraction = [random.randint(0, 10) for _ in range(10)]
    # acc_10fraction = [random.randint(0, 10) for _ in range(10)]
    # unlearningtimes_15fraction = [random.randint(0, 10) for _ in range(10)]
    # acc_15fraction = [random.randint(0, 10) for _ in range(10)]