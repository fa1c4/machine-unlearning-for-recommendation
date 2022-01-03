import pandas as pd
import time
import matplotlib.pyplot as plt


if __name__ == '__main__':
    res = pd.read_csv('../results/fullretrain_res.csv')
    print(res)
    accuracys = res['accuracy']
    unlearning_times = res['unlearning time']
    print(accuracys)
    print(type(accuracys))

    # x = range(0, test_cls_fullretrain.acc_num, test_cls_fullretrain.batchsize)
    # plt.plot(x, accuracys, label='acc', linewidth=1, color='r', marker='o')
    # # plt.plot(x, KNN_Unl.MAE, label='MAE', linewidth=1, color='b', marker='*')
    # plt.xlabel('unlearning data points')
    # plt.ylabel('accuracy')
    # plt.title('accuracy of full retrain unlearning')
    # plt.legend()
    # plt.show()