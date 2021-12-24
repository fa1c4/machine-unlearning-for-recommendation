import os, io, collections
import pandas as pd
import time
from surprise import Dataset, KNNBaseline, SVD, accuracy, Reader
from surprise.model_selection import cross_validate, train_test_split
import random
import matplotlib.pyplot as plt

user_total_num = 943
RMSE = []
MAE = []


def data_readin(path_to_udata):
    # 协同过滤方法
    # 载入movielens-100k数据集，一个经典的公开推荐系统数据集，有选项提示是否下载。
    # data = Dataset.load_builtin('ml-100k')

    global data, data_df, item_df, item_dict, user_based_sim_option, item_based_sim_option

    # 或载入本地数据集# 数据集路径 path to dataset file
    file_path = os.path.expanduser(path_to_udata)
    # 使用Reader指定文本格式，参数line_format指定特征（列名），参数sep指定分隔符
    reader = Reader(line_format='user item rating timestamp', sep='\t')

    # 加载数据集
    data = Dataset.load_from_file(file_path, reader=reader)
    data_df = pd.read_csv(file_path, sep='\t', header=None, names=['user','item','rating','timestamp'])
    item_df = pd.read_csv(os.path.expanduser('~/.surprise_data/ml-100k/ml-100k/u.item'),
                          sep='|', encoding='ISO-8859-1', header=None, names=['mid','mtitle']+[x for x in range(22)])
    # 每列都转换为字符串类型
    data_df = data_df.astype(str)
    item_df = item_df.astype(str)
    # 电影id到电影标题的映射
    item_dict = { item_df.loc[x, 'mid']: item_df.loc[x, 'mtitle'] for x in range(len(item_df)) }

    user_based_sim_option = {'name': 'pearson_baseline', 'user_based': True}
    # item-based
    item_based_sim_option = {'name': 'pearson_baseline', 'user_based': False}

    return data_df


def getDictKey(Dict, Value):
    return [k for k, v in Dict.items() if v == Value]


def item_to_movie_name(item):
    return item_dict[item]


def movie_name_to_item(movie_name):
    movies_list = getDictKey(item_dict, movie_name)
    return movies_list[0]


def seconds_to_ctime(seconds):
    seconds = seconds if type(seconds) == "int" else int(seconds)
    return time.ctime(seconds)


# search for the rating history of user with uid
def getting_user_history(uid):
    uid = uid if type(uid) == "str" else str(uid)

    data_df_sorted = data_df.sort_values(by=["user", "timestamp"], ascending=[True, True])
    data_user = data_df_sorted.loc[data_df["user"] == uid]
    # print(data_user)
    # print(len(data_user))

    cnt = 0
    for idx, row in data_user.iterrows():
        row["item"] = item_to_movie_name(row["item"])
        row["timestamp"] = seconds_to_ctime(row["timestamp"])
        print("{}, {}, {}, {}".format(idx, row["item"], row["rating"], row["timestamp"]))
        cnt += 1

    return data_user


# unlearning request for movie name
def unlearning_request(history_index, movie_index):
    if not isinstance(history_index, int):
        print("input int type index the input type is {}".format(type(history_index)))
        return ValueError

    data_df_droped = data_df.drop([history_index], axis=0)
    data_df_droped.to_csv("../data/u-unlearning{}.csv".format(history_index), sep="\t", header=None, index=False)
    print("user requested to forget history of movie: {}".format(item_to_movie_name(movie_index)))
    print("---- forgetting request done! -----")


# train model
def train_model():
    # 获取训练集，这里取数据集全部数据
    trainset = data.build_full_trainset()
    # 考虑基线评级的协同过滤算法
    algo = KNNBaseline(sim_option = user_based_sim_option)

    # validatation_res = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
    # # print(validatation_res)
    # test_rmse = validatation_res['test_rmse']
    # test_mae = validatation_res['test_mae']

    # 拟合训练集
    algo.fit(trainset)

    return algo
    # return algo, test_rmse, test_mae

# 为用户推荐n部电影，基于用户的协同过滤算法，先获取10个相似度最高的用户，把这些用户评分高的电影加入推荐列表。
def get_similar_users_recommendations(algo, uid, n=10):
    # 将原始id转换为内部id
    inner_id = algo.trainset.to_inner_uid(uid)
    # 使用get_neighbors方法得到10个最相似的用户
    neighbors = algo.get_neighbors(inner_id, k=10)
    neighbors_uid = ( algo.trainset.to_raw_uid(x) for x in neighbors )
    recommendations = set()

    #把评分为5的电影加入推荐列表
    for user in neighbors_uid:
        if len(recommendations) > n:
            break
        item = data_df[data_df['user']==user]
        item = item[item['rating']=='5']['item']
        for i in item:
            recommendations.add(item_dict[i])
    print('\nrecommendations for user %s:' %uid)
    for cnt, movie_name in enumerate(list(recommendations)):
        if cnt >= 10:
            break
        print(movie_name)


def train_unlearning_model(validate_flag):
    trainset = data.build_full_trainset()
    # 考虑基线评级的协同过滤算法
    algo = KNNBaseline(sim_option = user_based_sim_option)

    if validate_flag:
        validatation_res = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
        # print(validatation_res)
        test_rmse = validatation_res['test_rmse']
        test_mae = validatation_res['test_mae']

        rmse_mean = sum(test_rmse) / len(test_rmse)
        mae_mean = sum(test_mae) / len(test_mae)
    # 拟合训练集
    algo.fit(trainset)

    if validate_flag:
        return algo, rmse_mean, mae_mean
    else:
        return algo


def recommendation_unlearning(forgetting_index, validate_flag):
    global data, data_df, item_df, item_dict, user_based_sim_option, item_based_sim_option

    path_to_u_data = "../data/u-unlearning{}.csv".format(forgetting_index)
    data_df = data_readin(path_to_u_data)
    print(data_df)
    # uid = input("input the user who wanna forget his/her history in recommendation system\n")
    uid = random.randint(0, user_total_num)
    print(uid)
    data_user = getting_user_history(uid)
    print(data_user)
    indexarr = data_user.index.values
    # print(indexarr)
    # history_index = input("input the movie ranking history index user wanna forget\n")
    history_index = random.choice(indexarr).item()
    print(history_index)
    # history_index = int(history_index)
    dest = data_user.loc[history_index]
    movie_name = dest["item"]
    movie_index = movie_name_to_item(movie_name)

    unlearning_request(history_index, movie_index)

    if validate_flag:
        alg, rmse, mae = train_unlearning_model(validate_flag)
        RMSE.append(rmse)
        MAE.append(mae)
    else:
        alg = train_unlearning_model(validate_flag)

    # get_similar_users_recommendations(alg, uid, 10)

    return history_index


# def time_backtrack():
#     validatation_res = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
#     print(validatation_res)


if __name__ == "__main__":
    data_df = data_readin('~/.surprise_data/ml-100k/ml-100k/u.data')
    # algorithm, rmse, mae = train_model()
    algorithm = train_model()
    get_similar_users_recommendations(algorithm, '7', 10)
    data_df.to_csv("../data/u-unlearning0.csv", sep="\t", header=None, index=False)

    time_start = time.time()
    RMSE, MAE = [], []
    next_index = recommendation_unlearning(0, True)
    for i in range(500):
        validate_flag = False
        if i % 10 == 9:
            validate_flag = True
        print("---- trainning rounds {} ----".format(i))
        print('----- time elapsed {} s -----'.format(time.time() - time_start))
        next_index = recommendation_unlearning(next_index, validate_flag)

    res = {'RMSE': RMSE, 'MAE': MAE}
    res_data = pd.DataFrame(res)
    res_data.to_csv('../results/RMSE_MAE_res.cvs')

    x = range(0, 510, 10)
    plt.plot(x, RMSE, label='RMSE', linewidth=1, color='r', marker='o')
    # markerfacecolor='blue',markersize=12)
    plt.plot(x, MAE, label='MAE', linewidth=1, color='b', marker='*')
    plt.xlabel('unlearning data points')
    plt.ylabel('error')
    plt.title('RMSE and MAE\nfor machine unlearning')
    plt.legend()
    plt.show()

