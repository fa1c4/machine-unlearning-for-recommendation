import os
import pandas as pd
import time
from surprise import Dataset, KNNBaseline, Reader
from surprise.model_selection import cross_validate
import random


class KNNbase_Unlearning():

    def __init__(self, shuffle=False, shards=1, sharding_idx=0, remove_files_flag=True):
        if shards <= 0:
            print('shards number is invalid!')
            exit(0)

        self.RMSE, self.MAE = [], []
        self.reader, self.data, self.data_df, self.item_df, self.item_dict = None, None, None, None, None
        self.user_based_sim_option, self.item_based_sim_option, self.data_df_droped = None, None, None
        self.alg, self.user_ids, self.algo = None, None, None
        self.shuffle, self.shards, self.sharding_idx = shuffle, shards, sharding_idx
        self.remove_files_flag, self.original_flag = remove_files_flag, True
        self.file_path = None

    def data_readin(self, path_to_udata):
        self.file_path = os.path.expanduser(path_to_udata)
        # print(file_path)
        # 使用Reader指定文本格式，参数line_format指定特征（列名），参数sep指定分隔符
        self.reader = Reader(line_format='user item rating timestamp', sep='\t')
        # 加载数据集
        self.data = Dataset.load_from_file(self.file_path, reader=self.reader)
        self.data_df = pd.read_csv(self.file_path, sep='\t', header=None, names=['user', 'item', 'rating', 'timestamp'])

        if self.remove_files_flag and self.original_flag == False:
            os.remove(self.file_path)
        self.original_flag = False

        # sorted data dataframe as user id and timestamp
        if not self.shuffle: # self.shuffle == False
            self.data_df = self.data_df.sort_values(by=["user", "timestamp"], ascending=[True, True])

        self.user_ids = self.data_df['user']
        # print(self.user_ids)
        self.item_df = pd.read_csv(os.path.expanduser('~/.surprise_data/ml-100k/ml-100k/u.item'),
                                   sep='|', encoding='ISO-8859-1', header=None,
                                   names=['mid', 'mtitle'] + [x for x in range(22)])
        # 每列都转换为字符串类型
        self.data_df = self.data_df.astype(str)
        self.item_df = self.item_df.astype(str)
        # 电影id到电影标题的映射
        self.item_dict = {self.item_df.loc[x, 'mid']: self.item_df.loc[x, 'mtitle'] for x in range(len(self.item_df))}
        self.user_based_sim_option = {'name': 'pearson_baseline', 'user_based': True}
        # item-based
        self.item_based_sim_option = {'name': 'pearson_baseline', 'user_based': False}

    def getDictKey(self, Dict, Value):
        return [k for k, v in Dict.items() if v == Value]

    def item_to_movie_name(self, item):
        return self.item_dict[item]

    def movie_name_to_item(self, movie_name):
        movies_list = self.getDictKey(self.item_dict, movie_name)
        return movies_list[0]

    def seconds_to_ctime(self, seconds):
        seconds = seconds if isinstance(seconds, int) else int(seconds)
        return time.ctime(seconds)

    # search for the rating history of user with uid
    def getting_user_history(self, uid):
        uid = uid if isinstance(uid, str) else str(uid)
        # print(type(uid))
        data_user = self.data_df.loc[self.data_df["user"] == uid]

        for idx, row in data_user.iterrows():
            row["item"] = self.item_to_movie_name(row["item"])
            row["timestamp"] = self.seconds_to_ctime(row["timestamp"])
            # print("{}, {}, {}, {}".format(idx, row["item"], row["rating"], row["timestamp"]))

        return data_user

    # unlearning request for movie name
    def unlearning_request(self, history_index: list):
        if not isinstance(history_index, list):
            print("input int type index the input type is {}".format(type(history_index)))
            return ValueError

        self.data_df_droped = self.data_df.drop(history_index, axis=0)
        if self.shards > 1:
            if self.shuffle:  # shuffled
                self.data_df_droped.to_csv("../../data/shards_{}_shuffled/shard{}-unlearning{}.csv".format(self.shards, self.sharding_idx, history_index[-1]),
                                           sep="\t", header=None, index=False)
            else:  # ordered
                self.data_df_droped.to_csv("../../data/shards_{}_ordered/shard{}-unlearning{}.csv".format(self.shards, self.sharding_idx, history_index[-1]),
                                           sep="\t", header=None, index=False)
        else:
            self.data_df_droped.to_csv("../../data/u-unlearning{}.csv".format(history_index[-1]), sep="\t", header=None, index=False)
        # print("user requested to forget history of movie: {}".format(self.item_to_movie_name(movie_index)))
        if self.shards > 1:
            print("---- shard{} forgetting request done! -----".format(self.sharding_idx))
        else:
            print("---- forgetting request done! -----")

    # train model
    def train_model(self):
        trainset = self.data.build_full_trainset()
        algo = KNNBaseline(sim_option=self.user_based_sim_option)
        algo.fit(trainset)
        self.algo = algo
        return algo

    # 为用户推荐n部电影，基于用户的协同过滤算法，先获取10个相似度最高的用户，把这些用户评分高的电影加入推荐列表。
    def get_similar_users_recommendations(self, algo, uid, n=10):
        # 将原始id转换为内部id
        inner_id = algo.trainset.to_inner_uid(uid)
        # 使用get_neighbors方法得到10个最相似的用户
        neighbors = algo.get_neighbors(inner_id, k=10)
        neighbors_uid = (algo.trainset.to_raw_uid(x) for x in neighbors)
        recommendations = set()
        # 把评分为5的电影加入推荐列表
        for user in neighbors_uid:
            if len(recommendations) > n:
                break
            item = self.data_df[self.data_df['user'] == user]
            item = item[item['rating'] == '5']['item']
            for i in item:
                recommendations.add(self.item_dict[i])
        # print('\nrecommendations for user %s:' % uid)
        # for cnt, movie_name in enumerate(list(recommendations)):
        #     if cnt >= 10:
        #         break
        #     print(movie_name)

    def train_unlearning_model_crossvalidate(self):
        trainset = self.data.build_full_trainset()
        algo = KNNBaseline(sim_option=self.user_based_sim_option)

        validatation_res = cross_validate(algo, self.data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
        test_rmse = validatation_res['test_rmse']
        test_mae = validatation_res['test_mae']
        rmse_mean = sum(test_rmse) / len(test_rmse)
        mae_mean = sum(test_mae) / len(test_mae)

        algo.fit(trainset)

        return algo, rmse_mean, mae_mean

    def recommendation_unlearning(self, forgetting_index, batchsize):
        if self.shards == 1:
            path_to_u_data = "../../data/u-unlearning{}.csv".format(forgetting_index)
        else:
            if self.shuffle:  # shuffled
                path_to_u_data = "../../data/shards_{}_shuffled/shard{}-unlearning{}.csv".format(self.shards, self.sharding_idx, forgetting_index)
            else:
                path_to_u_data = "../../data/shards_{}_ordered/shard{}-unlearning{}.csv".format(self.shards, self.sharding_idx, forgetting_index)

        self.data_readin(path_to_u_data)

        uids = []
        for _ in range(batchsize):
            uid = random.choice(self.user_ids).item()
            uids.append(uid)

        history_index = []
        for _ in range(batchsize):
            data_user = self.getting_user_history(uids[_])
            indexarr = data_user.index.values
            history_index_temp = random.choice(indexarr).item()
            history_index.append(history_index_temp)

        self.unlearning_request(history_index)
        self.alg = self.train_model()
        return history_index[-1]


'''
        if validate_flag:
            self.alg, rmse, mae = self.train_unlearning_model_crossvalidate()
            self.RMSE.append(rmse)
            self.MAE.append(mae)
        else:
            self.alg = self.train_model()
'''