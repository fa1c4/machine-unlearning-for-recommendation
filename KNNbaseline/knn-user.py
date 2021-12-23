import os, io, collections
import pandas as pd
from surprise import Dataset, KNNBaseline, SVD, accuracy, Reader
from surprise.model_selection import cross_validate, train_test_split

# 协同过滤方法
# 载入movielens-100k数据集，一个经典的公开推荐系统数据集，有选项提示是否下载。
data = Dataset.load_builtin('ml-100k')

# 或载入本地数据集# 数据集路径 path to dataset file
file_path = os.path.expanduser('~/.surprise_data/ml-100k/ml-100k/u.data')
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

# 为用户推荐n部电影，基于用户的协同过滤算法，先获取10个相似度最高的用户，把这些用户评分高的电影加入推荐列表。
def get_similar_users_recommendations(uid, n=10):
    # 获取训练集，这里取数据集全部数据
    trainset = data.build_full_trainset()
    # 考虑基线评级的协同过滤算法
    algo = KNNBaseline(sim_option = user_based_sim_option)
    # 拟合训练集
    algo.fit(trainset)
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
    for i, j in enumerate(list(recommendations)):
        if i >= 10:
            break
        print(j)

get_similar_users_recommendations('1', 10)