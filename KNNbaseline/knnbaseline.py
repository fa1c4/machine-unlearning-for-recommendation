import os
from surprise import KNNBaseline
from surprise import Dataset
import time

# step 1 : train model
def TrainModel():
    data = Dataset.load_builtin('ml-100k')
    trainset = data.build_full_trainset()
    # use pearson_baseline to compute similarity
    sim_options = {'name' : 'pearson_baseline', 'user_based' : False}
    algo = KNNBaseline(sim_options=sim_options)
    # train
    algo.fit(trainset)
    return algo

# step 2 : get id_name and name_id
def Get_Dict():
    file_name = os.path.expanduser('~/.surprise_data/ml-100k/ml-100k/u.item')
    id_name = {}
    name_id = {}
    with open(file_name, 'r', encoding='ISO-8859-1') as f:
        for line in f:
            line = line.split('|')
            id_name[line[0]] = line[1]
            name_id[line[1]] = line[0]
    return id_name, name_id

# step 3 : recommend movies based on the model
def RecommendMovie(movieName, algo, id_name, name_id, recommendNum):
    # get movie's raw id
    raw_id = name_id[movieName]
    # translate raw_id to inner_id
    inner_id = algo.trainset.to_inner_iid(raw_id)
    # recommend movies
    recommendations = algo.get_neighbors(inner_id, recommendNum)
    # translate inner_id to raw_id
    raw_ids = [algo.trainset.to_raw_iid(inner_id) for inner_id in recommendations]
    # get movie name
    movies = [id_name[raw_id] for raw_id in raw_ids]
    for movie in movies:
        print(movie)

if __name__ == '__main__':
    id_name, name_id = Get_Dict()

    starttime = time.time()
    algo = TrainModel()
    endtime = time.time()
    print("training time: {}s".format(endtime - starttime))

    showMovies = RecommendMovie('Toy Story (1995)', algo, id_name, name_id, 10)