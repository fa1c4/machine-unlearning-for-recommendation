from surprise import Dataset, Reader


class knnpred():

    def __init__(self, algo, shuffle=False):
        self.algo = algo
        reader = Reader(line_format='user item rating', sep=',')
        if shuffle:
            data_validate = Dataset.load_from_file('../../data/scoretest_dataset/score_data_shuffled.csv', reader=reader)
        else:
            data_validate = Dataset.load_from_file('../../data/scoretest_dataset/score_data_ordered.csv', reader=reader)
        testdata = data_validate.build_full_trainset()
        self.testset = testdata.build_testset()

    def calculate_accuracy(self):
        predictions = self.algo.test(self.testset)
        hits = 0
        for pred in predictions:
            if pred[2] > 3.0 and pred[3] > 3.0:
                hits += 1
            elif pred[2] < 3.0 and pred[3] < 3.0:
                hits += 1
            else:
                continue

        accuracy = hits / len(predictions)
        print('prediction accuracy is: {}'.format(accuracy))
        return accuracy