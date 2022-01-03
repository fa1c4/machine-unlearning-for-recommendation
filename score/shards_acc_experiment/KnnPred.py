from surprise import Dataset, Reader


class knnpred():

    def __init__(self, algo, shuffle=False):
        self.algo = algo
        reader = Reader(line_format='user item rating', sep='\t')
        if shuffle:
            data_validate = Dataset.load_from_file('../../data/scoretest_dataset/score_data_shuffled.csv', reader=reader)
        else:
            data_validate = Dataset.load_from_file('../../data/scoretest_dataset/score_data_ordered.csv', reader=reader)
        testdata = data_validate.build_full_trainset()
        self.testset = testdata.build_testset()

    def get_testlist(self):
        predictions = self.algo.test(self.testset)
        testlist = []
        for i in range(len(predictions)):
            if predictions[i][2] > 3.0:
                testlist.append(1)
            else:
                testlist.append(0)

        return testlist

    def calculate_accuracy(self):
        predictions = self.algo.test(self.testset)
        hits = 0
        predlist = []
        for pred in predictions:
            if pred[3] > 3.0:
                predlist.append(1)
                if pred[2] > 3.0: hits += 1
            elif pred[3] < 3.0:
                predlist.append(0)
                if pred[2] < 3.0: hits += 1
            else:
                continue

        accuracy = hits / len(predictions)
        # print('prediction accuracy is: {}'.format(accuracy))
        return accuracy, predlist
