# @File    : KNN.py
# @Time    : 2019-07-31 15:19
# @Author  : Tianzhi Li
# @Email   : tianzhipengfei@gmail.com
# @Software: PyCharm

class KNN:
    def __init__(self, k):
        self.k = k

    def train(self, X, Y):
        if len(X) == 0:
            print('no train data input')
            return
        length_of_features = len(X[0])
        data_num = len(X)

    def constract_tree(self, node, node_list, depth, label):

