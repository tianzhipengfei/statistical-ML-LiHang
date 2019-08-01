# @File    : KNN.py
# @Time    : 2019-07-31 15:19
# @Author  : Tianzhi Li
# @Email   : tianzhipengfei@gmail.com
# @Software: PyCharm

import pandas as pd
import cv2
import numpy as np

from sklearn.neighbors import KDTree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class KNN:
    def __init__(self, k = 3):
        self.k = k

    def predict(self, train_data, train_label, test_data):
        tree = KDTree(train_data)
        res = []
        for point in test_data:
            _, indices = tree.query([point], k=self.k)
            vote_dic = dict()
            for voter_index in indices[0]:
                if train_label[voter_index] not in vote_dic:
                    vote_dic[train_label[voter_index]] = 1
                else:
                    vote_dic[train_label[voter_index]] += 1
            res.append((sorted(vote_dic.items(), key=lambda item: item[1]))[-1][0])
        return res


def get_hog_features(trainset):
    features = []
    hog = cv2.HOGDescriptor('../data/hog.xml')
    for img in trainset:
        img = np.reshape(img, (28, 28))
        cv_img = img.astype(np.uint8)
        hog_feature = hog.compute(cv_img)
        # hog_feature = np.transpose(hog_feature)
        features.append(hog_feature)
    features = np.array(features)
    features = np.reshape(features, (-1, 324))
    return features


if __name__ == '__main__':
    raw_data = pd.read_csv('../data/train.csv', header=0)
    data = raw_data.values

    imgs = data[0::, 1::]
    labels = data[::, 0]

    features = get_hog_features(imgs)

    # 选取 2/3 数据作为训练集， 1/3 数据作为测试集
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.33,
                                                                                random_state=23323)
    print(len(test_features))
    knn = KNN()
    test_predict = knn.predict(train_features, train_labels, test_features)
    score = accuracy_score(test_labels, test_predict)
    print ("The accruacy socre is ", score)