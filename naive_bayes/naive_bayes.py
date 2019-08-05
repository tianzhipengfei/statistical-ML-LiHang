# @File    : naive_bayes.py
# @Time    : 2019-08-04 23:01
# @Author  : Tianzhi Li
# @Email   : tianzhipengfei@gmail.com
# @Software: PyCharm

# Firstly, I use float as the probability, but with a number(same as features' num) of multiply,
# the probability will be 0 eventually. As a result, this machine cannot find a class the data
# belongs to. To fix that, use the number of case instead. However, this method may make the
# probability become bigger than the INT_MAX. I have no idea how to fix this problem. If you
# have a great idea, please tell me. Thanks a lot.


import numpy as np
import pandas as pd
import cv2
import random

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class Naive_bayes:
    def __init__(self, smoothing=1):
        self.smoothing = smoothing

    def train(self, X, Y):
        if len(X) == 0:
            print('no train data input')
            return
        length_of_features = len(X[0])
        self.data_num = len(X)
        self.priori_probability = np.zeros((10))
        self.conditional_probability = np.zeros((10, length_of_features, 2))
        for i in range(self.data_num):
            img = binaryzation(X[i])
            self.priori_probability[Y[i]] += 1
            for j in range(length_of_features):
                xij = img[j]
                self.conditional_probility[Y[i]][j][xij] += 1
        for i in range(10):
            for j in range(length_of_features):
                pix_0 = self.conditional_probility[i][j][0]
                pix_1 = self.conditional_probility[i][j][1]

                # 计算0，1像素点对应的条件概率
                probalility_0 = (float(pix_0) / float(pix_0 + pix_1)) * 100 + self.smoothing
                probalility_1 = (float(pix_1) / float(pix_0 + pix_1)) * 100 + self.smoothing

                self.conditional_probility[i][j][0] = probalility_0
                self.conditional_probility[i][j][1] = probalility_1

    def predict(self, X):
        if (len(self.priori_probability)) == 0:
            print('no label input')
            return []
        if len(X) == 0:
            print('no predict data input')
            return []
        feature_total_num = len(X[0])
        res = []
        for i in range(len(X)):
            predict_data = binaryzation(X[i])
            possible_label = -1
            max_possibility = 0
            for label in range(10):
                label_num = self.priori_probability[label]
                possibility = (int) (label_num + self.smoothing)
                for j in range(feature_total_num):
                    possibility *= (int) (self.conditional_probility[label][j][predict_data[j]])
                if possibility > max_possibility:
                    max_possibility = possibility
                    possible_label = label
            res.append(possible_label)
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


def binaryzation(img):
    cv_img = img.astype(np.uint8)
    res = cv2.threshold(cv_img, 50, 1, cv2.THRESH_BINARY_INV, cv_img)  # cv2.cv.CV_THRESH_BINARY_INV = 1 found in doc
    return cv_img


if __name__ == '__main__':
    raw_data = pd.read_csv('../data/train.csv', header=0)
    data = raw_data.values

    imgs = data[0::, 1::]
    labels = data[::, 0]

    # 选取 2/3 数据作为训练集， 1/3 数据作为测试集
    train_features, test_features, train_labels, test_labels = train_test_split(imgs, labels, test_size=0.33,
                                                                                random_state=23323)
    naive_bayes = Naive_bayes()
    naive_bayes.train(train_features, train_labels)
    test_predict = naive_bayes.predict(test_features)
    score = accuracy_score(test_labels, test_predict)
    print ("The accruacy socre is ", score)
