# @File    : Perceptron.py
# @Time    : 2019-07-30 10:56
# @Author  : Tianzhi Li
# @Email   : tianzhipengfei@gmail.com
# @Software: PyCharm

import numpy as np
import pandas as pd
import cv2
import random
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class Perceptron:
    def __init__(self, learning_step=0.01, max_iteration=1000, max_correction_ratio=0.95):
        self.max_iteration = max_iteration
        self.learning_step = learning_step
        self.max_correction_ratio = max_correction_ratio

    def normal_train(self, X, Y):
        if len(X) == 0:
            print('no train data input')
            return
        length_of_features = len(X[0])
        data_num = len(X)
        self.w = [0.0] * (length_of_features)
        self.b = 0
        iteration_time = 0
        correct_time = 0
        while iteration_time < self.max_iteration and correct_time < data_num * self.max_correction_ratio:
            index = random.randint(0, data_num - 1)
            x = X[index]
            y = Y[index]
            wx = sum([self.w[j] * x[j] for j in range(len(self.w))])
            if wx * y > 0:
                correct_time += 1
                continue
            else:
                correct_time = 0
                for i in range(len(self.w)):
                    self.w[i] += self.learning_step * y * x[i]
                self.b += y
                iteration_time += 1

    def dual_train(self, X, Y):
        if len(X) == 0:
            print('no train data input')
            return
        length_of_features = len(X[0])
        self.w = [0.0] * (length_of_features)
        data_num = len(X)
        w1 = [0.0] * (data_num + 1)
        print('before init gram matrix')
        gram_matrix = np.zeros((data_num, data_num))
        print('after init gram matrix')
        for i in range(data_num):
            print(i)
            for j in range(data_num):
                gram_matrix[i][j] = sum([X[i][k] * X[j][k] for k in range(length_of_features)])
        print('end calculate gram_matrix')
        iteration_time = 0
        correct_time = 0
        while iteration_time < self.max_iteration and correct_time < data_num * self.max_correction_ratio:
            index = random.randint(0, data_num - 1)
            y = Y[index]
            wx = sum([(w1[j] * Y[j] * gram_matrix[j][index] + w1[-1]) for j in range(len(w1) - 1)])
            if wx * y > 0:
                iteration_time += 1
                correct_time += 1
                continue
            else:
                correct_time = 0
                w1[index] += self.learning_step
                w1[-1] += y
                iteration_time += 1
        self.b = w1[-1]
        for i in range(length_of_features):
            for j in range(len(w1) - 1):
                self.w[i] += X[j][i] * w1[j]

    def predict(self, X):
        res = []
        for i in range(len(X)):
            x = X[i]
            ans = sum([self.w[j] * x[j] for j in range(len(self.w))]) + self.b
            res.append(int(ans > 0))
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
    raw_data = pd.read_csv('../data/train_binary.csv', header=0)
    data = raw_data.values
    imgs = data[0::, 1::]
    labels = data[::, 0]
    features = get_hog_features(imgs)

    # 选取 2/3 数据作为训练集， 1/3 数据作为测试集
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels,
                                                                                test_size=0.33, random_state=23323)
    perceptron = Perceptron()
    print('start train')
    perceptron.dual_train(train_features, train_labels)
    test_predict = perceptron.predict(test_features)
    score = accuracy_score(test_labels, test_predict)
    print("The accruacy socre is ", score)
