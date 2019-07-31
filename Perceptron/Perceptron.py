# @File    : Perceptron.py
# @Time    : 2019-07-30 10:56
# @Author  : Tianzhi Li
# @Email   : tianzhipengfei@gmail.com
# @Software: PyCharm

import numpy as np
import random
import time


class Perceptron:
    def __init__(self, learning_step = 0.01, max_iteration = 1000, max_correction_ratio = 0.95):
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
        Gram_matrix = [[0.0 for i in range(data_num)] for j in range(data_num)]
        for i in range(data_num):
            for j in range(data_num):
                Gram_matrix[i][j] = sum([X[i][k] * X[j][k] for k in range(length_of_features)])
        iteration_time = 0
        correct_time = 0
        while iteration_time < self.max_iteration and correct_time < data_num * self.max_correction_ratio:
            index = random.randint(0, data_num - 1)
            y = Y[index]
            wx = sum([(w1[j] * Y[j] * Gram_matrix[j][index] + w1[-1]) for j in range(len(w1) - 1)])
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


    def predict(self, x):
        res = sum([self.w[j] * x[j] for j in range(len(self.w))])
        lala = "w: "
        for i in range(len(self.w)):
            lala += str(self.w[i]) + " "
        return (res > 0) * 2 - 1


if __name__ == '__main__':
    X = [[3, 3],
        [4, 3],
        [1, 1]]
    Y = [1, 1, -1]
    X1 = [1, 1]
    perceptron1 = Perceptron(1, 7)
    perceptron2 = Perceptron(1, 7)
    perceptron1.dual_train(X, Y)
    print(perceptron1.predict(X1))
    perceptron2.normal_train(X, Y)
    print(perceptron2.predict(X1))

