#-*- coding: utf-8 -*-
#This is a sample to simulate a function y = theta1*x1 + theta2*x2

import random

def gradientDescentDemo():
    input_x = [
        [1, 4], [2, 5], [5, 1], [4, 2]
    ]
    y = [19, 26, 19, 20]
    theta = [1, 1]
    loss = 10
    step_size = 0.001
    eps = 0.001
    max_iters = 2000
    error = 0
    iter_count = 0

    while loss > eps and iter_count < max_iters :
        loss = 0
        # 这里更新权重的时候所有的样本点都用上了
        for i in range(len(input_x)):
            pred_y = theta[0]*input_x[i][0]+theta[1]*input_x[i][1]
            theta[0] = theta[0]-step_size *(pred_y-y[i])*input_x[i][0]
            theta[1] = theta[1]-step_size*(pred_y-y[i])*input_x[i][0]

        for i in range(len(input_x)):
            pred_y = theta[0]*input_x[i][0]+theta[1]*input_x[i][1]
            error = 0.5*(pred_y-y[i])**2
            loss = loss + error

        iter_count += 1
        if iter_count % 50 == 0:
            print('theta: ', theta)
            print('final loss: ', loss)
            print('iters: ', iter_count)

    print('theta: ', theta)
    print('final loss: ', loss)
    print('iters: ', iter_count)


def stochasticGradientDescentDemo():
    # 随机梯度下降
    # This is a sample to simulate a function y = theta1*x1 + theta2*x2
    input_x = [[1, 4], [2, 5], [5, 1], [4, 2]]
    y = [19, 26, 19, 20]
    theta = [1, 1]
    loss = 10
    step_size = 0.001
    eps = 0.0001
    max_iters = 10000
    error = 0
    iter_count = 0
    while (loss > eps and iter_count < max_iters): # 要么误差收敛(convergence)了，要么限定的迭代次数到了
        loss = 0
        # 每一次选取随机的一个点进行权重的更新
        i = random.randint(0, 3)
        pred_y = theta[0] * input_x[i][0] + theta[1] * input_x[i][1]
        theta[0] = theta[0] - step_size * (pred_y - y[i]) * input_x[i][0]
        theta[1] = theta[1] - step_size * (pred_y - y[i]) * input_x[i][1]
        for i in range(3):
            pred_y = theta[0] * input_x[i][0] + theta[1] * input_x[i][1]
            error = 0.5 * (pred_y - y[i]) ** 2
            loss = loss + error
        iter_count += 1
        print('iters_count', iter_count)

    print('theta: ', theta)
    print('final loss: ', loss)
    print('iters: ', iter_count)
    pass


def batchStochasticGradientDescentDemo():
    # This is a sample to simulate a function y = theta1*x1 + theta2*x2
    input_x = [[1, 4], [2, 5], [5, 1], [4, 2]]
    y = [19, 26, 19, 20]
    theta = [1, 1]
    loss = 10
    step_size = 0.001
    eps = 0.0001
    max_iters = 10000
    error = 0
    iter_count = 0
    while (loss > eps and iter_count < max_iters):
        loss = 0

        i = random.randint(0, 3)  # 注意这里，我这里批量每次选取的是2个样本点做更新，另一个点是随机点+1的相邻点
        j = (i + 1) % 4
        pred_y = theta[0] * input_x[i][0] + theta[1] * input_x[i][1]
        theta[0] = theta[0] - step_size * (pred_y - y[i]) * input_x[i][0]
        theta[1] = theta[1] - step_size * (pred_y - y[i]) * input_x[i][1]

        pred_y = theta[0] * input_x[j][0] + theta[1] * input_x[j][1]
        theta[0] = theta[0] - step_size * (pred_y - y[j]) * input_x[j][0]
        theta[1] = theta[1] - step_size * (pred_y - y[j]) * input_x[j][1]
        for i in range(3):
            pred_y = theta[0] * input_x[i][0] + theta[1] * input_x[i][1]
            error = 0.5 * (pred_y - y[i]) ** 2
            loss = loss + error
        iter_count += 1
        print('iters_count', iter_count)

    print('theta: ', theta)
    print('final loss: ', loss)
    print('iters: ', iter_count)
    pass

if __name__ == '__main__':
    # gradientDescentDemo()
    # stochasticGradientDescentDemo()
    batchStochasticGradientDescentDemo()
    pass