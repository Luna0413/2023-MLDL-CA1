from datasets import iris_data, iris_two_feature
from logistic import inner
import numpy as np

# train 70%, test 30% iris dataset
x_train, x_test, y_train, y_test = iris_data()
train_x, test_x, train_y, test_y = iris_two_feature()


def perceptron(weight): # for 4 feature 3 label
    error = -1
    iteration = 0

    while(error != 0 and iteration < 100):
        error = 0
        for k in range(len(x_train)):
            x = x_train[k]
            y = y_train[k]

            val = [inner(weight[0], x), inner(weight[1], x), inner(weight[2], x)]
            e_y = val.index(max(val))

            if e_y != y:
                error += 1
                for i in range(len(x)):
                    weight[e_y][i+1] -= x[i]
                    weight[y][i+1] += x[i]

        iteration += 1

        print(str(iteration) + " epoch end")
        print(weight)

    return weight

def train_perceptron(threshold): # for 4 feature 3 label
    weight=np.array([[threshold[0], 0, 0, 0, 0],
                     [threshold[1], 0, 0, 0, 0],
                     [threshold[2], 0, 0, 0, 0]])
    
    return perceptron(weight)

def test_perceptron(weight): # for 4 feature 3 label
    error = 0

    for k in range(len(x_test)): # epoch
        x = x_test[k]
        y = y_test[k]

        val = [inner(weight[0], x), inner(weight[1], x), inner(weight[2], x)]
        e_y = val.index(max(val))
        if e_y != y:
            error += 1

    return 1 - error/len(y_test)

def test_perceptron_train(weight): # for 4 feature 3 label
    error = 0

    for k in range(len(x_train)): # epoch
        x = x_train[k]
        y = y_train[k]

        val = [inner(weight[0], x), inner(weight[1], x), inner(weight[2], x)]
        e_y = val.index(max(val))
        if e_y != y:
            error += 1

    return 1 - error/len(y_train)

def perceptron_binary(weight): # for 2 feature 2 label
    error = -1
    iteration = 0

    while(error != 0 and iteration < 200):
        error = 0
        for k in range(len(train_x)):
            # print(weight)
            x = train_x[k]
            if train_y[k] == 1:
                y = 1
            else:
                y = -1

            # print(x, y)

            if inner(weight, x) > 0:
                e_y = 1
            else:
                e_y = -1

            if e_y != y:
                error += 1
                for i in range(len(x)):
                    weight[i+1] -= e_y*x[i]


        iteration += 1

        print(str(iteration) + " epoch end")
        print(weight)

    return weight

def train_perceptron_binary(threshold): # for 2 feature 2 label
    weight=np.array([threshold, 0, 0])
    
    return perceptron_binary(weight)

def test_perceptron_binary(weight): # for 2 feature 2 label
    error = 0

    for k in range(len(test_x)): # epoch
        x = test_x[k]
        if test_y[k] == 1:
                y = 1
        else:
                y = -1

        if inner(weight, x) > 0:
                e_y = 1
        else:
                e_y = -1

        if e_y != y:
            error += 1

    return 1 - error/len(test_y)

def test_perceptron_binary_train(weight): # for 2 feature 2 label
    error = 0

    for k in range(len(train_x)): # epoch
        x = train_x[k]
        if train_y[k] == 1:
                y = 1
        else:
                y = -1

        if inner(weight, x) > 0:
                e_y = 1
        else:
                e_y = -1

        if e_y != y:
            error += 1

    return 1 - error/len(train_y)
