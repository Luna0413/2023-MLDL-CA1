import logistic as lg
import perceptron as pt
from datasets import iris_data, iris_two_feature
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron

def find_max_eta_ga(binary):
    eta = 0
    max_acc = 0
    weight = []
    eta_list = []
    acc_list = []
    print("start")
    for i in range(100):
        if binary == True:
            train_weight = lg.train_gradient_ascent_binary(100, 0.1*i)
            acc = lg.test_binary(train_weight)
            eta_list.append(0.1*i)
            acc_list.append(acc)
        else:
            train_weight = lg.train_gradient_ascent_3_label(100, 0.1*i)
            acc = lg.test_3_label(train_weight)
            eta_list.append(0.1*i)
            acc_list.append(acc)
        if max_acc < acc:
            max_acc = acc
            if binary == True:
                eta= 0.1*i
            else:
                eta = 0.1*i
            weight = train_weight

    print(eta, max_acc)
    print(weight)
    plt.plot(eta_list, acc_list)
    plt.xlabel('Value of eta')
    plt.ylabel('Accuracy')
    plt.show()

def find_max_ir_sga(binary):
    ir = 0
    max_acc = 0
    weight = []
    ir_list = []
    acc_list = []
    print("start")
    for i in range(500):
        if binary == True:
            train_weight = lg.train_stochastic_gradient_ascent_binary(20*i, 0.7)
            acc = lg.test_binary(train_weight)
            ir_list.append(20*i)
            acc_list.append(acc)
        else:
            train_weight = lg.train_stochastic_gradient_ascent_3_label(10*i, 4.5)
            acc = lg.test_3_label(train_weight)
            ir_list.append(10*i)
            acc_list.append(acc)
        if max_acc < acc:
            max_acc = acc
            if binary == True:
                ir = 20*i
            else:
                ir = 10*i
            weight = train_weight

    print(ir, max_acc)
    print(weight)
    plt.scatter(ir_list, acc_list)
    plt.xlabel('Value of iteration')
    plt.ylabel('Accuracy')
    plt.show()

def find_max_lambda_MCAP(binary):
    eta = 0
    max_acc = 0
    weight = []
    eta_list = []
    acc_list = []
    print("start")
    for i in range(100):
        if binary == True:
            train_weight = lg.train_MCAP_binary(100, 0.7, 0.05*i)
            acc = lg.test_binary(train_weight)
            eta_list.append(0.05*i)
            acc_list.append(acc)
        else:
            train_weight = lg.train_MCAP_3_label(100, 4.5, 0.05*i)
            acc = lg.test_3_label(train_weight)
            eta_list.append(0.05*i)
            acc_list.append(acc)
        if max_acc < acc:
            max_acc = acc
            if binary == True:
                eta = 0.05*i
            else:
                eta = 0.05*i
            weight = train_weight

    print(eta, max_acc)
    print(weight)
    plt.plot(eta_list, acc_list)
    plt.xlabel('Value of lambda')
    plt.ylabel('Accuracy')
    plt.show()

def find_threshold_perceptron():
    max_acc = -1
    max_threshold = [-1, -1, -1]
    weight = []
    threshold_list = [[],[],[]]
    acc_list = []
    for i in range(10):
        for j in range(10):
            for k in range(10):
                train_weight = pt.train_perceptron([-0.5 + 0.1*i, -0.5 + 0.1*j, -0.5 + 0.1*k])
                acc = pt.test_perceptron(train_weight)
                threshold_list[0].append(-0.5+0.1*i)
                threshold_list[1].append(-0.5+0.1*j)
                threshold_list[2].append(-0.5+0.1*k)
                acc_list.append(acc)
                if max_acc < acc:
                    max_acc = acc
                    max_threshold = [-0.5+0.1*i, -0.5+0.1*j, -0.5+0.1*k]
                    weight = train_weight
    
    print(max_threshold, max_acc)
    print(weight)
    # print(threshold_list)
    plt.subplot(1, 3, 1)
    plt.scatter(threshold_list[0], acc_list)
    plt.xlabel('Value of lambda')
    plt.ylabel('Accuracy')
    plt.subplot(1, 3, 2)
    plt.scatter(threshold_list[1], acc_list)
    plt.xlabel('Value of lambda')
    plt.ylabel('Accuracy')
    plt.subplot(1, 3, 3)
    plt.scatter(threshold_list[2], acc_list)
    plt.xlabel('Value of lambda')
    plt.ylabel('Accuracy')
    plt.show()
                    

def find_threshold_perceptron_binary():
    max_acc = -1
    max_threshold = -1
    weight = []
    threshold_list = []
    acc_list = []

    for i in range(100):
        train_weight = pt.train_perceptron_binary(-10+0.2*i)
        acc = pt.test_perceptron_binary(train_weight)
        threshold_list.append(-10+0.2*i)
        acc_list.append(acc)
        if max_acc < acc:
            max_acc = acc
            max_threshold = -10+0.2*i
            weight = train_weight

    print(max_threshold, max_acc)
    print(weight)
    plt.plot(threshold_list, acc_list)
    plt.xlabel('Value of lambda')
    plt.ylabel('Accuracy')
    plt.show()



def graph_test(weight):
    x_train, x_test, y_train, y_test = iris_two_feature()
    print(weight)
    
    fig, ax = plt.subplots()
    scatter = ax.scatter(x_test[:, 0], x_test[:, 1], c=y_test)
    legend = ax.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
    ax.add_artist(legend)

    x_1 = np.linspace(4, 8, 100)

    y_0 = -(weight[1]/weight[2])*x_1-(weight[0]/weight[2])
    plt.plot(x_1, y_0, color='purple')



    # 그래프 옵션 설정
    ax.set_xlabel('Feature '+ str(1))
    ax.set_ylabel('Feature '+ str(2))
    ax.set_title('2D Data with 2 Labels')

    # 그래프 출력
    plt.show()

def graph_train(weight):
    x_train, x_test, y_train, y_test = iris_two_feature()
    print(weight)
    
    fig, ax = plt.subplots()
    scatter = ax.scatter(x_train[:, 0], x_train[:, 1], c=y_train)
    legend = ax.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
    ax.add_artist(legend)

    x_1 = np.linspace(4, 8, 100)

    y_0 = -(weight[1]/weight[2])*x_1-(weight[0]/weight[2])
    plt.plot(x_1, y_0, color='purple')



    # 그래프 옵션 설정
    ax.set_xlabel('Feature '+ str(1))
    ax.set_ylabel('Feature '+ str(2))
    ax.set_title('2D Data with 2 Labels')

    # 그래프 출력
    plt.show()

def sklearn_result(binary):

    if binary == True:
        x_train, x_test, y_train, y_test = iris_two_feature()
        print("Binary version")
    else:
        x_train, x_test, y_train, y_test = iris_data()
        print("Multinomial version")

    model = LogisticRegression(penalty=None)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    error = 0
    for i in range(len(y_test)):
        if y_test[i] != y_pred[i]:
            error += 1
    print("scikit learn vanilar logistic gression accuracy with default solver is "+ str(1-(error/len(y_test))))

    model1 = LogisticRegression(penalty=None, solver='sag')
    model1.fit(x_train, y_train)
    y_pred = model1.predict(x_test)
    error = 0
    for i in range(len(y_test)):
        if y_test[i] != y_pred[i]:
            error += 1
    print("scikit learn vanilar logistic gression accuracy 'sga' solver is "+ str(1-(error/len(y_test))))

    

    model2 = LogisticRegression(penalty='l2')
    model2.fit(x_train, y_train)
    y_pred = model2.predict(x_test)
    error2 = 0
    for i in range(len(y_test)):
        if y_test[i] != y_pred[i]:
            error2 += 1
    print("scikit learn L2 regularized logistic gression accuracy is "+ str(1-(error2/len(y_test))))

    model3 = Perceptron()
    model3.fit(x_train, y_train)
    y_pred = model2.predict(x_test)
    error2 = 0
    for i in range(len(y_test)):
        if y_test[i] != y_pred[i]:
            error2 += 1
    print("scikit learn perceptron accuracy is "+ str(1-(error2/len(y_test))))


    #print(y_pred)

sklearn_result(0)
# find_max_eta_ga(0)
# find_max_ir_sga(0)
# find_max_lambda_MCAP(1)
# find_threshold_perceptron()
# find_threshold_perceptron_binary()
# w = lg.train_MCAP_3_label(100, 4.5, 1.5)
# print(lg.test_3_label_train(w))
# print(lg.test_3_label(w))
# w = pt.train_perceptron([-0.5, -0.5, -0.4])
# print(pt.test_perceptron_train(w))
# print(pt.test_perceptron(w))
# graph_train(w)
# graph_test(w)
# print(w)