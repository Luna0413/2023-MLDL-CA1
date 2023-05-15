from datasets import iris_data, iris_two_feature
import numpy as np
import matplotlib.pyplot as plt
import random

# train 70%, test 30% iris dataset
x_train, x_test, y_train, y_test = iris_data()
train_x, test_x, train_y, test_y = iris_two_feature()

def inner(weight, x):
    sum = 0
    for i in range(len(weight)):
        if i == 0:
            sum += weight[i]
        else:
            sum += x[i-1]*weight[i]
    return sum

def sigmoid(weight, x):
    """
    print(weight)
    print(x)
    """
    return 1/(1+np.exp(-inner(weight, x)))

def loss(weight, x, y):
    return (sigmoid(weight, x)-y)

def gradient_ascent(weight, maxIter, eta, lable): # for 4 feature 3 label
    sum_g = np.zeros(5)
    """
    print(str(maxIter) + " epoch left") 
    print("weight")
    print(weight)
    #print(sum_g)
    """

    for k in range(len(x_train)): # epoch
        x = x_train[k]
        if lable == y_train[k]:
            y = 1
        else:
            y = 0

        """
        print(x, y, weight)
        print("--------------------------------------------------------------------------")
        """

        for i in range(len(sum_g)):
            if i == 0:
                sum_g[i] += eta*(1/len(x_train))*(y - (np.exp(inner(weight, x))/(1+np.exp(inner(weight, x)))))
            else:
                sum_g[i] += eta*(1/len(x_train))*x[i-1]*(y - (np.exp(inner(weight, x))/(1+np.exp(inner(weight, x)))))
            
        

        """
        print("training ")
        print(x, y)    
        """
    """
    print("sum_g")
    print(sum_g) 
    """
    weight += sum_g
    

    eta *= 0.9
    maxIter -= 1
    if maxIter == 0:
        return weight
    else:
        gradient_ascent(weight, maxIter, eta, lable)

def stochastic_gradient_ascent(weight, maxIter, eta, lable): # for 4 feature 3 label
    while(maxIter > 0):
        ran_g = [0, 0, 0, 0, 0]

        random_number = random.randrange(0,len(x_train))
        x = x_train[random_number]
        if lable == y_train[random_number]:
            y = 1
        else:
            y = 0

        for i in range(len(ran_g)):
            if i == 0:
                ran_g[i] += eta*(y - (np.exp(inner(weight, x))/(1+np.exp(inner(weight, x)))))
            else:
                ran_g[i] += eta*x[i-1]*(y - (np.exp(inner(weight, x))/(1+np.exp(inner(weight, x)))))

        weight += ran_g

        # eta *= 0.9
        maxIter -= 1
    
    return weight


def MCAP(weight, maxIter, eta, lable, ld): # for 4 feature 3 label
    sum_g = np.zeros(5)
    sq_w = 0
    for w in weight:
        sq_w += w*w


    for k in range(len(x_train)): # epoch
        x = x_train[k]
        if lable == y_train[k]:
            y = 1
        else:
            y = 0

        """
        print(x, y, weight)
        print("--------------------------------------------------------------------------")
        """

        for i in range(len(sum_g)):
            if i == 0:
                sum_g[i] += eta*(1/len(x_train))*(y - (np.exp(inner(weight, x))/(1+np.exp(inner(weight, x)))))
            else:
                sum_g[i] += eta*(1/len(x_train))*x[i-1]*(y - (np.exp(inner(weight, x))/(1+np.exp(inner(weight, x)))))
            
        

        """
        print("training ")
        print(x, y)    
        """
    """
    print("sum_g")
    print(sum_g) 
    """
    weight += (eta*(-ld*weight)) + sum_g
    

    eta *= 0.9
    maxIter -= 1
    if maxIter == 0:
        return weight
    else:
        gradient_ascent(weight, maxIter, eta, lable)

def train_gradient_ascent_3_label(maxIter, eta): # for 4 feature 3 label
    weight = np.ones((3,5))
    for i in range(len(weight)):
        gradient_ascent(weight[i], maxIter, eta, i)
    
    
    print("after training weight is ")
    print(weight)
    
    return weight

def train_stochastic_gradient_ascent_3_label(maxIter, eta): # for 4 feature 3 label
    weight = np.ones((3,5))
    for i in range(len(weight)):
        stochastic_gradient_ascent(weight[i], maxIter, eta, i)

    print("after training weight is ")
    print(weight)
    
    return weight

def train_MCAP_3_label(maxIter, eta, ld): # for 4 feature 3 label
    weight = np.ones((3,5))
    for i in range(len(weight)):
        MCAP(weight[i], maxIter, eta, i, ld)

    sq_w = [0, 0, 0]
    for i in range(len(weight)):
        for w in weight[i]:
            sq_w[i] += w*w
    print("after training, sum of weight is ")
    print(sq_w)
    
    return weight

def test_3_label(weight): # for 4 feature 3 label
    error = 0
    for k in range(len(x_test)): # epoch
        x = x_test[k]
        y = y_test[k]
        temp = [0, 0, 0]
        for i in range(len(weight)):
            temp[i] = sigmoid(weight[i], x)
        
        """
        print(x)
        print(temp)
        print("--------------------------------------------------------------------------")

        print("test value = "+str(temp.index(max(temp))))
        print("real value = "+str(y))
        print("--------------------------------------------------------------------------")
        """

        if temp.index(max(temp)) != y:
            error += 1

    return 1 - error/len(y_test)

def test_3_label_train(weight): # for 4 feature 3 label
    error = 0
    for k in range(len(x_train)): # epoch
        x = x_train[k]
        y = y_train[k]
        temp = [0, 0, 0]
        for i in range(len(weight)):
            temp[i] = sigmoid(weight[i], x)
        
        """
        print(x)
        print(temp)
        print("--------------------------------------------------------------------------")

        print("test value = "+str(temp.index(max(temp))))
        print("real value = "+str(y))
        print("--------------------------------------------------------------------------")
        """

        if temp.index(max(temp)) != y:
            error += 1

    return 1 - error/len(y_train)

def gradient_ascent_binary(weight, maxIter, eta): # for 2 feature 2 label
    while(maxIter > 0):
        sum_g = np.zeros(3)   
        for k in range(len(train_x)): # epoch
            x = train_x[k]
            y = train_y[k]

            for i in range(len(sum_g)):
                if i == 0:
                    sum_g[i] += eta*(1/len(train_x))*(y - (np.exp(inner(weight, x))/(1+np.exp(inner(weight, x)))))
                else:
                    sum_g[i] += eta*(1/len(train_x))*x[i-1]*(y - (np.exp(inner(weight, x))/(1+np.exp(inner(weight, x)))))
                
        # print(sum_g)
        # print(weight)        
        weight = weight + sum_g
        

        eta *= 0.9
        maxIter -= 1
    
    print(weight)
    return weight

def stochastic_gradient_ascent_binary(weight, maxIter, eta): # for 2 feature 2 label
    while(maxIter > 0):
        ran_g = [0, 0, 0]
        random_number = random.randrange(0,len(train_x))
        x = train_x[random_number]
        y= train_y[random_number]

        for i in range(len(ran_g)):
            if i == 0:
                ran_g[i] += eta*(y - (np.exp(inner(weight, x))/(1+np.exp(inner(weight, x)))))
            else:
                ran_g[i] += eta*x[i-1]*(y - (np.exp(inner(weight, x))/(1+np.exp(inner(weight, x)))))

        weight = weight + ran_g

        eta *= 0.9
        maxIter -= 1

    return weight

def MCAP_binary(weight, maxIter, eta, ld): # for 2 feature 2 label
    while(maxIter > 0):
        sum_g = np.zeros(3)
        sq_w = 0
        for w in weight:
            sq_w += w*w


        for k in range(len(train_x)): # epoch
            x = train_x[k]
            y = train_y[k]

            for i in range(len(sum_g)):
                if i == 0:
                    sum_g[i] += eta*(1/len(train_x))*(y - (np.exp(inner(weight, x))/(1+np.exp(inner(weight, x)))))
                else:
                    sum_g[i] += eta*(1/len(train_x))*x[i-1]*(y - (np.exp(inner(weight, x))/(1+np.exp(inner(weight, x)))))
                

        weight = weight + (eta*(-ld*weight)) + sum_g
        

        eta *= 0.9
        maxIter -= 1

    return weight

def train_gradient_ascent_binary(maxIter, eta): # for 2 feature 2 label
    weight = np.array([1, 1, 1])
    weight = gradient_ascent_binary(weight, maxIter, eta)
    
    print("after training weight is ")
    print(weight)
    
    return weight

def train_stochastic_gradient_ascent_binary(maxIter, eta): # for 2 feature 3 label
    weight = np.array([1, 1, 1])
    weight = stochastic_gradient_ascent_binary(weight, maxIter, eta)

    print("after training weight is ")
    print(weight)
    
    return weight

def train_MCAP_binary(maxIter, eta, ld): # for 2 feature 2 label
    weight = np.array([1, 1, 1])
    
    weight = MCAP_binary(weight, maxIter, eta, ld)

    """
    sq_w = 0
    for w in weight:
        sq_w += w*w
    print("after training, sum of weight is ")
    print(sq_w)
    """
    
    return weight

def test_binary(weight): # for 2 feature 2 label
    error = 0
    for k in range(len(test_x)): # epoch
        x = test_x[k]
        y = test_y[k]
        if sigmoid(weight, x) > 0.5:
            temp = 1
        else:
            temp = 0   

        if temp != y:
            error += 1

    return 1 - error/len(test_y)

def test_binary_train(weight): # for 2 feature 2 label
    error = 0
    for k in range(len(train_x)): # epoch
        x = train_x[k]
        y = train_y[k]
        if sigmoid(weight, x) > 0.5:
            temp = 1
        else:
            temp = 0   

        if temp != y:
            error += 1

    return 1 - error/len(train_y)
