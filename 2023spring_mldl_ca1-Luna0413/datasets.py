from sklearn.feature_extraction.text import CountVectorizer
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import os

def naive_dataset(ham_traing, spam_traing):
    corpus = []
    
    ham_files = os.listdir(ham_traing)
    spam_files = os.listdir(spam_traing)

    """
    print(ham_files)
    print(spam_files)
    """

    for file in ham_files:
        # print(os.path.join(ham_traing, file))
        with open(os.path.join(ham_traing, file), 'r') as f:
            corpus.append(f.read())

    for file in spam_files:
        # print(os.path.join(spam_traing, file))
        with open(os.path.join(spam_traing, file), 'r') as f:
            corpus.append(f.read())
    

    """
    print("--------------------------------------------------------------------------")
    print("print each sentense")
    print(corpus)
    print("--------------------------------------------------------------------------")
    """
    

    vectorizer = CountVectorizer()
    bag = vectorizer.fit_transform(corpus).toarray()
    words = vectorizer.get_feature_names_out()

    """
    print("word list")
    print(words)
    print(bag)
    print("--------------------------------------------------------------------------")
    print("ham array")
    print(bag[:len(ham_files)])
    print("--------------------------------------------------------------------------")
    print("spam array")
    print(bag[len(ham_files):])
    print("--------------------------------------------------------------------------")
    """

    return bag[:len(ham_files)], bag[len(ham_files):], words

def iris_data(): # for 4 feature 3 label
    iris = datasets.load_iris()
    x=iris.data
    y=iris.target

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=43)
    """
    print("--------------------------------------------------------------------------")
    print(x)
    print(y)
    print("--------------------------------------------------------------------------")
    """
    return x_train, x_test, y_train, y_test



def iris_two_feature(): # for 2 feature 2 label
    iris = datasets.load_iris()
    x=iris.data
    y=iris.target

    """
    for i in range(len(y)):
        if y[i] == 2:
            print(i)
            break
    print(y) # 100
    """


    x = x[:100]
    y = y[:100]

    new_x = np.empty((0, 2))
    for x_i in x:
        new_x = np.append(new_x, np.array([[x_i[0], x_i[1]]]), 0)

    """
    print(x)
    print(new_x)
    """
    
    # print(y)

    x_train, x_test, y_train, y_test = train_test_split(new_x, y, test_size=0.3, random_state=43)

    return x_train, x_test, y_train, y_test

# iris_data()
# iris_two_feature()