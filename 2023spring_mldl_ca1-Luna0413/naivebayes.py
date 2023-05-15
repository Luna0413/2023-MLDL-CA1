import os 
from datasets import naive_dataset
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# h(x)=argmax_y P(y)\mult_i=1^LengthDoc P(x_i|y)

# 6. Get result 


print("------------------------------")

# 1. Get datasets from datasets.py with directory myham, myspam

ham_bag, spam_bag, word_list = naive_dataset('data/myham', 'data/myspam')

print(word_list)
print(ham_bag)
print(spam_bag)
print("------------------------------")

# 2. Let P(ham) = P(spam) = 1/2 => doesn't use prior because prior is same

# 3. Traing P(x_i|ham) and P(x_i|spam) can set laplace smoothing in here

laplace = 1 
ham_count = [laplace] * len(word_list)
spam_count = [laplace] * len(word_list)


print("initial count(x_i|ham) and count(x_i|spam)")
print(ham_count)
print(spam_count)
print("------------------------------")


for ham_list in ham_bag:
    ham_count += ham_list

for spam_list in spam_bag:
    spam_count += spam_list

print("count(x_i|ham) and count(x_i|spam)")
print(ham_count)
print(spam_count)
print("------------------------------")

# 4. Get test data from mytest directory

test_files = os.listdir('data/mytest')

for file in test_files:
    # print(os.path.join('data/mytest', file))
    with open(os.path.join('data/mytest', file), 'r') as t:

        text = t.read()
        print(text)
        print("------------------------------")

        # Compute P(ham)*PIP(x_i|ham) and P(spam)*PIP(x_i|spam)

        P_ham = 1
        P_spam = 1

        for word_n in text.split():
            word = word_n.lower()
            if word in word_list:
                """
                print(word)
                print("index of '"+ word + "' is " + str(np.where(word_list == word)[0]))
                print("P(word|ham) = " + str(ham_count[np.where(word_list == word)[0]]/sum(ham_count)))
                print("P(word|ham) = " + str(spam_count[np.where(word_list == word)[0]]/sum(spam_count)))
                print("------------------------------")
                """
                P_ham *= ham_count[np.where(word_list == word)[0]]/sum(ham_count)
                P_spam *= spam_count[np.where(word_list == word)[0]]/sum(spam_count)
        
        # 5. Compare P(ham)*PIP(x_i|ham) and P(spam)*PIP(x_i|spam)

        print("Probablity of ham is " + str(P_ham))
        print("Probablity of spam is " + str(P_spam))

        if (P_ham > P_spam):
            print("Test mail is ham")
        else:
            print("Test mail is spam")
        print("------------------------------") 


def sklearn_NB_result():
    print("start")
    corpus = []
    y = []

    ham_files = os.listdir('data/myham')
    spam_files = os.listdir('data/myspam')
    test_files = os.listdir('data/mytest')

    for file in ham_files:
        # print(os.path.join(ham_traing, file))
        with open(os.path.join('data/myham', file), 'r') as f:
            corpus.append(f.read())
            y.append(1)

    for file in spam_files:
        # print(os.path.join(spam_traing, file))
        with open(os.path.join('data/myspam', file), 'r') as f:
            corpus.append(f.read())
            y.append(0)

    vectorizer = CountVectorizer()
    bag = vectorizer.fit_transform(corpus)
    model = MultinomialNB()
    model.fit(bag, y)




    for file in test_files:
        # print(os.path.join(spam_traing, file))
        with open(os.path.join('data/mytest', file), 'r') as f:
            text = f.read()
            print(text)   
            x_test= vectorizer.transform([text])
            r= model.predict(x_test)
            if r[0]== 1:
                print("scikit-result is ham")
            else:
                print("scikit-result is spam")

sklearn_NB_result()
