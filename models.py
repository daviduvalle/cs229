import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import f1_score, confusion_matrix
from sklearn import preprocessing, svm
from sklearn.ensemble import RandomForestClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

import numpy as np
from sklearn.preprocessing import scale

def load_data():
    train_data = pd.read_csv('data/train.csv')
    test_data = pd.read_csv('data/test.csv')
    train_data = train_data.drop(['Unnamed: 0'], axis=1)
    test_data = test_data.drop(['Unnamed: 0'], axis=1)

    train = train_data.to_numpy()
    test = test_data.to_numpy()

    train_x = train[:,0:10]
    train_y = train[:,10].astype(int)

    test_x = test[:,0:10]
    test_y = test[:,10].astype(int)

    return train_x, train_y, test_x, test_y

def polynomial_regression():
    train_x, train_y, test_x, test_y = load_data()

    poly = PolynomialFeatures(5)

    train_x = preprocessing.scale(train_x)

    train_x = poly.fit_transform(train_x)

    logistic_regression = LogisticRegression(verbose=5, solver='liblinear', max_iter=5000)
    logistic_regression.fit(train_x, train_y)

    test_x = preprocessing.scale(test_x)
    test_x = poly.fit_transform(test_x)
    output = logistic_regression.predict(test_x)

    score = logistic_regression.score(test_x, test_y)

    print('score {}'.format(score))

    print('f1 {}'.format(f1_score(test_y, output)))

    print('c_matrix {}'.format(confusion_matrix(test_y, output)))


def logistic_regression():
    train_x, train_y, test_x, test_y = load_data()
    train_x = preprocessing.scale(train_x)
    test_x = preprocessing.scale(test_x)

    logistic = LogisticRegression(verbose=10, max_iter=1000)
    logistic.fit(train_x, train_y)

    output = logistic.predict(test_x)

    score = logistic.score(test_x, test_y)

    print('score {}'.format(score))

    print('f1 {}'.format(f1_score(test_y, output)))

    print('c_matrix {}'.format(confusion_matrix(test_y, output)))

def support_vector():
    train_x, train_y, test_x, test_y = load_data()
    train_x = preprocessing.scale(train_x)
    test_x = preprocessing.scale(test_x)

    clf = svm.SVC(kernel='poly', verbose=10)
    clf.fit(train_x, train_y)
    output = clf.predict(test_x)

    print('f1 {}'.format(f1_score(test_y, output)))

    print('c_matrix {}'.format(confusion_matrix(test_y, output)))

def decision():
    train_x, train_y, test_x, test_y = load_data()
    train_x = preprocessing.scale(train_x)
    test_x = preprocessing.scale(test_x)

    clf = RandomForestClassifier()
    clf.fit(train_x, train_y)
    output = clf.predict(test_x)

    score = clf.score(test_x, test_y)

    print('score {}'.format(score))

    print('f1 {}'.format(f1_score(test_y, output)))

    print('c_matrix {}'.format(confusion_matrix(test_y, output)))

def deep_learn():
    train_x, train_y, test_x, test_y = load_data()
    train_x = preprocessing.scale(train_x)
    test_x = preprocessing.scale(test_x)

    model = Sequential()
    model.add(Dense(20, input_dim=train_x.shape[1], activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['mae', 'accuracy'])

    model.fit(train_x, train_y,
              epochs=50,
              batch_size=1)

    output = model.predict(test_x)
    output = (output > 0.5)

    score = model.evaluate(test_x, test_y)
    print('{}'.format(model.metrics_names))
    print('score {}'.format(score))
    print('f1 {}'.format(f1_score(test_y, output)))

    print('c_matrix {}'.format(confusion_matrix(test_y, output)))

def check_pca():
    train_x, train_y, test_x, test_y = load_data()
    train_x = preprocessing.scale(train_x)
    #test_x = preprocessing.scale(test_x)

    pca = PCA(n_components=3)
    output = pca.fit_transform(train_x)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    xs = output[:,0]
    ys = output[:,1]
    zs = output[:,2]

    c_array = []
    labels = []
    labels.append('red')
    labels.append('blue')
    for i in range(train_y.shape[0]):
        if train_y[i] == 1:
            c_array.append('red')
        else:
            c_array.append('blue')

    ax.scatter(xs, ys, zs, c=c_array, s=50, alpha=0.6, edgecolors='w')
    plt.title('PCA 3-dimension reduction')

    plt.legend(['Healthy'])

    plt.show()

def data_info():
    train_data = pd.read_csv('data/train.csv')
    train_data = train_data.drop(['Unnamed: 0'], axis=1)
    train_data.describe().to_csv('data_description.csv')

def main():
    #data_info()
    #polynomial_regression()
    #logistic_regression()
    #support_vector()
    #decision()
    deep_learn()
    #check_pca()

main()