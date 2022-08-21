import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

def data():
    x_train = pd.read_csv("C:\\Users\\asus\\Desktop\\ml.hw3\\fashion-mnist_train.csv")
    y_train = x_train['label']
    x_test = pd.read_csv("C:\\Users\\asus\\Desktop\\ml.hw3\\fashion-mnist_test.csv")
    y_test = x_test['label']
    del x_train['label']
    del x_test['label']
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size = 0.1)
    return [x_train, y_train, x_test, y_test, x_valid, y_valid]

x_train, y_train, x_test, y_test, x_valid, y_valid = data()


SOLVER = 'adam'
LR = 0.1
clf = MLPClassifier(solver = SOLVER, learning_rate = "constant", learning_rate_init = LR, hidden_layer_sizes = (100, 2))
clf.fit(x_train, y_train)
plt.plot(clf.loss_curve_)
plt.title("Train loss: solver:{} learning_rate:{} layerrs:{} layers_weight:{}".format(SOLVER, LR, 2, 100))
clf.fit(x_valid, y_valid)
plt.plot(clf.loss_curve_)
plt.legend(["train", "test"])
plt.show()
ypred = clf.predict(x_test)
accuracy_score(y_test, ypred)
