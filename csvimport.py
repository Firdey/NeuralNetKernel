# Copy into other code to run directly!!!! Though has bugs!!!
import numpy as np
import csv
with open('forestfiretrans.csv', 'rb') as csvfile:
    next(csvfile, None)
    forest = csv.reader(csvfile)
    forest_data = [map(float, l) for l in forest]
    print forest_data
    #for row in forest:
for i in range(0, 517):
    forest_data[i].pop(0)

yforest = [0 for i in range(517)]
for j in range(0,517):
    yforest[j] = forest_data[j][8]
    forest_data[j].pop(8)
print yforest
print forest_data
"""
layer_size = [8, 30, 1]
loss = Mean_S_E
activ = Sigmoid
xdata = forest_data
ydata = yforest
epsilon = 0.05
alpha = 0.01
penalty = [0.1, 0.1]
max_it = 1000
tolerance = 0.5
Forest_neural = Network(layer_size, loss, activ, xdata, ydata, epsilon, alpha, penalty, max_it, tolerance)
print Forest_neural.layer_size
"""
from sklearn import cross_validation
X_train, X_test, y_train, y_test = cross_validation.train_test_split(forest_data, yforest, test_size=0.3, random_state=2)
#print X_train
#print y_test
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)
K = 11
print len(X_train)
print len(y_train)
kf = cross_validation.KFold(len(y_train),K , shuffle = True, random_state = 2)
print(kf)

for train_index, test_index in kf:
    print("TRAIN:", train_index, "TEST:", test_index)
    print X_train[train_index]
    print X_train[test_index]
    print y_train[train_index]
    print y_train[test_index]
    X_trainK = X_train[train_index]
    X_testK = X_train[test_index]
    y_trainK = y_train[train_index]
    y_testK = y_train[test_index]
    #Forest_neural = Network(layer_size, loss, activ, X_train, X_test, epsilon, alpha, penalty, max_it, tolerance)
