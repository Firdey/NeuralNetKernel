# Copy into other code to run directly!!!! Though has bugs!!!

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
