import load_MNIST as ms
import load_heart_disease as hd
import load_MISR as misr
import load_Adult as ad
import neural_net as nn
import neural_net_ext as nne
import randff as rff
import randff_ext as rffe
import numpy as np
from sklearn import cross_validation


train_data, test_data = ad.load_Adult_wrapper()
K = 3
kf = cross_validation.KFold(len(train_data),K , shuffle = True, random_state = 2)
print(kf)

#train_data_subset, test_data_subset = ad.cross_validation_method(4, train_data)
# print (test_data_subset[0][0:1])

lamda = [0.01]
m = [100, 200, 300, 400, 500]
cross_val_matrix = np.zeros([K,len(lamda)])
train_data = np.array(train_data)
K = 0
for train_index, test_index in kf:
    net = nn.Network([108, 2, 2], nn.Quad, nn.Cos, nn.Sin, False)
    for i in range(0, len(lamda)):
        cross_val_matrix[K][i] = net.train_network(train_data[train_index], 20, 10, 0.1, lamda[i], train_data[test_index])
    K = K + 1
mean_matrix = np.mean(cross_val_matrix, axis = 0)
model_index = np.argmax(mean_matrix)
print lamda[model_index]

net.train_network(train_data, 20, 10, 0.1, lamda[model_index], test_data)

"""
for k in range(0,3):
    test_data_subset[k] = [l[0] for l in test_data_subset[k]]
    train_data_subset[k] = [l[0] for l in train_data_subset[k]]
print test_data_subset[0][1:2]
"""
