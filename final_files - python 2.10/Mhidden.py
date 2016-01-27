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
m = [100, 150, 200, 250, 300, 350, 400, 450, 500]
train_data = np.array(train_data)
m_matrix = np.zeros(9)
for i in range(0, 9):
    net = nn.Network([108, m[i], 2], nn.Quad, nn.Cos, nn.Sin, False)
    m_matrix[i] = net.train_network(train_data, 20, 10, 0.1, 0.1, test_data)
print m_matrix
