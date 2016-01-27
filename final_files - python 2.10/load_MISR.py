import scipy.io
import numpy as np

def ret_filename(which):
    if which == 1:
        temp = "MISR1.mat"
    elif which == 2:
        temp = "MIST2.mat"
    else:
        temp = "MODIS.mat"
    filename = "./data/MIR_datasets/{}".format(temp)
    return filename

def normalize(a):
    a = (a-np.mean(a))/np.std(a)
    return (a-np.amin(a))/(np.amax(a) - np.amin(a))

def load_MISR(method = 1):
    data = scipy.io.loadmat(ret_filename(1))["MISR1"]
    data = data.T
    for i in range(1,17):
        data[i] = normalize(data[i])
    data = data.T
    data_batches = [[] for i in range(815)]
    [data_batches[int(x[0])-1].append(x) for x in data]
    data_batches = [x for x in data_batches if len(x) > 0]
    if method == 1:
        data_batches = [([x[1:-1] for x in xs],xs[0][-1]) for xs in data_batches]
    else:
        data_batches = ([np.array([x[1:-1] for x in xs]) for xs in data_batches],
                        [xs[0][-1] for xs in data_batches])
    return data_batches

