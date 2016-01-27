import numpy as np
import csv

def load_Adult(data_path):    
    with open(data_path, "r") as d:
        temp = csv.reader(d, delimiter= ",")
        data = [np.array(line).astype(float) for line in temp]
    data = [(np.array(x[1:-1]),np.array([x[-1]-1,2 - x[-1]])) for x in data]
    return data

def load_Adult2(data_path):
    with open(data_path, "r") as d:
        temp = csv.reader(d, delimiter= ",")
        data = [np.array(line, dtype = float) for line in temp]
    X = np.array([x[1:-1] for x in data])
    y = np.array([x[-1]-1 for x in data])
    return (X,y)


def load_Adult_wrapper(method = 1):
    train_path = "./data/Adult/Adult_train_data.csv"
    test_path = "./data/Adult/Adult_test_data.csv"
    if method == 1:
        train_data = load_Adult(train_path)
        test_data = load_Adult(test_path)
    else:
        train_data = load_Adult2(train_path)
        test_data = load_Adult2(test_path)
    return (train_data, test_data)

