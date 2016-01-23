import numpy as np

def swap(a):
    if a =="Present":
        a = 1
    else:
        a = 0
    return a

def swap2(a):
    if a =="1\n":
        a = 1
    else:
        a = 0
    return a

def normalize(a):
    a = (a - np.mean(a)) / np.std(a)
    a_min = np.amin(a)
    a_max = np.amax(a)
    a = (a - a_min)/(a_max - a_min)
    return a

def change_y(a):
    if a==1:
        return np.array([0,1])
    else:
        return np.array([1,0])
    

def get_heart_disease_data():
    # read data from the file
    filename = "data/SAheart.data"
    with open(filename) as f:
        content = f.readlines()

    # modify data to a desired format
    data = [np.array(line.split(',')) for line in content]
    # change words to numbers
    [(x.itemset(5, swap(x[5])), x.itemset(10, swap2(x[10])))  for x in data]
    # row.name variable is not needed
    data = [x[1:] for x in data]
    # change strings to floats
    data = np.array([x.astype(np.float) for x in data])
    data = data.T
    for i in range(4):
        data[i] = normalize(data[i])
    for i in range(5,9):
        data[i] = normalize(data[i])
    return [(x[:-1],change_y(x[-1])) for x in data.T]

