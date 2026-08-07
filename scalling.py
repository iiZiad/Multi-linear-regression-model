import numpy as np
import copy

def z_score(x_copy, num_of_feartures, data, col = -1):
    x = copy.deepcopy(x_copy)
    if num_of_feartures > 1:
        for i in range(num_of_feartures):
            mean , std = np.mean(data[:,i]) , np.std(data[:,i])
            x[:,i] = (x[:,i] - mean) / std

    else :
        mean , std = np.mean(data[:,col]) , np.std(data[:,col])
        x = (x - mean) / std

    return x