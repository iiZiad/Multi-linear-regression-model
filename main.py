import numpy as np
import data, regression
from scalling import z_score
import matplotlib.pyplot as plt
from inout import input_data
import os
os.system('cls')
np.set_printoptions(precision=2, suppress= True)
w_init = np.random.randn(data.number_of_features)
b_init = 0.5
w, b, cost_hist = regression.reg(
    data.x_scalled,
    data.y,
    w_init,
    b_init,
    data.data_size,
    data.iterations_lmt,
    data.number_of_features,
    data.alpha)
input_data(w, b, cost_hist)
# # print( np.dot(w,[])+b )
# print( np.dot(w,[z_score(10, 0, data.x_raw, 0),z_score(25, 0, data.x_raw, 1)])+b )
# # print(f' gpa = {np.dot(w,z_score(10, 1, data.x_raw))+b} ')
