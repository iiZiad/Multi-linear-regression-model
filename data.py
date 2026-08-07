import numpy as np
import scalling 

np.random.seed(3)
data_size = 100
max_min = [
    [1,120],
    [1,60],
]
x_col1_hours = np.linspace(1,max_min[0][1],data_size) + np.random.normal(0,2,data_size)
x_col2_atnds = np.linspace(1,max_min[1][1],data_size) + np.random.normal(0,1,data_size)

x_raw = np.column_stack((x_col1_hours, x_col2_atnds))
# x_raw = x_col1_hours
number_of_features = x_raw.shape[1]
# number_of_features = 1

x_scalled = scalling.z_score(x_raw, number_of_features, x_raw)


y = (np.linspace(1,4-0.01,data_size) + np.random.normal(0,0.05,data_size) ).reshape(-1)

iterations_lmt = 10000
alpha = 0.0025