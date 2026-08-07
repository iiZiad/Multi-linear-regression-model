import numpy as np
from time import sleep
def compute_cost(y, m, f_wb):
    return (1 / (2*m)) * np.sum( (f_wb - y)**2 ) 

def compute_f_wb(x, w, b, m):
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i]= (np.dot(x[i], w) + b).item()
    return f_wb

def compute_descent(x, y, m, feture_cols, f_wb):
    dj_dw = np.zeros(feture_cols)
    for j in range(feture_cols):
        if feture_cols > 1:
            dj_dw[j] = ( 1/m ) * np.sum((f_wb - y)*x[:, j])
        else :
            dj_dw[j] = ( 1/m ) * np.sum((f_wb - y)*x)
    if feture_cols < 2 :
        dj_dw = dj_dw.item()
    dj_db = ( 1/m ) * np.sum( f_wb - y )
    return dj_dw, dj_db

def gradient_descent(w, b, dj_dw, dj_db, alpha):
    w = w - alpha*dj_dw
    b = b - alpha*dj_db
    return w, b
def summery_of_iter(counter, cost, w, b, feture_cols,cond = 0):
    # if counter % 1000 == 0 :
    #     print(f'counter = {counter},cost = {cost}',end="")
    #     for x in range(feture_cols):
    #         print(f'w = {w[x]} ', sep="", end="")
    #     print(f'b = {b}')
    ...
def reg(x, y, w, b, m, iterations, feture_cols, alpha):
    cost_hist = []
    f_wb = compute_f_wb(x, w, b, m)
    cost = compute_cost(y, m, f_wb)
    counter = 0
    cond = 0
    print('Loading...',end='\r')
    while cost > 0.0005 and iterations >= counter :
        dj_dw, dj_db = compute_descent(x, y, m, feture_cols, f_wb)
        w, b = gradient_descent(w, b, dj_dw, dj_db, alpha)
        f_wb = compute_f_wb(x, w, b, m)
        cost = compute_cost(y, m, f_wb)
        cost_hist.append(cost)
        #summery_of_iter(counter, cost, w, b, feture_cols)
        counter += 1
    print('Training Completed!')
    return w, b, cost_hist