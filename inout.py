import numpy as np
import data
from scalling import z_score
import matplotlib.pyplot as plt
from time import sleep
import os
hours = 0
attendas = 0

def input_data(w, b, cost_hist):
    hours = int(input(f"Input number of hours (out of {data.max_min[0][1]})\n> "))
    attendas = int(input(f"Input number of attendas (out of {data.max_min[1][1]})\n> "))
    if not(data.max_min[0][0] <= hours <= data.max_min[0][1]) or not(data.max_min[1][0] <= attendas <= data.max_min[1][1]) :
        print('Invald input', end='\r')
        sleep(2)
        os.system('cls')
        input_data(hours, attendas, cost_hist)
    output(w,b,hours, attendas, cost_hist)
def output(w,b,hours, attendas, cost_hist):
    prediction = (np.dot(w,[z_score(hours, 0, data.x_raw, 0),z_score(attendas, 0, data.x_raw, 1)])+b ).item()
    print(f'Predicted GPA = {float(prediction):.2f}')

    ax, fig = plt.subplots(1, 2, figsize=(10,4))
    line = np.linspace(-2,2,100)
    fig[0].scatter(data.x_scalled[:,0],data.y, color='green', label='data', s=15)
    fig[0].plot(line, ((w[0]+w[1])*line)+b, label='best fit')
    fig[0].set_xlabel('$w_{1}$ ,$w_{2}$')
    fig[0].set_ylabel('$Y$')
    fig[0].set_title('GPA prediection\naccording to study hours & number of attendans', fontsize=10)
    plt.legend()
    fig[1].plot(np.arange(data.iterations_lmt), cost_hist[:-1])
    fig[1].set_xlabel('iterations')
    fig[1].set_ylabel('cost')
    plt.show()
