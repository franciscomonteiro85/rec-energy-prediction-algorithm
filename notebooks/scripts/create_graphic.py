import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import sys

def get_metrics_lists(lines):
    model_names = [lines[0][:-1],lines[5][0:-1],lines[10][0:-1]]
    mse = [lines[1].split(":")[1][:-1],lines[6].split(":")[1][:-1],lines[11].split(":")[1][:-1]]
    wape = [lines[2].split(":")[1][:-1],lines[7].split(":")[1][:-1],lines[12].split(":")[1][:-1]]
    r2 = [lines[3].split(":")[1][:-1],lines[8].split(":")[1][:-1],lines[13].split(":")[1][:-1]]
    mse = list(map(float, mse))
    wape = list(map(float, wape))
    wape = list(map(lambda x: x/100, wape))
    r2 = list(map(float, r2))
    return model_names, mse, wape, r2

def get_ml_metrics(lines):
    model_names = [lines[0][:-1], lines[4][0:-1]]
    mse = [lines[1].split(":")[1][:-1],lines[5].split(":")[1][:-1]]
    wape = [lines[2].split(":")[1][:-1],lines[6].split(":")[1][:-1]]
    r2 = [lines[3].split(":")[1][:-1],lines[7].split(":")[1][:-1]]
    mse = list(map(float, mse))
    wape = list(map(float, wape))
    r2 = list(map(float, r2))
    return model_names, mse, wape, r2

def show_graphic(lines, lines_noml, title, filename):
    no_ml, mse_no, wape_no, r2_no = get_ml_metrics(lines_noml)
    model_names, mse, wape, r2 = get_metrics_lists(lines)
    model_names.insert(0,no_ml[0])
    mse.insert(0,mse_no[0])
    wape.insert(0,wape_no[0])
    r2.insert(0,r2_no[0])
    X_names = ['MSE', 'WAPE', 'R2']
    X_axis = np.arange(len(X_names))
    pos = 0
    bar_width = 0.15
    fig, ax = plt.subplots()

    for i in range(len(model_names)):
        bar = (mse[i], wape[i], r2[i])
        b = ax.bar(X_axis + pos, bar, bar_width-0.01, label=model_names[i])
        pos = pos + bar_width
        ax.bar_label(b, fontsize=6, fmt='%.3f')
        
    ax.set_ylim(0,1)
    ax.set_xticks(X_axis+bar_width+bar_width/2,X_names)
    ax.set_xlabel("Model Comparison")
    ax.set_ylabel("Metrics")
    ax.set_title(title)
    ax.legend()
    plt.savefig(filename)
    plt.show()

# 1st argument - No ML gpu_log file
# 2nd argument - 3 model gpu_log file
# 3rd argument - Graphic Title
# 4th argument - File to save the image

def main():
    with open(sys.argv[1]) as f:
        lines_noml = f.readlines()
    lines_noml = lines_noml[1:9]

    with open(sys.argv[2]) as f:
        lines = f.readlines()
    lines_porto = lines[-14:]

    show_graphic(lines_porto, lines_noml, sys.argv[3], sys.argv[4])

if __name__ == "__main__":
    main()