import matplotlib.pyplot as plt
import numpy as np

def get_noml_metrics(lines):
    model_names = [lines[0][:-1]]
    mse = [lines[1].split(":")[1][:-1]]
    wape = [lines[2].split(":")[1][:-1]]
    r2 = [lines[3].split(":")[1][:-1]]
    mse = list(map(float, mse))
    wape = list(map(float, wape))
    r2 = list(map(float, r2))
    return model_names, mse, wape, r2

def get_ml_metrics(lines):
    model_names = [lines[0][:-1],lines[4][0:-1],lines[8][0:-1]]
    mse = [lines[1].split(":")[1][:-1],lines[5].split(":")[1][:-1],lines[9].split(":")[1][:-1]]
    wape = [lines[2].split(":")[1][:-1],lines[6].split(":")[1][:-1],lines[10].split(":")[1][:-1]]
    r2 = [lines[3].split(":")[1][:-1],lines[7].split(":")[1][:-1],lines[11].split(":")[1][:-1]]
    mse = list(map(float, mse))
    wape = list(map(float, wape))
    wape = list(map(lambda x: x/100, wape))
    r2 = list(map(float, r2))
    return model_names, mse, wape, r2

def get_cluster_metrics(lines):
    cursor = 0
    model_names, mse, wape, r2, number_clust = [], [], [], [], []
    for i in lines:
        if i.startswith("Number"):
            number_clust.append(int(i[-3:-1].strip()))
        else:
            if cursor == 0:
                model_names.append(i[:-1])
                cursor += 1
            elif cursor == 1:
                mse.append(i.split(":")[1][:-1])
                cursor += 1
            elif cursor == 2:
                wape.append(i.split(":")[1][:-1])
                cursor += 1
            elif cursor == 3:
                r2.append(i.split(":")[1][:-1])
                cursor = 0
    mse = list(map(float, mse))
    wape = list(map(float, wape))
    wape = list(map(lambda x: x/100, wape))
    r2 = list(map(float, r2))
    return model_names, mse, wape, r2, number_clust
            
def show_metric_per_cluster(metric, metric_name, number_clust, title, filename):
    plt.plot(number_clust, metric)
    plt.xlabel("Number of Clusters")
    plt.ylabel(metric_name)
    plt.title(title)
    plt.savefig(filename)
    plt.show()

def show_cluster_graphic(lines, title, filename):
    model_names, mse, wape, r2, number_clust = get_cluster_metrics(lines)
    X_names = ['MSE', 'WAPE', 'R2']
    X_axis = np.arange(len(X_names))
    pos = 0
    bar_width = 0.15
    fig, ax = plt.subplots()
    for i in (0,1):# range(len(set(model_names))):
        for j in range(len(set(model_names))):
            bar = (mse[j], wape[j], r2[j])
            b = ax.bar(X_axis + pos, bar, bar_width-0.01, label=model_names[j])
            pos = pos + bar_width
            ax.bar_label(b, fontsize=6, fmt='%.3f')
        break
    ax.set_ylim(0,1)
    ax.set_xticks(X_axis+bar_width+bar_width/2,X_names)
    ax.set_xlabel("Model Comparison")
    ax.set_ylabel("Metrics")
    ax.set_title(title)
    ax.legend()
    plt.savefig(filename)
    plt.show()

def show_graphic(lines, lines_noml, title, filename):
    no_ml, mse_no, wape_no, r2_no = get_noml_metrics(lines_noml)
    model_names, mse, wape, r2 = get_ml_metrics(lines)
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