import numpy as np
import pandas as pd
import re
import time
from sklearn.model_selection import train_test_split
from scripts.function_utils import normalize_training, performance_metrics, expanding_window_split_location

def read_clusters_from_file(filename, n_clusters):
    all_clusters = []
    readFile = open(filename, 'r')
    file = readFile.readlines()
    clcl = file[2:n_clusters+1]
    for c in clcl:
        clcli = list(c.split(","))
        clcli[0] = clcli[0].split("[")[-1]
        clcli.pop() ## pop \n element
        house_list = np.array(clcli)
        cluster_list = []
        for i in house_list:
            result = re.sub(r'[^0-9]','',i)
            cluster_list.append(int(result))
        cluster_list = np.array(cluster_list)
        all_clusters.append(cluster_list)
    return all_clusters

def split_into_clusters(cluster):
    clstr_lst = []
    for i in range(len(np.unique(cluster))):
        clstr_lst.append(np.where(cluster == i)[0])
    return clstr_lst

def dataframe_by_cluster(cl_list, df):
    clusters = []
    for i in cl_list:
        dataframe = df.iloc[:, i]
        clusters.append(dataframe)
    return clusters

def cluster_predict(df, estimators, names, n_splits=10):
    preds_list, y_test_list = [], []
    for e in estimators:
        model = e
        for i in range(n_splits):
            X_train, X_test, y_train, y_test = train_test_split(df.drop(['Energy', 'Time', 'Location'], axis=1), df['Energy'], train_size=0.8, random_state=i*3)
            X_train_norm, scaler = normalize_training(X_train)
            X_test_norm = scaler.transform(X_test)
            init = time.time()
            model.fit(X_train_norm, y_train)
            y_pred = model.predict(X_test_norm)
            preds_list.append(y_pred)
            y_test_list.append(y_test)
        end = time.time()
        print('Elapsed time training and predicting: {:.4f} s'.format(end - init))
    return preds_list, y_test_list

def cluster_predict_ew(df, estimators, names, n_splits=10):
    preds_list, y_test_list = [], []
    for e in estimators:
        model = e
        init = time.time()
        for i in range(n_splits):
            X_train, X_test, y_train, y_test = expanding_window_split_location(df, cv=i, n_splits=n_splits)
            X_train = X_train.drop(['Time', 'Location', 'Energy'], axis=1)
            X_test = X_test.drop(['Time', 'Location', 'Energy'], axis=1)
            y_train = y_train['Energy']
            y_test = y_test['Energy']
            X_train_norm, scaler = normalize_training(X_train)
            X_test_norm = scaler.transform(X_test)
            model.fit(X_train_norm, y_train)
            y_pred = model.predict(X_test_norm)
            preds_list.append(y_pred)
            y_test_list.append(y_test)
        end = time.time()
        print('Elapsed time training and predicting: {:.4f} s'.format(end - init))
    return preds_list, y_test_list

def aggregate_cluster_predictions(pred_list, actual_list, names, filename):
    print("Aggregating predictions...")
    dictio, dictio_act = {}, {}
    rmse_list, wape_list, r2_list = [], [], []
    for model in range(0, len(names)):
        preds, actuals = [], []
        for cluster in range(0, len(pred_list)):
            preds = np.append(preds, pred_list[cluster][model])
            actuals = np.append(actuals, actual_list[cluster][model])
            dictio[names[model]] = preds
            dictio_act[names[model]] = actuals
        preds = dictio[names[model]]
        actuals = dictio_act[names[model]]
        print("\n----------------------------", file=filename)
        print("{}".format(names[model]), file=filename)
        print("----------------------------\n", file=filename)
        print("\n----------------------------")
        print("{}".format(names[model]))
        print("----------------------------\n")
        rmse, wape, r2 = performance_metrics(preds, actuals, filename)
        rmse_list.append(rmse)
        wape_list.append(wape)
        r2_list.append(r2)
    return rmse_list, wape_list, r2_list

def select_past_timesteps(file_df, cluster, selected_columns=[]):
    df = pd.read_csv(file_df)
    if selected_columns:
        df_cluster = df[selected_columns]
        if 'Location' in selected_columns:
            df_cluster = df_cluster[df_cluster.Location.isin(cluster)]
    else:
        df_cluster = df[df.Location.isin(cluster)]
    return df_cluster