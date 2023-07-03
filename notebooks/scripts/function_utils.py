import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
import time
import math
from tqdm import tqdm

# Clustering
def select_past_timesteps(file_df, cluster, selected_columns=[]):
    df = pd.read_csv(file_df)
    if selected_columns:
        df_cluster = df[selected_columns]
        if 'Location' in selected_columns:
            df_cluster = df_cluster[df_cluster.Location.isin(cluster)]
    else:
        df_cluster = df[df.Location.isin(cluster)]
    return df_cluster

# Aggregated
def plot_results(preds: np.array, actuals: np.array, title: str):
    
    plt.scatter(actuals, preds, c='b', label='predicted')
    plt.xlabel('actual')
    plt.ylabel('predicted')
    plt.title(title)
    plt.xlim(0, plt.xlim()[1])
    plt.ylim(0, plt.ylim()[1])
    _ = plt.plot([0, 100], [0, 100], '--r', label='y=x')
    plt.legend()
    plt.show()

def build_model(estimator, X_train: np.array, y_train: np.array, X_test: np.array):
    
    model = estimator
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    return model, preds

def validate(estimator, X_train, y_train):
    scores = cross_validate(estimator, X_train, y_train, scoring=['r2', 'neg_mean_squared_error'])
    return scores

def total_averaged_metrics(metrics_list, filename):
    
    print("Total Averaged MSE: {}".format(np.round(sum(i for i, j, k in metrics_list)/len(metrics_list),3)), file=filename)
    print("Total Averaged WAPE: {}".format(np.round(sum(j for i, j, k in metrics_list)/len(metrics_list),3)), file=filename)
    print("Total Averaged R2: {}".format(np.round(sum(k for i, j, k in metrics_list)/len(metrics_list),3)), file=filename)
    print("Total Averaged MSE: {}".format(np.round(sum(i for i, j, k in metrics_list)/len(metrics_list),3)))
    print("Total Averaged WAPE: {}".format(np.round(sum(j for i, j, k in metrics_list)/len(metrics_list),3)))
    print("Total Averaged R2: {}".format(np.round(sum(k for i, j, k in metrics_list)/len(metrics_list),3)))

def predict_results(X_train, X_test, y_train, y_test, estimator):
    model, preds = build_model(estimator, X_train, y_train, X_test)

    mse, wape, r2 = performance_metrics(preds, y_test.values)
    return mse, wape, r2


def last_energy_points(df, number_timesteps):
    X = pd.DataFrame()
    for i in range(1, (number_timesteps + 1) ):
        X[f'lag_{i*15}'] = df.shift(i)
    X.dropna(inplace=True)
    X.reset_index(drop=True, inplace=True)
    y = pd.DataFrame(df[number_timesteps:])
    y.reset_index(drop=True, inplace=True)
    y.columns = ["Energy"]
    return X, y

def prepare_polynomial(X, y, deg):
    poly = PolynomialFeatures(degree=deg, include_bias=False)
    poly_features = poly.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(poly_features, y, train_size=0.8)
    return X_train, X_test, y_train, y_test

def normalize_training(X_train, feat_range=(0,1)):
    scaler = MinMaxScaler(feature_range=feat_range)
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    return X_train, scaler

def retrieve_selected_features(df, selected_features, start_date=0):
    X = pd.DataFrame()
    for i in selected_features:
        X[f'lag_{i}'] = df.shift(i)
    if 'Time' in df.columns:
        dateindex = df['Time']
    else:
        if start_date == 0:
            raise Exception("Date should be passed because there is no Time column")
        num_samples_per_house = df.shape[0]
        dateindex = pd.date_range(start_date, periods=num_samples_per_house, freq='15T', name='Time')
    X["DayOfWeek"] = dateindex.dayofweek
    X["Hour"] = dateindex.hour
    X.dropna(inplace=True)
    X.reset_index(drop=True, inplace=True)
    y = df
    y = y.iloc[selected_features[-1]:]
    y.reset_index(drop=True, inplace=True)
    return X,y

# Algorithms
def correct_wind_direction(df):
    df.loc[(df["Rumo_Vento_Med"] < 0), "Rumo_Vento_Corrected"] = 0 # SR
    df.loc[(df["Rumo_Vento_Med"] >= 0) & (df["Rumo_Vento_Med"] < 22.5), "Rumo_Vento_Corrected"] = 1 # N
    df.loc[(df["Rumo_Vento_Med"] >= 22.5) & (df["Rumo_Vento_Med"] < 67.5), "Rumo_Vento_Corrected"] = 2 # NE
    df.loc[(df["Rumo_Vento_Med"] >= 67.5) & (df["Rumo_Vento_Med"] < 112.5), "Rumo_Vento_Corrected"] = 3 # E
    df.loc[(df["Rumo_Vento_Med"] >= 112.5) & (df["Rumo_Vento_Med"] < 157.5), "Rumo_Vento_Corrected"] = 4 # SE
    df.loc[(df["Rumo_Vento_Med"] >= 157.5) & (df["Rumo_Vento_Med"] < 202.5), "Rumo_Vento_Corrected"] = 5 # S
    df.loc[(df["Rumo_Vento_Med"] >= 202.5) & (df["Rumo_Vento_Med"] < 247.5), "Rumo_Vento_Corrected"] = 6 # SW
    df.loc[(df["Rumo_Vento_Med"] >= 247.5) & (df["Rumo_Vento_Med"] < 292.5), "Rumo_Vento_Corrected"] = 7 # W
    df.loc[(df["Rumo_Vento_Med"] >= 292.5) & (df["Rumo_Vento_Med"] < 337.5), "Rumo_Vento_Corrected"] = 8 # NW
    df.loc[(df["Rumo_Vento_Med"] >= 337.5) & (df["Rumo_Vento_Med"] <= 360.0), "Rumo_Vento_Corrected"] = 1 # N
    return df

def truncate_metric(metric):
    m = math.trunc(10000 * metric) / 10000
    return m 
    
def performance_metrics(preds: np.array, actuals: np.array, filename=None):

    # calculate performance metrics
    
    mse = truncate_metric(mean_squared_error(actuals, preds))
    rmse = truncate_metric(mean_squared_error(actuals, preds, squared=False))
    wape = truncate_metric(np.sum(np.abs(preds - actuals)) / np.sum(np.abs(actuals))) * 100
    r2 = truncate_metric(r2_score(actuals, preds))
    
    # print performance metrics
    if(filename != None):
        print('MSE: %.4f' % mse, file=filename)
        print('WAPE: %.2f' % wape, file=filename)
        print('R2: %.4f' % r2, file=filename)
    print('MSE: %.4f' % mse)
    print('RMSE: %.4f' % rmse)
    print('WAPE: %.2f' % wape)
    print('R2: %.4f' % r2)
    return mse, wape, r2

def past_timesteps(df, number_of_timesteps):
    df = df.sort_values(by=['Location', 'Time'])
    for i in tqdm(range(1, (number_of_timesteps + 1))):
        df.loc[df['Time'].shift(i) == df['Time'] - pd.Timedelta(i * 15, 'm'), f"lag_{i}"] = df['Energy'].shift(i)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def train_test_split_timeseries(df, train_size=0.8):
    df_train, df_test = [], []
    column_names = df.columns
    for i in df.Location.unique():
        df_loc = df[df.Location == i]
        t_size = int(df_loc.shape[0] * train_size)
        train = df_loc.iloc[:t_size, :]
        test = df_loc.iloc[t_size:, :]
        df_train.append(train)
        df_test.append(test)
    for d in df_train:
        train = d if train.empty else pd.concat([train, d], axis=0)
    for d in df_test:
        test = d if test.empty else pd.concat([test, d], axis=0)
    train.columns = column_names
    test.columns = column_names
    return train, test

# def train_test_split_timeseries(df, train_size=0.8):
#     column_names = df.columns
#     tmp_tr = pd.DataFrame()
#     tmp_te = pd.DataFrame()
#     for i in df.Location.unique():
#         df_loc = df[df.Location == i]
#         train_size = int(df_loc.shape[0] * train_size)
#         train = df_loc.iloc[:train_size, :]
#         test = df_loc.iloc[train_size:, :]
#         tmp_tr = pd.concat([tmp_tr, train], axis=1) if not tmp_tr.empty else train
#         tmp_te = pd.concat([tmp_te, test], axis=1) if not tmp_te.empty else test
#     train = pd.DataFrame(tmp_tr, columns=column_names)
#     test = pd.DataFrame(tmp_te, columns=column_names)
#     return train, test

def test_leave_house_out(df, estimator, locations, filename, split_timeseries=False, train_size=0.8):
    if(split_timeseries):
        print("split timeseries")
        train, test = train_test_split_timeseries(df,train_size=train_size)
    else:
        print("split location")
        test = df[df['Location'].isin(locations)]
        train = df[~df['Location'].isin(locations)]
    print("Train set: ", train.shape)
    print("Test set: ", test.shape)
    X_train = train.drop(['Time', 'Energy', 'Location'], axis=1)
    X_test = test.drop(['Time', 'Energy', 'Location'], axis=1)
    y_train = train['Energy']
    y_test = test['Energy']

    X_train_norm, scaler = normalize_training(X_train)
    X_test_norm = scaler.transform(X_test)
    model = estimator
    init = time.time()
    model.fit(X_train_norm, y_train)
    y_pred = model.predict(X_test_norm)
    end = time.time()
    print('Elapsed time: {:.4f} s'.format(end - init), file=filename)
    mse, wape, r2 = performance_metrics(y_pred, y_test.values.reshape(-1), filename)
    return mse, wape, r2, model

# Individual
def split_train_test_timeseries(X, y, train_size=0.8):
    n_train_samples = int(len(X) * train_size)
    X_train = X[:n_train_samples]
    X_test = X[n_train_samples:]
    y_train = y[:n_train_samples]
    y_test = y[n_train_samples:]
    return X_train, X_test, y_train, y_test

def total_averaged_metrics_individual(metrics_list):
    mse, wape, r2 = 0,0,0,
    for house in metrics_list:
        mse += sum(i for i, j, k in house)
        wape += sum(j for i, j, k in house)
        r2 += sum(k for i, j, k in house)
    t_mse = np.round(mse/len(metrics_list),3)
    t_wape = np.round(wape/len(metrics_list),3)
    t_r2 = np.round(r2/len(metrics_list),3)
    print("Total Averaged MSE: {}".format(t_mse))
    print("Total Averaged WAPE: {}".format(t_wape))
    print("Total Averaged R2: {}".format(t_r2))
    return t_mse, t_wape, t_r2

def build_predict_show(df, number_timesteps, estimator, selected_features=None, normalize=False, train_size=0.8, start_timestep=1 ):
    full_start = time.time()
    metrics_list = []
    for i in range(start_timestep,(number_timesteps + 1)):
        start = time.time()
        print("\nNumber of features ", i)
        if(selected_features != None):
            X, y = retrieve_selected_features(df, selected_features, "2019-01-01")
        else:
            X, y = last_energy_points(df, i)
        
        X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=train_size)
        
        if(normalize):
            scaler = MinMaxScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
        
        model, preds = build_model(estimator, X_train, y_train, X_test)
        mse, wape, r2 = performance_metrics(preds.reshape(-1), y_test.values.reshape(-1))
        metrics_list.append((mse, wape,r2))
        print("\nElapsed time: %.3f seconds" % (time.time() - start))
    print("\nFull Elapsed time: %.3f seconds" % (time.time() - full_start))
    return model, preds, metrics_list

def show_graphic_per_timestep(metrics_list, number_of_houses):
    mse_list = []
    wape_list = []
    r2_list = []

    for i in range(0,len(metrics_list)):
        mse_list.append(metrics_list[i][0][0])
        wape_list.append(metrics_list[i][0][1])
        r2_list.append(metrics_list[i][0][2])
        
    plt.plot(range(0,number_of_houses), mse_list)
    plt.title('MSE per house')
    plt.xlabel('Number of houses')
    plt.ylabel('MSE')
    plt.show()
    
    plt.plot(range(0,number_of_houses), wape_list)
    plt.title('WAPE per house')
    plt.xlabel('Number of houses')
    plt.ylabel('WAPE')
    plt.show()
    
    plt.plot(range(0,number_of_houses), r2_list)
    plt.title('R2 per house')
    plt.xlabel('Number of houses')
    plt.ylabel('R2')
    plt.show()


 #["energy_lag_1", "energy_lag_2", "energy_lag_3", "energy_lag_4", "energy_lag_96", "energy_lag_192", "energy_lag_288", "energy_lag_384", "energy_lag_480", "energy_lag_576", "energy_lag_672", "DayOfWeek", "Hour"]