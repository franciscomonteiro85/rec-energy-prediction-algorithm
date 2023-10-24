import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
import time
import math
from cuml.metrics import mean_squared_error as mse_gpu
from cuml.metrics import r2_score as r2_gpu
import cudf
import cupy
from tqdm import tqdm
from sklearn.model_selection import TimeSeriesSplit

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

def total_averaged_metrics(metrics_list, filename=0):
    rmse = np.round(sum(i for i, j, k in metrics_list)/len(metrics_list),3)
    wape = np.round(sum(j for i, j, k in metrics_list)/len(metrics_list),3)
    r2 = np.round(sum(k for i, j, k in metrics_list)/len(metrics_list),3)
    if(filename != 0):
        print("Total Averaged RMSE: {}".format(rmse), file=filename)
        print("Total Averaged WAPE: {}".format(wape), file=filename)
        print("Total Averaged R2: {}".format(r2), file=filename)
    print("Total Averaged RMSE: {}".format(rmse))
    print("Total Averaged WAPE: {}".format(wape))
    print("Total Averaged R2: {}".format(r2))
    return rmse, wape, r2

def predict_results(X_train, X_test, y_train, y_test, estimator):
    model, preds = build_model(estimator, X_train, y_train, X_test)

    rmse, wape, r2 = performance_metrics(preds.flatten(), y_test.values.flatten())
    return rmse, wape, r2, model


def last_energy_points(df, number_timesteps):
    X = pd.DataFrame()
    for i in range(1, (number_timesteps + 1) ):
        X[f'lag_{i}'] = df.shift(i)
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

def normalize_training(X_train, feat_range=(0,1), gpu=False):
    if(gpu):
        scaler = cuMinMaxScaler()
    else:
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

def expanding_window_split(X, y, cv=0, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for i, (train_index, test_index) in enumerate(tscv.split(X)):
        if i == cv:
            return X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
    return 0

def no_ml_predict(X: np.array, y: np.array):
    rmse = truncate_metric(mean_squared_error(X, y, squared=False))
    wape = truncate_metric(float(np.sum(np.abs(X - y)) / np.sum(np.abs(y))))
    r2 = truncate_metric(r2_score(X, y))
    
    print('RMSE: %.4f' % rmse)
    print('WAPE: %.2f' % (wape * 100))
    print('R2: %.4f' % r2)
    return rmse, wape, r2

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
    
def performance_metrics(preds, actuals, filename=None, gpu=False):

    # calculate performance metrics
    if(gpu):
        rmse = truncate_metric(mse_gpu(actuals, preds, squared=False))
        wape = truncate_metric(cupy.sum(cupy.abs(preds - actuals)) / cupy.sum(cupy.abs(actuals)) * 100)
        r2 = truncate_metric(r2_gpu(actuals, preds))
    else:
        rmse = truncate_metric(mean_squared_error(actuals, preds, squared=False))
        wape = truncate_metric(np.sum(np.abs(preds - actuals)) / np.sum(np.abs(actuals))) * 100
        r2 = truncate_metric(r2_score(actuals, preds))
    
    # print performance metrics
    if(filename != None):
        print('RMSE: %.4f' % rmse, file=filename)
        print('WAPE: %.2f' % wape, file=filename)
        print('R2: %.4f' % r2, file=filename)
    print('RMSE: %.4f' % rmse)
    print('WAPE: %.2f' % wape)
    print('R2: %.4f' % r2)
    return rmse, wape, r2

def past_timesteps(df, number_of_timesteps):
    df = df.sort_values(by=['Location', 'Time'])
    for i in tqdm(range(1, (number_of_timesteps + 1))):
        df.loc[df['Time'].shift(i) == df['Time'] - pd.Timedelta(i * 15, 'm'), f"lag_{i}"] = df['Energy'].shift(i)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def train_test_split_timeseries(df, train_size=0.8, cv=0):
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

def expanding_window_split_location(df, cv=0, n_splits=10):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    X_tr, X_te, y_tr, y_te = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    for h in df.Location.unique():
        house = df[df.Location == h]
        for i, (train_index, test_index) in enumerate(tscv.split(house)):
            if i == cv:
                X_train, X_test, y_train, y_test = house.iloc[train_index], house.iloc[test_index], house.iloc[train_index], house.iloc[test_index]
                if X_tr.empty:
                    X_tr, X_te, y_tr, y_te = X_train, X_test, y_train, y_test   
                else:
                    X_tr = pd.concat([X_tr,X_train], axis=0)
                    X_te = pd.concat([X_te,X_test], axis=0)
                    y_tr = pd.concat([y_tr,y_train], axis=0)
                    y_te = pd.concat([y_te,y_test], axis=0)
    return X_tr, X_te, y_tr, y_te

def test_leave_house_out(df, estimator, locations, filename, split_timeseries=False, train_size=0.8, gpu=False, cv=0):
    if(split_timeseries):
        print("split timeseries")
        X_train, X_test, y_train, y_test = expanding_window_split_location(df, cv=cv, n_splits=10)
        print("Train set: ", X_train.shape)
        print("Test set: ", X_test.shape)
        X_train = X_train.drop(['Time', 'Location', 'Energy'], axis=1)
        X_test = X_test.drop(['Time', 'Location', 'Energy'], axis=1)
        y_train = y_train['Energy']
        y_test = y_test['Energy']
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
    
    X_train_norm, scaler = normalize_training(X_train, gpu=gpu)
    X_test_norm = scaler.transform(X_test)
    model = estimator
    init = time.time()
    model.fit(X_train_norm, y_train)
    y_pred = model.predict(X_test_norm)
    end = time.time()
    print('Elapsed time: {:.4f} s'.format(end - init), file=filename)
    rmse, wape, r2 = performance_metrics(y_pred, y_test.values.reshape(-1), filename, gpu=gpu)
    return rmse, wape, r2, model

# Individual
def split_train_test_timeseries(X, y, train_size=0.8):
    n_train_samples = int(len(X) * train_size)
    X_train = X[:n_train_samples]
    X_test = X[n_train_samples:]
    y_train = y[:n_train_samples]
    y_test = y[n_train_samples:]
    return X_train, X_test, y_train, y_test

def total_averaged_metrics_individual(metrics_list):
    rmse, wape, r2 = 0,0,0,
    for house in metrics_list:
        rmse += sum(i for i, j, k in house)
        wape += sum(j for i, j, k in house)
        r2 += sum(k for i, j, k in house)
    t_rmse = np.round(rmse/len(metrics_list),3)
    t_wape = np.round(wape/len(metrics_list),3)
    t_r2 = np.round(r2/len(metrics_list),3)
    print("Total Averaged RMSE: {}".format(t_rmse))
    print("Total Averaged WAPE: {}".format(t_wape))
    print("Total Averaged R2: {}".format(t_r2))
    return t_rmse, t_wape, t_r2

def build_predict_show(df, number_timesteps, estimator, selected_features=None, normalize=False, train_size=0.8, start_timestep=1, cv=10):
    full_start = time.time()
    metrics_list = []
    
    for i in range(start_timestep,(number_timesteps + 1)):
        rmse_avg, wape_avg, r2_avg, discard = 0,0,0,0
        start = time.time()
        print("\nNumber of features ", i)
        if(selected_features != None):
            X, y = retrieve_selected_features(df, selected_features, "2019-01-01")
        else:
            X, y = last_energy_points(df, i)
        
        for j in range(cv):
            X_train, X_test, y_train, y_test = expanding_window_split(X,y, cv=j, n_splits=cv)
            #X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8, random_state=j*5)

            if(normalize):
                scaler = MinMaxScaler().fit(X_train)
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)
            
            model, preds = build_model(estimator, X_train, y_train, X_test)
            rmse, wape, r2 = performance_metrics(preds.reshape(-1), y_test.values.reshape(-1))
            if(rmse > 1):
                discard += 1
                continue
            rmse_avg += rmse
            wape_avg += wape
            r2_avg += r2
        metrics_list.append((rmse_avg/(cv-discard), wape_avg/(cv-discard),r2_avg/(cv-discard)))
        print("\nElapsed time: %.3f seconds" % (time.time() - start))
    print("\nFull Elapsed time: %.3f seconds" % (time.time() - full_start))
    return model, preds, metrics_list

def build_predict_show_rf(df, number_timesteps, estimator, selected_features=None, normalize=False, train_size=0.8, start_timestep=1, cv=5):
    full_start = time.time()
    metrics_list = []
    for i in range(start_timestep,(number_timesteps + 1)):
        rmse_avg, wape_avg, r2_avg, discard = 0,0,0,0
        start = time.time()
        print("\nNumber of features ", i)
        if(selected_features != None):
            X, y = retrieve_selected_features(df, selected_features, "2019-01-01")
        else:
            X, y = last_energy_points(df, i)
        for j in range(cv):
            X_train, X_test, y_train, y_test = expanding_window_split(X,y, cv=j, n_splits=5)
            
            if(normalize):
                scaler = MinMaxScaler().fit(X_train)
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)
            model, preds = build_model(estimator, X_train, y_train, X_test)
            rmse, wape, r2 = performance_metrics(pd.DataFrame(preds).values.reshape(-1), y_test.values.reshape(-1))
            if(rmse > 1):
                discard += 1
                continue
            rmse_avg += rmse
            wape_avg += wape
            r2_avg += r2
        metrics_list.append((rmse_avg/(cv-discard), wape_avg/(cv-discard),r2_avg/(cv-discard)))
        print("\nElapsed time: %.3f seconds" % (time.time() - start))
    print("\nFull Elapsed time: %.3f seconds" % (time.time() - full_start))
    return model, preds, metrics_list

def show_graphic_per_timestep(metrics_list, number_of_houses):
    rmse_list = []
    wape_list = []
    r2_list = []

    for i in range(0,len(metrics_list)):
        rmse_list.append(metrics_list[i][0][0])
        wape_list.append(metrics_list[i][0][1])
        r2_list.append(metrics_list[i][0][2])
        
    plt.plot(range(0,number_of_houses), rmse_list)
    plt.title('RMSE per house')
    plt.xlabel('Number of houses')
    plt.ylabel('RMSE')
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

def show_all_metrics_per_house(metrics_list, number_of_houses):
    rmse_list = []
    wape_list = []
    r2_list = []

    for i in range(0,len(metrics_list)):
        rmse_list.append(metrics_list[i][0][0])
        wape_list.append(metrics_list[i][0][1] / 100)
        r2_list.append(metrics_list[i][0][2])
        
    plt.plot(range(0, number_of_houses), rmse_list, label='RMSE')

    # Plot the second list
    plt.plot(range(0, number_of_houses), wape_list, label='WAPE')

    # Plot the third list
    plt.plot(range(0, number_of_houses), r2_list, label='R2')

    # Set the title, xlabel, and ylabel
    #plt.title('Performance Metrics per House')
    plt.xlabel('House')
    plt.ylabel('Performance')

    # Add a legend to differentiate the lines
    plt.legend()

    plt.yticks(np.arange(0, 1.1, 0.1))

    # Display the combined plot
    plt.show()

def add_energy_variation(df):
    # Create shifted series
    lag = df.Energy.shift(1).dropna()
    lag.reset_index(drop=True, inplace=True)
    
    # Cut first row due to no variation
    df_tmp = df.Energy.iloc[1:]
    df_tmp.reset_index(drop=True, inplace=True)
    
    # Divide actuals per last 15-minute interval, adding 0.00001 to avoid division by zero
    variation = np.divide(df_tmp,(lag + 0.00001))

    # Subtract by one to present the positive or negative difference on the energy
    variation_1 = np.subtract(variation, 1)

    # Insert first value as 0 as it doesn't have previous comparison
    variation_2 = pd.concat([pd.Series([0,0]), variation_1])
    variation_2.reset_index(drop=True, inplace=True)

    # Add Variation to Dataframe and relocate Energy to last column of df
    df["Variation"] = variation_2
    first_time, second_time = df.Time.iloc[0], df.Time.iloc[1]
    df.loc[df.Time == first_time, 'Variation'] = 0
    df.loc[df.Time == second_time, 'Variation'] = 0
    energy = df.Energy
    df.drop("Energy", axis=1, inplace=True)
    df["Energy"] = energy
    df.dropna(inplace=True)
    
    return df

class cuMinMaxScaler():
    def __init__(self):
        self.feature_range = (0,1)

    def _reset(self):

        if hasattr(self, 'scale_'):
            del self.scale_
            del self.min_

    def fit(self, X): #X is assumed to be a cuDF dataframe, no type checking

        self._reset()        

        X = X.dropna()

        data_min = X.min(axis = 0) #cuDF series
        data_max = X.max(axis = 0) #cuDF series

        data_range = data_max - data_min #cuDF series

        data_range[data_range==0] = 1 #replaced with 1 is range is 0

        feature_range = self.feature_range

        self.scale_ = (feature_range[1] - feature_range[0]) / data_range # element-wise divison, produces #cuDF series
        self.min_ = feature_range[0] - data_min * self.scale_ # element-wise multiplication, produces #cuDF series

        return self

    def transform(self, X):

        X *= self.scale_ # element-wise divison, match dataframe column to series index
        X += self.min_ # element-wise addition, match dataframe column to series index

        return X

 #["energy_lag_1", "energy_lag_2", "energy_lag_3", "energy_lag_4", "energy_lag_96", "energy_lag_192", "energy_lag_288", "energy_lag_384", "energy_lag_480", "energy_lag_576", "energy_lag_672", "DayOfWeek", "Hour"]