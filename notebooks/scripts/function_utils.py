import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler

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

def performance_metrics(preds: np.array, actuals: np.array):

    # calculate performance metrics
    
    mse = mean_squared_error(actuals, preds)
    rmse = mean_squared_error(actuals, preds, squared=False)
    rmse_percentage = (rmse / (max(actuals) - min(actuals))) * 100
    rmse_percentage_std = (rmse / np.std(actuals)) * 100
    #wape = np.sum(np.abs(preds - actuals)) / np.sum(np.abs(actuals)) * 100
    wape = mean_absolute_error(actuals, preds) / actuals.mean() * 100
    r2 = r2_score(actuals, preds)

    # print performance metrics
    print('MSE: %.4f' % mse)
    print('RMSE: %.4f' % rmse)
    print('RMSE2: %.4f' % rmse_percentage)
    print('RMSE3: %.4f' % rmse_percentage_std)
    print('WAPE: %.4f' % wape)
    print('R2: %.4f' % r2)
    return mse, wape, r2

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
    mse, wape, r2 = performance_metrics(preds, y_test.values.reshape(-1))
    return mse, wape, r2

def last_energy_points(df, number_timesteps):
    X_total = pd.DataFrame()
    for i in range(1, (number_timesteps + 1) ):
        X_total[f'Energy_{i}'] = df.shift(i)
    X_total.dropna(inplace=True)
    X_total.reset_index(drop=True, inplace=True)
    y_total = pd.DataFrame(df[number_timesteps:])
    y_total.reset_index(drop=True, inplace=True)
    y_total.columns = ["Energy"]
    return X_total, y_total

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
        X[f'Energy_{i}'] = df.shift(i)
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
    y = df["Energy"]
    y = y.iloc[selected_features[-1]:]
    y.reset_index(drop=True, inplace=True)
    return X,y

# Individual
def split_train_test_timeseries(X, y, train_size=0.8):
    n_train_samples = int(len(X) * train_size)
    X_train = X[:n_train_samples]
    X_test = X[n_train_samples:]
    y_train = y[:n_train_samples]
    y_test = y[n_train_samples:]
    return X_train, X_test, y_train, y_test




 #["energy_lag_1", "energy_lag_2", "energy_lag_3", "energy_lag_4", "energy_lag_96", "energy_lag_192", "energy_lag_288", "energy_lag_384", "energy_lag_480", "energy_lag_576", "energy_lag_672", "DayOfWeek", "Hour"]