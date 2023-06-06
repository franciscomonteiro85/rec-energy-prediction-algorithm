import pandas as pd
import numpy as np

def select_past_timesteps(file_df, cluster, selected_columns=[]):
    df = pd.read_csv(file_df)
    if selected_columns:
        df_cluster = df[selected_columns]
        if 'Location' in selected_columns:
            df_cluster = df_cluster[df_cluster.Location.isin(cluster)]
    else:
        df_cluster = df[df.Location.isin(cluster)]
    return df_cluster
    

#cluster = select_past_timesteps("../data/porto_final_7days.csv", [5,39,44])
#print(cluster)

 #["energy_lag_1", "energy_lag_2", "energy_lag_3", "energy_lag_4", "energy_lag_96", "energy_lag_192", "energy_lag_288", "energy_lag_384", "energy_lag_480", "energy_lag_576", "energy_lag_672", "DayOfWeek", "Hour"]