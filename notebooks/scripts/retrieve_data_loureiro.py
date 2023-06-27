import requests
import pandas as pd
import json
import numpy as np
from tqdm import tqdm
import re
from datetime import datetime

def login():
    url = "https://api.comsolve.pt/internal/login"
    headers = {'Accept': 'application/json'}
    body = {
        "empresa": "CSCMNDDS",
        "language": "pt-PT",
        "username": "comsolve@digitalmente.net",
        "password": "COMSOLVE!2022"
    }

    r = requests.post(url=url, json=body, headers=headers)
    return r.json()

def retrieve_data(url, headers, body):
    r = requests.post(url=url, json=body, headers=headers)
    return r

def get_datetime_dataframe(json_obj):
    dia = json_obj[0]["dia"]
    hora = re.sub('\D', '', json_obj[0]["hora"][:2])
    minuto = json_obj[0]["minuto"][:2].strip()
    dia_fim = json_obj[-1]["dia"]
    hora_fim = re.sub('\D', '', json_obj[-1]["hora"][:2])
    minuto_fim = json_obj[-1]["minuto"][:2].strip()
    data_inicio = dia + " " + hora + ":" + (minuto if len(minuto) == 2 else minuto + minuto) + ":" + "00"
    data_fim = dia_fim + " " + hora_fim + ":" + minuto_fim + ":" + "00"
    datetime_inicio = datetime.strptime(data_inicio, '%Y-%m-%d %H:%M:%S')
    datetime_fim = datetime.strptime(data_fim, '%Y-%m-%d %H:%M:%S')
    datetime_index = pd.date_range(start=datetime_inicio, end=datetime_fim, freq="15T")
    
    return datetime_index

def create_full_datetime_dataframe(json_obj):
    dictio = dict()
    for i in json_obj:
        dia = i["dia"]
        hora = re.sub('\D', '', i["hora"][:2])
        minuto = i["minuto"][:2].strip()
        data = dia + " " + hora + ":" + (minuto if len(minuto) == 2 else minuto + minuto) + ":" + "00"
        dictio[data] = i['energiaEntrada']
    return dictio

def join_dataframes(energy):
    first = 1
    for df2 in energy:
        if first:

            df1 = df2
            first = 0
        else:
            df1 = pd.merge(df1, df2, on='Time', how='outer')
            df1 = df1.sort_values(by="Time")
            df1.reset_index(drop=True, inplace=True)

    return df1

response = login()
token = response["token"]
token = "Bearer " + token
headers = {'Accept': 'application/json', 'Authorization': token}
url = "https://api.comsolve.pt/internal/curvas/getConsumosPerdasCurvasCargaPT"

body = {
    "dataInicio": "2022-01-01",
    "dataFim": "2023-06-30",
    "cpe": "anonymous_1"
}
energy = []
for i in tqdm(range(1,173)):
    body["cpe"] = "anonymous_" + str(i)
    response = retrieve_data(url, headers, body)
    series = []
    cpe = "CPE_" + str(i)
    json_obj = response.json()[cpe]["consumos"]

    series = create_full_datetime_dataframe(json_obj)
    series = pd.DataFrame.from_dict(series, orient="index")
    series["Time"] = series.index
    series = series.iloc[:,[1,0]]
    series = series.rename(columns={0: "Energy_{}".format(str(i))})
    series.reset_index(drop=True, inplace=True)
    energy.append(series)

merged_df = join_dataframes(energy)

merged_df.to_csv("merged_loureiro.csv", index=None)

#Number of houses with 25000 plus measurements: 134
#Number of houses with 30000 plus measurements: 112
#Number of houses with 35000 plus measurements: 81