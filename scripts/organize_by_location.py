import pandas as pd
import sys
import datetime

# 1st arg - file to read
# 2nd arg - name of pickle file to create
# 3rd arg - start date of the records (format: dd/mm/yyyy)

def verify_date(start_date):
    try:
        datetime.datetime.strptime(start_date,"%-d/%-m/%Y")
    except ValueError as err:
        raise Exception("Date is not valid")
    

def main():
    file = sys.argv[1]
    writefile = sys.argv[2]
    if len(sys.argv) == 4:
        start_date = sys.argv[3]

    if(file.endswith('csv')):
        df = pd.read_csv(file)
    elif(file.endswith('xlsx')):
        df = pd.read_excel(file)
    else:
        raise Exception("File format not supported")
    if 'Time' in df.columns:
        dateindex = df['Time']
        df.drop(['Time'], axis=1, inplace=True)
    else:
        num_samples_per_house = df.shape[0]
        dateindex = pd.date_range(start_date, periods=num_samples_per_house, freq='15T', name='Time').to_frame(index=False)

    house_list = []
    test = dateindex
    for i in df:
        test = pd.concat([dateindex, df[i]], axis=1)
        test = test.rename(columns={i: 'Energy'})
        test['Location'] = i
        house_list.append(test)
    df_location = house_list[0]
    for i in house_list[1:]:
        df_location = pd.concat([df_location, i], axis=0)
    df_location.reset_index(drop=True, inplace=True)
    df_location.to_pickle(writefile)
    print(df_location.shape)

if __name__ == "__main__":
    main()