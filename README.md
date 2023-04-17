[Português](README.pt-br.md)
# Energy Prediction Algorithm
## Preprocessing
In order to execute the available algorithms, first, one must create a folder to put the datasets.
By default the location to be read is "../datasets/".
After this, the Datasets need to be better organized and cleaned. 
The script "organize_by_location.py" should be executed to create a pickle with the data.
This script is under the "scripts" folder.
Example: `python3 organize_by_location.py ../../datasets/Dataset.xlsx ../data/porto.pkl 1/1/2019`

Some pickles are available in the data folder, feel free to use them.

## Execution

There are several notebooks in this repository. Let's describe what which one does.

#### Aggregated Porto and Shared
This Notebook creates the aggregated sum of each house consumption per time interval and runs a Linear and a Polynomial Regression model with different number of past records.

#### Algorithms Porto, Shared and Banes
Main notebooks to run the ML models. Here, the feature dataframe is created with the past timesteps of the energy consumption.
Then, the model is trained leaving 10 random houses for testing and the remaining for training. This is done 10 times, in order to cross validate the data and then averaged to find more accurate results. Also, the data is normalized between 0 and 1 to be compared with the other datasets. The ML models used are Linear Regression, XGBoost and Random Forest. By default, the output of is being written to a .txt file to the "gpu_logs" folder.

#### Dataset_Energy_Weather_Merging (Not Finished)
As of now, this notebook only resamples the meteorological data from 10-minute intervals to 15-minute intervals in order to then
merge with the energy consumption dataset.

#### Data_Visualization (Not Finished)
Some visualization of the data in different time periods (all year, weekly, monthly).

#### Graphics
Graphics comparing the different performance of the models. Created graphics are saved to the "images" folder.

#### Shared_Dataset_Creation
Aggregation of the multiple houses of the shared dataset into one file.

#### REC_Energy_Algorithm
First tests on the dataset, outdated and only uses one building to train the model.

#### REC_Without_ML
Tests on the Porto Dataset to see if simply assigning the previous 15-minute energy consumption value to the prediction does better or worse
than the models. Baseline to compare with the other models.

## Available data
Folder "data" contains multiple files (pickles and csv) of the used data. 
shared_1year.csv contains the Shared dataset with data of 7 houses' energy consumption over a year
shared_total.csv contains the same houses information, but for a longer period of time (some houses only some months others more).
porto.pkl contains the Porto Dataset organized by location (each location is a different house).
station pickles are weather data of Aveiro and Viseu to use together with the energy consumption data.

## Requirements
These notebooks are programmed to run on a GPU and to do so, they use some libraries. To run the XGBoost model, the XGBoost python library is used. For the Random Forest, the Rapids-AI cuML library is used, which provides a similar model to the scikit-learn library, but optimized for GPU.