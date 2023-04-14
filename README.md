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

#### Aggregated Porto
This Notebook create the aggregated sum of each
