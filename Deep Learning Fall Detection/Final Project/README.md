# Fall Prediction using Deep Learning using Posenet

The [posenet git repo](https://github.com/rwightman/posenet-python)<br>

# Usage:
After pulling from this repo
1. pip install -r requirements.txt
1. (Optional) 
    1. Replace or add videos in "data" folder
    1. Delete "Cumulative.csv" and "entire_data.csv"1
    1. run => python pipeline.py
1. Run all cells in "Live Prediction Trial.ipynb"

# Log
June 28:
- Created notebook (Week 3) to read videos from [data](data) folder and create [entire_data.csv](entire_data.csv) with the skeleton points along with frame number and action

July 4:
- Created notebook (Week 4 part 1) to read data from [entire_data.csv](entire_data.csv) and transform it into standard format for machine learning [cumulative.csv](cumulative.csv)

July 15:
- Created noteook (Week 4 part 2) where Random Forest Classifier is used as benchmark for development of Deep Learning models.
- Added ANN model (more details in the notebook)

July 26:
- Updated noteook (Week 4 part 2) added CNN, RNN and LSTM.

July 31:
- Updated file with pipeline.py to create csv and replace existing Deep Learning models 
