# Will update this as the there is progress in the project

The [posenet git repo](https://github.com/rwightman/posenet-python)<br>

# Usage:
After pulling data from this repo to create a csv of your data,
1. Replace existing videos from "Data" folder
1. Optionally - delete "entire_data.csv" in the root directory, or your data will just be appended to it after running "Week 3.ipynb"
1. Similarly you may choose to delete "cumulative.csv" in the root directory. The "Week 4.ipynb" will read from "entire_data.csv" and prepare it for Machine Learning

# Log
June 28:
- Created notebook to read videos from [data](data) folder and create [entire_data.csv](entire_data.csv) with the skeleton points along with frame number and action

July 4:
- Created notebook to read data from [entire_data.csv](entire_data.csv) and transform it into standard format for machine learning [cumulative.csv](cumulative.csv)

