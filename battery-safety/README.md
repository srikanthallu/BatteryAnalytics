collection of ML algorithms for battery safety data
# Building the python environment

1. virtualenv -p python3 ~/Documents/virtualenv3/batteryanalytics
1. source ~/Documents/virtualenv3/batteryanalytics/bin/activate
1. pip install -r requirements.txt
1. python -m ipykernel install --user --name batteryanalytics --display-name "Battery Analytics"
1. jupyter lab
1. Select the battery analytics kernel for running the notebooks
