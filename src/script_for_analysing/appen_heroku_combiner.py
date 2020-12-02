# !/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from heroku_functions import *
from appen_functions import *

# Set name of heroku and appen file
herokuFile  = 'entries_1.json'
appenFile   = 'f1669822.csv'
results_folder   ='/home/jim/HDDocuments/university/master/thesis/results/'


# Create heroku and appen class
h = heroku(herokuFile,results_folder)
a = appen(appenFile,results_folder)

# Find cheaters
a.find_cheaters('daniel') #'daniel' format or 'standard'

# Make CSV file of results
appenUniqueFile, appenCheaterFile   = a.makeCSV()
herokuResponseFile                  = h.makeCSV()

# Get pandas DataFrame from csv files
heroku_data = pd.read_csv(herokuResponseFile)
appen_data  = pd.read_csv(appenUniqueFile)

# Rename worker code column for merging
appen_data.rename(columns = {'type_the_code_that_you_received_at_the_end_of_the_experiment':'Meta:worker_code'}, inplace = True)

# Merger dataframes
merge_data = appen_data.merge(heroku_data, on = 'Meta:worker_code')

# Remove empty and old index columns
for key in merge_data.keys():
    if 'gold' in key:
        del merge_data[key]
    if 'Unnamed' in key:
        del merge_data[key]

# Save merged DataFrame as CSV
merge_data.to_csv(results_folder + 'filtered_responses/' + 'merged_data.csv')
print('There are %s unique responses' %len(merge_data))


