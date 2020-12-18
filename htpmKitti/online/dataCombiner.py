#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from online.herokuFunctions import heroku
from online.appenFunctions import appen

class combiner:
    def __init__(self,herokuFile,appenFile,resultsFolder):
        self.herokuFile     = herokuFile
        self.appenFile      = appenFile
        self.resultsFolder  = resultsFolder
        self.mergeDataFile = self.resultsFolder + 'filtered_responses/' + 'merged_data.csv'
        
    def combine(self,reportFormat = 'daniel'):
        # Create heroku and appen class
        h = heroku(self.herokuFile,self.resultsFolder)
        a = appen(self.appenFile,self.resultsFolder)

        # Find cheaters
        a.find_cheaters(reportFormat) #'daniel' format or 'standard'

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
        
        merge_data.to_csv(self.mergeDataFile)
        print('There are %s unique responses' %len(merge_data))
        return appenUniqueFile, herokuResponseFile,self.mergeDataFile

if __name__ == "__main__":
    herokuFile  = 'entries_1.json'
    appenFile   = 'f1669822.csv'
    resultsFolder   ='/home/jim/HDDocuments/university/master/thesis/results/'
    c = combiner(herokuFile,appenFile,resultsFolder)
    c.combine()
    
