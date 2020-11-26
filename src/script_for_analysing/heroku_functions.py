# !/usr/bin/env python

import os
import re
import json
import numpy as np
import pandas as pd

class heroku:
    def __init__(self, jsonFile,results_folder):
        self.results_folder = results_folder
        self.jsonFile = jsonFile
        
        # Set total number of stimuli in pool
        self.total_stimuli_number = 210

        # Open json file 
        f = open(self.results_folder + jsonFile, 'r')
        self.data = f.readlines()
        f.close()       
        
        # Remove different data from beginning  
        self.data = self.data[19::]

        # Get number of participants
        self.data_length = len(self.data)
        
        
        # Get number of stimuli
        # self.data_single        = json.loads(self.data[1])
        # self.number_stimuli     = len(self.data_single["data"])-1    
        
        # Create total python dict of responses
        self.dict_all           = {}
        meta_keys               = [u'rt', u'trial_type', u'browser_user_agent', u'browser_major_version', u'stimulus', u'time_elapsed', u'browser_name', u'image_id', u'internal_node_id', u'key_press', u'browser_full_version', u'browser_app_name', u'trial_index','worker_code']
        data_keys               = [u'rt', u'trial_type', u'slider_start', u'stimulus', u'time_elapsed', u'internal_node_id','worker_code', u'response', u'trial_index']

        # print(meta_keys)
        # print(data_keys)
        self.data_single = json.loads(self.data[101])      
        
        # Loop over participants
        for i in range(0, self.data_length):
        # for i in range(0, 10):
            worker_code = None

            # Load single participant data
            self.data_single = json.loads(self.data[i])      
            
            # Create python dictionary for storing
            dict_single = {}

            if  self.data_single['data'][0].keys() == meta_keys:
                for meta_key in meta_keys:
                    dict_single['Meta:'+meta_key] = self.data_single['data'][0][meta_key]
                worker_code = dict_single.pop('Meta:worker_code')
            elif self.data_single['data'][0].keys() == data_keys:
                for data_key in data_keys:
                        stimulus_string = self.data_single['data'][0]['stimulus']
                        stimulus_number = int(re.split('\.|/', stimulus_string)[2])
                        stimulus_key = 'Stimulus ' + str(stimulus_number) + ': ' +data_key
                        dict_single[stimulus_key] = self.data_single['data'][0][data_key]
                worker_code = dict_single.pop(('Stimulus ' + str(stimulus_number) + ': ' +'worker_code'))
            dict_single['Meta:worker_code'] = worker_code

            try: 
                self.dict_all[worker_code].update(dict_single)
            except: 
                self.dict_all[worker_code] = dict_single 

            # Get responses per frame
            # for j in range(1, self.number_stimuli):
            #     stimulus_string = self.data_single['data'][j]['stimulus']
            #     stimulus_number = int(re.split('\.|/', stimulus_string)[2])
            #     for data_key in data_keys:
            #         dict_single['Stimulus ' + str(stimulus_number) + ': ' +data_key] = self.data_single['data'][j][data_key]

            # self.dict_all.append(dict_single)

        self.heroku_data = pd.DataFrame(self.dict_all)
        self.heroku_data = self.heroku_data.transpose()             
        

            
    def makeCSV(self):
        heroku_name = self.jsonFile.split('.')[0]
        self.heroku_data.to_csv(self.results_folder + heroku_name + '_responses.csv')
        return self.results_folder + heroku_name + '_responses.csv'
        

if __name__ == "__main__":
    results_folder   ='/home/jim/HDDocuments/university/master/thesis/results/'
    herokuFile   = 'entries_1.json'
    h = heroku(herokuFile,results_folder)
    h.makeCSV()