# !/usr/bin/env python

import os
import shutil
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt


import cv2
from pandas.core.frame import DataFrame
from sklearn import decomposition, datasets
from sklearn.preprocessing import StandardScaler

class analyse:
    def __init__(self,dataPath,drive,mergedDataPath,modelDataPath,results_folder):   
        # Docstring    
        "Class container for analysing functions."
        
        # Set paths
        self.dataPath       = dataPath
        self.drive          = drive
        self.results_folder = results_folder

        # Load data
        self.merge_data = pd.read_csv(self.results_folder + 'filtered_responses/' + mergedDataPath)
        self.model_data = pd.read_csv(self.results_folder + 'model_responses/' + modelDataPath)
        
    
    def get_responses(self):
        # Get response per frame
        indices = []
        all_responses = []
        for key in self.merge_data.keys():
            for stimulus_index in range (0, len(self.merge_data.keys())):
                response_column = 'Stimulus %s: response' %stimulus_index
                if response_column in key:
                    indices.append(int(stimulus_index))
                    all_responses.append(self.merge_data[response_column])

        # Create response DataFrame
        self.response_data   = pd.DataFrame(all_responses, index = indices)
        self.response_data.sort_index(inplace = True)
        self.response_data.columns = self.merge_data['Meta:worker_code']
        self.response_data   = self.response_data.transpose()
        
        # Get mean and std of responses
        self.response_mean   = self.response_data.mean(skipna = True)
        self.response_std    = self.response_data.std(skipna = True)  
    
    def find_outliers(self,thresh):
        # Find outliers
        bad_indices = []
        for index in self.response_data.index:
            if(self.response_data.loc[index].std(skipna = True)<thresh):
                bad_indices.append(index)
        self.response_data.pop(ba)
        
    def info(self):
        # Get general info on data
        self.data_description = self.response_data.describe()
        self.data_description.to_csv(self.results_folder + 'filtered_responses/' + 'self.data_description.csv')
        # print(self.data_description)

    def split(self):
        # Split data
        middle = int(round(len(self.response_data)/2))

        self.response_data_first = self.response_data[0:middle]
        self.response_data_last = self.response_data[(middle+1):len(self.response_data)]

        self.response_mean_first   = self.response_data_first.mean(skipna = True)
        self.response_std_first    = self.response_data_first.std(skipna = True)

        self.response_mean_last   = self.response_data_last.mean(skipna = True)
        self.response_std_last    = self.response_data_last.std(skipna = True)

        r_fl = self.response_mean_first.corr(self.response_mean_last)
        r2_fl = r_fl*r_fl
        print('{:<30}'.format('Autocorrelation') + ': 1Half vs 2Half R^2 = %s' %r2_fl)

        fig_fl = plt.errorbar(self.response_mean_first, self.response_mean_last, self.response_std_first,self.response_std_last,linestyle = 'None', marker = '.',markeredgecolor = 'green')
        plt.title("First half vs. Last half | R^2 = %s" %r2_fl)
        plt.xlabel("First half")
        plt.ylabel("Second half")
        plt.savefig(self.results_folder + 'correlation_images/' + 'first_half_vs_last_half.png')
        
        # Random person R^2
        random_person = random.randrange(0,len(self.response_data),1)
        self.single_response = self.response_data.iloc[random_person]
        
        r_sm = self.single_response.corr(self.response_mean)
        r2_sm = r_sm*r_sm
        print('{:<30}'.format('Single random') + ': N' + '{:<4}'.format(str(random_person)) +  " vs Human R^2 = " + str(r2_sm))            
        
        plt.clf()
        fig_sm = plt.errorbar(self.single_response, self.response_mean, self.response_std,linestyle = 'None', marker = '.',markeredgecolor = 'green')

        plt.title("Single random " + str(random_person) + " vs. Mean responses | R^2 = %s" %r2_sm)
        plt.xlabel("Single random person")
        plt.ylabel("Mean responses")
        plt.savefig(self.results_folder + 'correlation_images/' + 'single_vs_mean.png')

        # d = {'single':self.single_response,'mean':self.response_mean}
        # df = pd.DataFrame(d)
        # print(df)      

    def model(self):
        # Get determinant of correlation
        for parameter in self.model_data.keys()[1::]:
            self.r = self.model_data[parameter].corr(self.response_mean)
            self.r2 = self.r*self.r
            print('{:<30}'.format(parameter) + ': Model vs Human R^2 = %s' %self.r2)
            
            plt.clf()
            fig_mr = plt.errorbar(self.model_data[parameter], self.response_mean,self.response_std,linestyle = 'None', marker = '.',markeredgecolor = 'green')
            plt.title("Model vs. responses " + parameter + " | R^2 = %s" %self.r2)
            plt.xlabel("Model response")
            plt.ylabel("Human response")


            # Create linear fit of model and responses
            linear_model = np.polyfit(self.model_data[parameter], self.response_mean, 1)
            linear_model_fn = np.poly1d(linear_model)
            x_s = np.arange(0, self.model_data[parameter].max())
            fig_fl = plt.plot(x_s,linear_model_fn(x_s),color="green")

            plt.savefig(self.results_folder + 'correlation_images/' + 'model_vs_human_' + parameter + '.png')           

    def risky_images(self):
        # Get most risky and least risky images
        response_mean_sorted = self.response_mean.sort_values()
        least_risky = response_mean_sorted.index[0:5]
        most_risky = response_mean_sorted.tail(5).index[::-1]

        # Save most and least risky images
        i = 1
        for image in least_risky:
            # os.path.join(self.dataPath + self.drive + '/image_02/data')
            shutil.copyfile(self.dataPath+ self.drive+ '/image_02/data/' +str(image)+ '.png', self.results_folder +'most_least_risky_images/' +'least_risky_%s.png' %i)
            i+=1
        i = 1
        for image in most_risky:
            # os.path.join(self.dataPath + self.drive + '/image_02/data')
            shutil.copyfile(self.dataPath+ self.drive+ '/image_02/data/' +str(image)+ '.png', self.results_folder +'most_least_risky_images/' +'most_risky_%s.png' %i)
            i+=1
                      
    def risk_ranking(self):
        response_mean_sorted = self.response_mean.sort_values()
        i = 0
        for image in response_mean_sorted.index:
            shutil.copyfile(self.dataPath+ self.drive+ '/image_02/data/' +str(image)+ '.png', self.results_folder +'risk_sorted_images/' + '%s.png' %i)
            i+=1

    def PCA(self):
        images = os.listdir(self.dataPath + self.drive+ '/image_02/data/')

        images_features = []
        for image in images:
            image_features = []
            full_path = self.dataPath + self.drive+ '/image_02/data/' + image
            loaded_image = cv2.imread(full_path)
            gray = cv2.cvtColor(loaded_image, cv2.COLOR_BGR2GRAY)
            gray_1_16 = cv2.resize(gray, (0, 0), fx=(1./16), fy=(1./16))
            for horizontal in gray_1_16:
                image_features = image_features + list(horizontal)
            images_features.append(image_features)

        std_slc = StandardScaler()
        X_std = std_slc.fit_transform(images_features)

        print(X_std.shape)
        print(X_std)

        pca = decomposition.PCA(n_components=4)

        X_std_pca = pca.fit_transform(X_std)

        print(X_std_pca.shape)
        print(X_std_pca)

        #1. load images as long lists of gray pixel values 
        # self.dataPath + self.drive+ '/image_02/data/' +str(image)+ '.png'
        # dataset = datasets.load_breast_cancer()

        # print(self.response_mean)
        
        

if __name__ == "__main__":
    dataPath        = '/home/jim/HDDocuments/university/master/thesis/ROS/data/2011_09_26'
    drive           = '/test_images'
    results_folder  = '/home/jim/HDDocuments/university/master/thesis/results/'
    mergedDataPath  = 'merged_data.csv'
    modelDataPath   = 'model_results.csv'
    
    data = analyse(dataPath,drive,mergedDataPath,modelDataPath,results_folder)
    data.get_responses()
    # data.find_outliers(10)
    data.info()
    data.split()
    data.model()
    # data.risky_images()
    # data.risk_ranking()