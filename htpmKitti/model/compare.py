# !/usr/bin/env python3
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
        self.modelDataPath = modelDataPath
        # Set paths
        self.dataPath       = dataPath
        self.drive          = drive
        self.results_folder = results_folder

        # Load data
        self.merge_data = pd.read_csv(self.results_folder + 'filtered_responses/' + mergedDataPath)
         
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
        
        # Get normalized response
        self.response_normal = (self.response_data-self.response_mean.mean())/self.response_std.mean()
        
        # Get researchers own responses
        # self.response_own =      
    
    def find_outliers(self,thresh):
        # Find outliers
        bad_indices = []
        for index in self.response_data.index:
            if(self.response_data.loc[index].std(skipna = True)<thresh):
                bad_indices.append(index)
                self.response_data =self.response_data.drop(index)
        
    def info(self):
        # Get mean and std of responses
        self.response_mean   = self.response_data.mean(skipna = True)
        self.response_std    = self.response_data.std(skipna = True)  
        
        # Get general info on data
        self.data_description = self.response_data.describe()
        self.data_description.to_csv(self.results_folder + 'filtered_responses/' + 'self.data_description.csv')
        # print(self.data_description)

    def split(self,useNorm = False):
        # Chose usage of normalized data
        if useNorm == True:
            data = self.response_normal 
        else:
            data = self.response_data
        
        # Get mean and std of responses
        self.response_mean   = data.mean(skipna = True)
        self.response_std    = data.std(skipna = True) 
        
        # Find middle
        middle = int(round(len(self.response_data)/2))

        # Get first half of data
        self.response_data_first    = data[0:middle]
        self.response_mean_first    = self.response_data_first.mean(skipna = True)
        self.response_std_first     = self.response_data_first.std(skipna = True)

        # Get last half of data
        self.response_data_last = data[(middle+1):len(data)]
        self.response_mean_last   = self.response_data_last.mean(skipna = True)
        self.response_std_last    = self.response_data_last.std(skipna = True)

        # Get correlation of first and last half
        r_fl = self.response_mean_first.corr(self.response_mean_last)
        r2_fl = r_fl*r_fl
        print('{:<30}'.format('Autocorrelation') + ': 1Half vs 2Half R^2 = %s' %r2_fl)
        
        # Plot correlation of first and last half
        self.plot_correlation(self.response_mean_first,self.response_mean_last,
                              self.response_std_last,self.response_std_first,
                              'First half','Last Half','general_autocorrelation',r2_fl)
        
    def random(self,useNorm=False,seed=100):
        # Chose usage of normalized data
        if useNorm == True:
            data = self.response_normal 
        else:
            data = self.response_data
            
           # Get mean and std of responses
        data_response_mean   = data.mean(skipna = True)
        data_response_std    = data.std(skipna = True) 
    
        # Chose random person 
        random.seed(seed)
        random_person = random.randrange(0,len(data),1)
        self.single_response = data.iloc[random_person]
        
        # Get correlation of data and random person
        r_sm = self.single_response.corr(data_response_mean)
        r2_sm = r_sm*r_sm
        print('{:<30}'.format('Single random') + ': N' + '{:<4}'.format(str(random_person)) +  " vs Human R^2 = " + str(r2_sm))     
        
        # Plot correlation of data and random person
        self.plot_correlation(self.single_response,data_response_mean,
                              data_response_std,[],
                              ('Person n'+str(random_person)),'Response mean','random',r2_sm) 

    def model(self,plotBool=True):
        self.model_data = pd.read_csv(self.results_folder + 'model_responses/' + self.modelDataPath)

        # Get determinant of correlation
        self.parameter_keys = list(self.model_data)
        self.parameter_keys.pop(0)
        
        for parameter in self.parameter_keys:
            # Get correlation
            r2 = self.model_data[parameter].corr(self.response_mean)**2
            
            # Print correlation
            print('{:<30}'.format(parameter) + ': Model vs Human R^2 = %s' %r2)
            
            # Save figure correlation
            if plotBool == True:
                self.plot_correlation(self.model_data[parameter],self.response_mean,
                                    None,self.response_std,
                                    str(parameter),'response_mean',parameter,r2)
                 
        r = self.model_data[self.parameter_keys[0]].corr(self.response_mean)

        return r*r

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
    
    def plot_correlation(self,series1,series2,
                         std1 = None,
                         std2 = None,
                         name1 = 'Series 1', name2 = 'Series 2',
                         parameter = 'Parameter',r2 = np.nan):
        
        # Plot errobar figure
        plt.clf()
        plt.errorbar(series1, series2,std2,std1,linestyle = 'None', marker = '.',markeredgecolor = 'green')
        plt.title(name1+ ' vs. ' + name2 +  " | R^2 = %s" %r2)
        plt.xlabel(name1)
        plt.ylabel(name2)

        
        # Create linear fit of model and responses 
        linear_model = np.polyfit(series1,series2, 1)
        linear_model_fn = np.poly1d(linear_model)
        x_s = np.arange(series1.min(), series1.max(),((series1.max()-series1.min())/1000))
        
        # Plot linear fit
        plt.plot(x_s,linear_model_fn(x_s),color="red")

        # Save figure
        plt.savefig(self.results_folder + 'correlation_images/' + parameter + '.png')        
        
        

if __name__ == "__main__":
    dataPath        = '/home/jim/HDDocuments/university/master/thesis/ROS/data/2011_09_26'
    drive           = '/test_images'
    results_folder  = '/home/jim/HDDocuments/university/master/thesis/results/'
    mergedDataPath  = 'merged_data.csv'
    modelDataPath   = 'model_results.csv'
    
    data = analyse(dataPath,drive,mergedDataPath,modelDataPath,results_folder)
    data.get_responses()
    data.info()
    data.split()
    data.model()
    # print(len(data.response_data))
    # data.find_outliers(20)
    # print(len(data.response_data))
    # data.split()
    # data.model()
    # data.risky_images()
    # data.risk_ranking()