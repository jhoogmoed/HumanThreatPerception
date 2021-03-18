# !/usr/bin/env python3
import os
import shutil
import numpy as np
import random
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import cv2

from pandas.core.frame import DataFrame
from sklearn import decomposition, datasets
from sklearn.preprocessing import StandardScaler

class analyse:
    def __init__(self,dataPath,drive,mergedDataFile,modelDataFile,results_folder):   
        # Docstring    
        "Class container for analysing functions."
        self.modelDataFile = modelDataFile
        # Set paths
        self.dataPath       = dataPath
        self.drive          = drive
        self.results_folder = results_folder

        # Load data
        self.merge_data = pd.read_csv(mergedDataFile)
         
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
        # print('{:<25}'.format('autocorrelation') + ': R^2 =',f'{r2_fl:.5f}')
        print('{:<25}'.format('autocorrelation') + ': R^2 =',f'{r2_fl:.5f}')
        
        
        # Plot correlation of first and last half
        self.plot_correlation(self.response_mean_first,self.response_mean_last,
                              self.response_std_last,self.response_std_first,
                              'first_half','last_half','general_autocorrelation',round(r2_fl,5))
        
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
        self.model_data = pd.read_csv(self.results_folder + 'model_responses/' + self.modelDataFile)
        
        # Get keys
        self.parameter_keys = list(self.model_data)
        self.parameter_keys.remove('general_frame_number')
        self.model_data.pop('general_frame_number')
        
        
        for parameter in self.parameter_keys:
            # Get correlation
            r2 = self.model_data[parameter].corr(self.response_mean)**2
            
            # Print correlation
            print('{:<25}'.format(parameter) + ': R^2 =',f'{r2:.5f}')
            
            # Save figure correlation
            if plotBool == True:
                self.plot_correlation(self.model_data[parameter],self.response_mean,
                                    None,self.response_std,
                                    str(parameter),'response_mean',parameter,round(r2,5))
        
        # Add mean response to correlation matrix
        self.model_data['response_mean'] = self.response_mean

        # Get correlation matrix
        corrMatrix = self.model_data.corr(method='pearson')

        # corrMatrix = corrMatrix.sort_values(by='response_mean')
        
        # Remove uppper triangle
        mask = np.zeros_like(corrMatrix)
        mask[np.triu_indices_from(mask,k=1)] = True
        
        # Plot correlation matrix
        plt.clf()
        sn.heatmap(corrMatrix,vmax = 1,vmin = -1,cmap = 'RdBu_r', linewidths=.5, annot=True)       
        plt.show()
        
        r = self.model_data['model_combination'].corr(self.response_mean)
        return r**2

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
            
            plt.clf()
            plt.errorbar(series1, series2,std2,std1,linestyle = 'None', marker = '.',markeredgecolor = 'green')
            plt.title(name1+ ' vs. ' + name2 +  " | R^2 = %s" %r2)
            plt.xlabel(name1)
            plt.ylabel(name2)
        print(response_mean_sorted)

    def PCA(self):
        images = sorted(os.listdir(self.dataPath + self.drive+ '/image_02/data/'))

        images_features_gray = []
        images_features_blue = []
        images_features_green = []
        images_features_red = []
        
        for image in images:
            image_features_gray = []
            image_features_blue = []
            image_features_green = []
            image_features_red = []
            
            full_path = self.dataPath + self.drive+ '/image_02/data/' + image
            loaded_image = cv2.imread(full_path)

            gray = cv2.cvtColor(loaded_image, cv2.COLOR_BGR2GRAY)
            blue = loaded_image[:,:,0]
            green = loaded_image[:,:,1]
            red  = loaded_image[:,:,2]
            
            scaling = 1./2
            gray_scaled = cv2.resize(gray, (0, 0), fx=(scaling), fy=(scaling))
            blue_scaled = cv2.resize(blue, (0, 0), fx=(scaling), fy=(scaling))
            green_scaled = cv2.resize(green, (0, 0), fx=(scaling), fy=(scaling))
            red_scaled = cv2.resize(red, (0, 0), fx=(scaling), fy=(scaling))
            scaled_shape = gray_scaled.shape
            

            for horizontal in gray_scaled:
                image_features_gray = image_features_gray + list(horizontal)
            images_features_gray.append(image_features_gray)
            
            for horizontal in blue_scaled:
                image_features_blue = image_features_blue + list(horizontal)
            images_features_blue.append(image_features_blue)
            
            for horizontal in green_scaled:
                image_features_green = image_features_green + list(horizontal)
            images_features_green.append(image_features_green)
            
            for horizontal in red_scaled:
                image_features_red = image_features_red + list(horizontal)
            images_features_red.append(image_features_red)
        
        
        
        # PCA decomposition
        # nc = 25
        nc = 50
        pca = decomposition.PCA(n_components=nc)
        
        std_gray = StandardScaler()
        gray_std = std_gray.fit_transform(images_features_gray)
        gray_pca = pca.fit_transform(gray_std)    
        eigen_frames_gray = np.array(pca.components_.reshape((nc,scaled_shape[0],scaled_shape[1])))
        
        std_blue = StandardScaler()
        blue_std = std_blue.fit_transform(images_features_blue)
        blue_pca = pca.fit_transform(blue_std)   
        eigen_frames_blue = np.array(pca.components_.reshape((nc,scaled_shape[0],scaled_shape[1])))
        
        std_green = StandardScaler()
        green_std = std_green.fit_transform(images_features_green)
        green_pca = pca.fit_transform(green_std)  
        eigen_frames_green = np.array(pca.components_.reshape((nc,scaled_shape[0],scaled_shape[1])))
        
        std_red = StandardScaler()
        red_std = std_red.fit_transform(images_features_red)
        red_pca = pca.fit_transform(red_std)    
        eigen_frames_red = np.array(pca.components_.reshape((nc,scaled_shape[0],scaled_shape[1])))        
         
        
        # Back tranform for check
        back_transform = pca.inverse_transform(gray_pca)
        back_transform_renormalize = std_gray.inverse_transform(back_transform)
        
        # Show before and after
        first_image = np.array(images_features[0]).reshape(scaled_shape)
        cv2.imshow('Before PCA',first_image)
        cv2.waitKey(0)  

        # second_image = np.array(back_transform_renormalize[0]).reshape(scaled_shape)
        cv2.imshow('After PCA',second_image)
        cv2.waitKey(0)  
        
        
        
       
 
        gray_pca_df = pd.DataFrame(gray_pca)
        blue_pca_df = pd.DataFrame(blue_pca)
        green_pca_df = pd.DataFrame(green_pca)
        red_pca_df = pd.DataFrame(red_pca)
        
        for i in range(0,nc):
            print('Feature: ',i)
            print('Gray correlation: ', gray_pca_df[i].corr(self.response_mean))
            print('Blue correlation: ', blue_pca_df[i].corr(self.response_mean))
            print('Green correlation: ', green_pca_df[i].corr(self.response_mean))
            print('Red correlation: ', red_pca_df[i].corr(self.response_mean))
            max_pixel_gray = np.max(abs(eigen_frames_gray[i]))
            max_pixel_blue = np.max(abs(eigen_frames_blue[i]))
            max_pixel_green = np.max(abs(eigen_frames_green[i]))
            max_pixel_red = np.max(abs(eigen_frames_red[i]))
            # cv2.imshow(('Feature: '+str(i)+' Eigenface'),eigen_frames[i]* (1/max_pixel))
            # cv2.waitKey(0)  
            # cv2.imwrite(os.path.join(self.results_folder,'pca',('gray' + str(i)+'.png')),((eigen_frames_gray[i])*1/max_pixel_gray)*255)         
            # cv2.imwrite(os.path.join(self.results_folder,'pca',('blue' + str(i)+'.png')),((eigen_frames[i])*1/max_pixel_blue)*255)      
            # cv2.imwrite(os.path.join(self.results_folder,'pca',('green' + str(i)+'.png')),((eigen_frames[i])*1/max_pixel_green)*255)      
            # cv2.imwrite(os.path.join(self.results_folder,'pca',('red' + str(i)+'.png')),((eigen_frames[i])*1/max_pixel_red)*255)      
            gray_channel = eigen_frames_gray[i]*1/max_pixel_gray*255
            blue_channel = eigen_frames_blue[i]*1/max_pixel_blue*255
            green_channel = eigen_frames_green[i]*1/max_pixel_green*255
            red_channel = eigen_frames_red[i]*1/max_pixel_red*255
            
            bgr_image = np.zeros((scaled_shape[0],scaled_shape[1],3)) 
            bgr_image[:,:,0] = blue_channel
            bgr_image[:,:,1] = green_channel
            bgr_image[:,:,2] = red_channel

            cv2.imwrite(os.path.join(self.results_folder,'pca',('color ' + str(i)+'.png')),bgr_image)
            cv2.imwrite(os.path.join(self.results_folder,'pca',('gray ' + str(i)+'.png')),gray_channel)
            cv2.imwrite(os.path.join(self.results_folder,'pca',('blue ' + str(i)+'.png')),blue_channel)
            cv2.imwrite(os.path.join(self.results_folder,'pca',('green' + str(i)+'.png')),green_channel)
            cv2.imwrite(os.path.join(self.results_folder,'pca',('red ' + str(i)+'.png')),red_channel)


    def risk_accidents(self):
        # Get accident answers
        accident_occurence = self.merge_data['how_many_accidents_were_you_involved_in_when_driving_a_car_in_the_last_3_years_please_include_all_accidents_regardless_of_how_they_were_caused_how_slight_they_were_or_where_they_happened']
        
        # Filter no responses
        accident_occurence = [np.nan if value == 'i_prefer_not_to_respond' else value for value in accident_occurence]
        accident_occurence = [6 if value == 'more_than_5' else value for value in accident_occurence]
        accident_occurence = [value if value == np.nan else float(value) for value in accident_occurence]
        
        # Get corresponding average risk score
        average_score = list(self.response_data.mean(axis=1))
        risk_accidents = pd.DataFrame({'Accidents':accident_occurence,'Average_score':average_score})
        print(risk_accidents)
        r = risk_accidents.corr()
        print(r)

        # print(average_score)
        
        # print(accident_occurence)
                                                                                                                    

    def cronbach_alpha(self,df):    # 1. Transform the df into a correlation matrix
        df_corr = df.corr()
        
        # 2.1 Calculate N
        # The number of variables equals the number of columns in the df
        N = df.shape[1]
        
        # 2.2 Calculate R
        # For this, we'll loop through the columns and append every
        # relevant correlation to an array calles "r_s". Then, we'll
        # calculate the mean of "r_s"
        rs = np.array([])
        for i, col in enumerate(df_corr.columns):
            sum_ = df_corr[col][i+1:].values
            rs = np.append(sum_, rs)
        mean_r = np.mean(rs)
    
    # 3. Use the formula to calculate Cronbach's Alpha 
        cronbach_alpha = (N * mean_r) / (1 + (N - 1) * mean_r)
        print(cronbach_alpha)
    
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
    resultsFolder  = '/home/jim/HDDocuments/university/master/thesis/results/'
    mergedDataFile  = '/home/jim/HDDocuments/university/master/thesis/results/filtered_responses/merged_data.csv'
    modelDataFile   = 'model_results.csv'
    
    analyse = analyse(dataPath,drive,mergedDataFile,modelDataFile,resultsFolder)
    analyse.get_responses()
    analyse.info()
    # analyse.split()
    # analyse.model()
    # analyse.risk_ranking()
    analyse.PCA()
    # analyse.plot_correlation(analyse.model_data['road_road'],analyse.model_data['general_velocity'])
    # analyse.risk_accidents()
    # analyse.cronbach_alpha(analyse.response_data)
    