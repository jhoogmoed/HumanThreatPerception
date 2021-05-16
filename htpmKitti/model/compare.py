# !/usr/bin/env python3
from math import isnan
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
from sklearn.linear_model import LinearRegression


class analyse:
    def __init__(self, dataPath, drive, mergedDataFile, modelDataFile, results_folder):
        # Docstring
        "Class container for analysing functions."
        self.modelDataFile = modelDataFile
        # Set paths
        self.dataPath = dataPath
        self.drive = drive
        self.results_folder = results_folder

        # Load data
        self.merge_data = pd.read_csv(mergedDataFile)
        print('Created analyse class')

    def get_responses(self):
        # Get response per frame
        indices = []
        all_responses = []
        for key in self.merge_data.keys():
            for stimulus_index in range(0, len(self.merge_data.keys())):
                response_column = 'Stimulus %s: response' % stimulus_index
                if response_column in key:
                    indices.append(int(stimulus_index))
                    all_responses.append(self.merge_data[response_column])

        # Create response DataFrame
        self.response_data = pd.DataFrame(all_responses, index=indices)
        self.response_data.sort_index(inplace=True)
        self.response_data.columns = self.merge_data['Meta:worker_code']
        self.response_data = self.response_data.transpose()

        # Get mean and std of responses
        self.response_mean = self.response_data.mean(skipna=True)
        self.response_std = self.response_data.std(skipna=True)

        # Get normalized response
        self.response_normal = (
            self.response_data-self.response_mean.mean())/self.response_std.mean()

        # Save responses
        response_data = pd.DataFrame(self.response_data)
        response_data.to_csv(self.results_folder +
                             'filtered_responses/' + 'response_data.csv')

        # Get anonymous data
        survey_data = pd.DataFrame(
            self.merge_data.loc[:, 'about_how_many_kilometers_miles_did_you_drive_in_the_last_12_months':'which_input_device_are_you_using_now'])
        survey_data.index = survey_data['Meta:worker_code']
        survey_data.pop('Meta:worker_code')
        self.survey_data = survey_data.join(response_data)

        self.survey_data.to_csv(self.results_folder +
                                'filtered_responses/' + 'survey_data.csv')
        # print(survey_data)
        print('Got responses')

    def find_outliers(self, thresh):
        # Find outliers
        self.bad_indices = []
        for index in self.response_data.index:
            if(self.response_data.loc[index].std(skipna=True) < thresh):
                self.bad_indices.append(index)
                self.response_data = self.response_data.drop(index)
        print('Found outliers')

    def info(self):
        # Get mean and std of responses
        self.response_mean = self.response_data.mean(skipna=True)
        self.response_std = self.response_data.std(skipna=True)

        # Get general info on data
        self.data_description = self.response_data.describe()
        self.data_description.to_csv(
            self.results_folder + 'filtered_responses/' + 'self.data_description.csv')
        # print(self.data_description)

        print('Got info')

    def split(self, useNorm=False):
        # Chose usage of normalized data
        if useNorm == True:
            data = self.response_normal
        else:
            data = self.response_data

        # Get mean and std of responses
        self.response_mean = data.mean(skipna=True)
        self.response_std = data.std(skipna=True)

        # Find middle
        middle = int(round(len(self.response_data)/2))

        # Get first half of data
        self.response_data_first = data[0:middle]
        self.response_mean_first = self.response_data_first.mean(skipna=True)
        self.response_std_first = self.response_data_first.std(skipna=True)

        # Get last half of data
        self.response_data_last = data[(middle+1):len(data)]
        self.response_mean_last = self.response_data_last.mean(skipna=True)
        self.response_std_last = self.response_data_last.std(skipna=True)

        # Get correlation of first and last half
        r_fl = self.response_mean_first.corr(self.response_mean_last)
        r2_fl = r_fl*r_fl
        # print('{:<25}'.format('autocorrelation') + ': R^2 =',f'{r2_fl:.5f}')
        # print('{:<25}'.format('autocorrelation') + ': R^2 =',f'{r2_fl:.5f}')

        # Plot correlation of first and last half
        self.plot_correlation(self.response_mean_first, self.response_mean_last,
                              self.response_std_last, self.response_std_first,
                              'first_half', 'last_half', 'general_autocorrelation', r=round(r_fl, 5))

        # self.training_data = self.response_data
        middle_frame = int(round(self.response_data.shape[1])/2)
        self.data_training = self.response_data.iloc[:, 0:middle_frame]
        self.data_testing = self.response_data.iloc[:,
                                                    middle_frame:self.response_data.shape[1]]

        print('Split data')

    def random(self, useNorm=False, seed=100):
        # Chose usage of normalized data
        if useNorm == True:
            data = self.response_normal
        else:
            data = self.response_data

           # Get mean and std of responses
        data_response_mean = data.mean(skipna=True)
        data_response_std = data.std(skipna=True)

        # Chose random person
        random.seed(seed)
        random_person = random.randrange(0, len(data), 1)
        self.single_response = data.iloc[random_person]

        # Get correlation of data and random person
        r_sm = self.single_response.corr(data_response_mean)
        r2_sm = r_sm*r_sm
        print('{:<30}'.format('Single random') + ': N' +
              '{:<4}'.format(str(random_person)) + " vs Human R^2 = " + str(r2_sm))

        # Plot correlation of data and random person
        self.plot_correlation(self.single_response, data_response_mean,
                              data_response_std, [],
                              ('Person n'+str(random_person)), 'Response mean', 'random', r2_sm)

        print('Got random correlation')

    def model(self, plotBool=True):
        self.model_data = pd.read_csv(
            self.results_folder + 'model_responses/' + self.modelDataFile)

        # Get keys
        self.parameter_keys = list(self.model_data)
        self.parameter_keys.remove('general_frame_number')
        self.model_data.pop('general_frame_number')

        for parameter in self.parameter_keys:
            # Get correlation
            r2 = self.model_data[parameter].corr(self.response_mean_last)**2

            # Print correlation
            # print('{:<25}'.format(parameter) + ': R^2 =',f'{r2:.5f}')

            # Save figure correlation
            if plotBool == True:
                self.plot_correlation(self.model_data[parameter], self.response_mean,
                                      None, self.response_std,
                                      str(parameter), 'response_mean', parameter, round(r2, 5))

        # Check model cronbach alpha
        # self.cronbach_alpha(self.model_data[['model_type',  'model_imminence',  'model_probability']])


        # Add mean response to correlation matrix
        self.model_data['response_mean_last'] = self.response_mean_last

        # Get correlation matrix
        corrMatrix = self.model_data.corr(method='pearson')
        # corrMatrix = corrMatrix.sort_values(by='response_mean')

        # Remove uppper triangle
        mask = np.zeros_like(corrMatrix)
        mask[np.triu_indices_from(mask, k=1)] = True

        # Get eigenvalues and vectors
        # Number of params
        n = len(self.parameter_keys)
        v = np.linalg.eig(corrMatrix.iloc[4:n, 4:n])

        v_sum = np.sum(v[0])
        v_csum = np.cumsum(v[0])
        v_ccurve = v_csum/v_sum
        v_cutoff = len(v_ccurve[(v_ccurve <= 0.8)])+1
        # print(v_cutoff)
        plt.clf()
        plt.plot(v[0])
        plt.plot(np.ones(len(v[0])))
        plt.title('Scree plot')
        plt.xlabel('Component')
        plt.ylabel('Eigenvalue')

        # Save figure
        plt.savefig(self.results_folder +
                    'regression/' + 'scree_plot' + '.png')
        
        plt.clf()
        plt.plot(v_ccurve)
        plt.title('Cumulative eigenvalue curve')
        plt.xlabel('Component')
        plt.ylabel('Cumulative eigenvalue')

        # Save figure
        plt.savefig(self.results_folder +
                    'regression/' + 'ccum_eigen_plot' + '.png')

        # Get significant params
        p_keys = self.model_data.keys()[4:n]
        # print(p_keys)

        significant_parameters = set([])
        # print(v_cutoff)
        
        for column in range(0, v_cutoff):
            for row in range(0, len(v[1])):
                if (abs(v[1][row, column]) >= 0.4):
                    # if (row <= 3):
                    #     pass
                    # else:
                    significant_parameters.add(p_keys[row])

        self.sig_params = list(significant_parameters)
        
        # Plot corr of sigs
        # plt.clf()
        # sn.heatmap(self.model_data[self.sig_params].corr(method='pearson'),vmax = 1,vmin = -1,cmap = 'RdBu_r', linewidths=.5, annot=True,yticklabels=self.sig_params)
        # plt.title('Correlations of significant parameters')
        # plt.show()

        # Get eigenvector heatmap
        # plt.clf()
        # sn.heatmap(v[1],vmax = 1,vmin = -1,cmap = 'RdBu_r', linewidths=.5, annot=True,yticklabels=p_keys)
        # plt.title('Eigenvectors')
        # plt.xlabel('Eigenvector index')
        # plt.ylabel('Parameter')
        # plt.show()

        # Save figure
        # plt.savefig(self.results_folder + 'regression/' + 'eigenvector_matrix' + '.png')

        # Plot correlation matrix
        if (plotBool == True):
            plt.clf()
            sn.heatmap(corrMatrix, vmax=1, vmin=-1,
                       cmap='RdBu_r', linewidths=.5, annot=True)
            plt.show()
            print('Got model data and correlations')
        else:
            pass

        r = self.model_data['model_combination'].corr(self.response_mean)

        return r**2

    def risky_images(self, model=False):
        # Get most risky and least risky images
        if (model == True):
            response_model_sorted = self.model_data['model_combination'].sort_values(
            )
            least_risky = response_model_sorted.index[0:5]
            most_risky = response_model_sorted.tail(5).index[::-1]
        else:
            response_model_sorted = self.response_mean.sort_values()
            least_risky = response_model_sorted.index[0:5]
            most_risky = response_model_sorted.tail(5).index[::-1]

        # Save most and least risky images
        i = 1
        for image in least_risky:
            # os.path.join(self.dataPath + self.drive + '/image_02/data')
            shutil.copyfile(self.dataPath + self.drive + '/image_02/data/' + str(image) + '.png',
                            self.results_folder + 'most_least_risky_images/' + 'least_risky_%s.png' % i)
            i += 1
        i = 1
        for image in most_risky:
            # os.path.join(self.dataPath + self.drive + '/image_02/data')
            shutil.copyfile(self.dataPath + self.drive + '/image_02/data/' + str(image) + '.png',
                            self.results_folder + 'most_least_risky_images/' + 'most_risky_%s.png' % i)
            i += 1

        print('Got risky images')

    def risk_ranking(self):
        # Sort list of mean response values
        response_mean_sorted = self.response_mean.sort_values()
        i = 0
        for image in response_mean_sorted.index:
            shutil.copyfile(self.dataPath + self.drive + '/image_02/data/' + str(
                image) + '.png', self.results_folder + 'risk_sorted_images/' + '%s.png' % i)
            i += 1

        # Sort list of model combination values
        response_model_sorted = pd.Series(
            self.model_data['model_combination']).sort_values()
        i = 0
        for image in response_model_sorted.index:
            shutil.copyfile(self.dataPath + self.drive + '/image_02/data/' + str(image) +
                            '.png', self.results_folder + 'risk_sorted_images/model' + '%s.png' % i)
            i += 1

        print('Ranked images on risk')

    def PCA(self):
        print("Starting PCA analysis")
        images = sorted(os.listdir(
            self.dataPath + self.drive + '/image_02/data/'))

        images_features_gray = []
        images_features_blue = []
        images_features_green = []
        images_features_red = []

        for image in images:
            image_features_gray = []
            image_features_blue = []
            image_features_green = []
            image_features_red = []

            full_path = self.dataPath + self.drive + '/image_02/data/' + image
            loaded_image = cv2.imread(full_path)

            gray = cv2.cvtColor(loaded_image, cv2.COLOR_BGR2GRAY)
            blue = loaded_image[:, :, 0]
            green = loaded_image[:, :, 1]
            red = loaded_image[:, :, 2]

            scaling = 1./2
            gray_scaled = cv2.resize(gray, (0, 0), fx=(scaling), fy=(scaling))
            blue_scaled = cv2.resize(blue, (0, 0), fx=(scaling), fy=(scaling))
            green_scaled = cv2.resize(
                green, (0, 0), fx=(scaling), fy=(scaling))
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
        print("Running decomposition")
        nc = 15  # number of model variables
        pca = decomposition.PCA(n_components=nc)

        std_gray = StandardScaler()
        gray_std = std_gray.fit_transform(images_features_gray)
        gray_pca = pca.fit_transform(gray_std)
        eigen_frames_gray = np.array(pca.components_.reshape(
            (nc, scaled_shape[0], scaled_shape[1])))

        std_blue = StandardScaler()
        blue_std = std_blue.fit_transform(images_features_blue)
        blue_pca = pca.fit_transform(blue_std)
        eigen_frames_blue = np.array(pca.components_.reshape(
            (nc, scaled_shape[0], scaled_shape[1])))

        std_green = StandardScaler()
        green_std = std_green.fit_transform(images_features_green)
        green_pca = pca.fit_transform(green_std)
        eigen_frames_green = np.array(pca.components_.reshape(
            (nc, scaled_shape[0], scaled_shape[1])))

        std_red = StandardScaler()
        red_std = std_red.fit_transform(images_features_red)
        red_pca = pca.fit_transform(red_std)
        eigen_frames_red = np.array(pca.components_.reshape(
            (nc, scaled_shape[0], scaled_shape[1])))

        # # Back tranform for check
        # back_transform = pca.inverse_transform(gray_pca)
        # back_transform_renormalize = std_gray.inverse_transform(back_transform)

        # # Show before and after
        # first_image = np.array(images_features[0]).reshape(scaled_shape)
        # cv2.imshow('Before PCA',first_image)
        # cv2.waitKey(0)

        # # second_image = np.array(back_transform_renormalize[0]).reshape(scaled_shape)
        # cv2.imshow('After PCA',second_image)
        # cv2.waitKey(0)

        gray_pca_df = pd.DataFrame(gray_pca)
        blue_pca_df = pd.DataFrame(blue_pca)
        green_pca_df = pd.DataFrame(green_pca)
        red_pca_df = pd.DataFrame(red_pca)

        self.pca = gray_pca_df
        
        r = round(gray_pca_df[2].corr(self.response_mean),5)
        self.plot_correlation(gray_pca_df,self.response_mean_last,name1='Gray pca component 2',name2='response_mean_last',r=r)

        print("Saving images")
        for i in range(0, nc):
            print('Feature: ', i)
            print('Gray correlation: ',
                  gray_pca_df[i].corr(self.response_mean))
            print('Blue correlation: ',
                  blue_pca_df[i].corr(self.response_mean))
            print('Green correlation: ',
                  green_pca_df[i].corr(self.response_mean))
            print('Red correlation: ', red_pca_df[i].corr(self.response_mean))
            max_pixel_gray = np.max(abs(eigen_frames_gray[i]))
            max_pixel_blue = np.max(abs(eigen_frames_blue[i]))
            max_pixel_green = np.max(abs(eigen_frames_green[i]))
            max_pixel_red = np.max(abs(eigen_frames_red[i]))

            gray_channel = eigen_frames_gray[i]*1/max_pixel_gray*255
            blue_channel = eigen_frames_blue[i]*1/max_pixel_blue*255
            green_channel = eigen_frames_green[i]*1/max_pixel_green*255
            red_channel = eigen_frames_red[i]*1/max_pixel_red*255

            bgr_image = np.zeros((scaled_shape[0], scaled_shape[1], 3))
            bgr_image[:, :, 0] = blue_channel
            bgr_image[:, :, 1] = green_channel
            bgr_image[:, :, 2] = red_channel

            cv2.imwrite(os.path.join(self.results_folder, 'pca',
                                     ('color ' + str(i)+'.png')), bgr_image)
            cv2.imwrite(os.path.join(self.results_folder, 'pca',
                                     ('gray ' + str(i)+'.png')), gray_channel)
            cv2.imwrite(os.path.join(self.results_folder, 'pca',
                                     ('blue ' + str(i)+'.png')), blue_channel)
            cv2.imwrite(os.path.join(self.results_folder, 'pca',
                                     ('green' + str(i)+'.png')), green_channel)
            cv2.imwrite(os.path.join(self.results_folder, 'pca',
                                     ('red ' + str(i)+'.png')), red_channel)

        print('Performed PCA')

    def multivariate_regression(self, pred='default'):
        # train = pd.DataFrame(self.pca, columns= ['0','1','2','3','4','5','6','7','8','9','10','11','12','13'])

        # train = self.pca.iloc[0:middle]
        # test = self.pca.iloc[middle:len(self.pca)]

        lr = LinearRegression(normalize=True)
        if (pred == 'default'):
            predictor_keys = ['general_velocity', 'general_distance_mean',
                              'general_number_bjects', 'manual_breaklight', 'occluded_mean']
        elif(pred == 'sig'):
            predictor_keys = self.sig_params
        elif(pred == 'all'):
            predictor_keys = self.model_data.keys()
        else:
            print('Wrong input, changing to default')
            predictor_keys = ['general_velocity', 'general_distance_mean',
                              'general_number_bjects', 'manual_breaklight', 'occluded_mean']
            

        predictors = self.model_data[predictor_keys]

        middle = int(round(predictors.shape[0]/2))

        # Fit regression
        print("Fitting regression model")
        print(self.sig_params)
        lr.fit(predictors[0:middle], self.response_mean[0:middle])
        predictions = lr.predict(predictors[middle:predictors.shape[0]])

        # Analyse result
        r = np.corrcoef(
            self.response_mean[middle:predictors.shape[0]], predictions)[0, 1]
        print('Correlation = {}'.format(r))
        self.plot_correlation(predictions, self.response_mean[middle:len(
            self.response_mean)], name1="Multivariate regression", name2="Response test", parameter="regression_multivariate", r=round(r, 5))
        print('Lr coef: {}'.format(lr.coef_))

        self.cronbach_alpha(self.model_data[predictor_keys])
        print('Performed multivariate regression')

    def risk_accidents(self, plotBool=False):
        # Get accident answers
        accident_occurence = self.merge_data['how_many_accidents_were_you_involved_in_when_driving_a_car_in_the_last_3_years_please_include_all_accidents_regardless_of_how_they_were_caused_how_slight_they_were_or_where_they_happened']

        # Filter no responses
        accident_occurence = [-1 if value ==
                              'i_prefer_not_to_respond' else value for value in accident_occurence]
        accident_occurence = [
            6 if value == 'more_than_5' else value for value in accident_occurence]
        accident_occurence = [value if value == np.nan else float(
            value) for value in accident_occurence]

        # Group by accidents
        n_bins = 20
        bins = np.linspace(0, 100, n_bins+1)
        binned = []

        for value in self.response_data.mean(axis=1):
            for b in bins:
                if (value <= b):
                    binned.append(b)
                    # print("Value:{} < bin:{}".format(value,b))
                    break

        # Get accident occurence
        average_score = list(self.response_data.mean(axis=1))
        risk_accidents = pd.DataFrame(
            {'Accidents': accident_occurence, 'Average_score': average_score})
        r = risk_accidents.corr().values[0, 1]
        self.plot_correlation(pd.Series(accident_occurence), pd.Series(
            average_score), name2='Accidents', name1='Average score', parameter='accident_score', r=round(r, 5))

        risk_accidents_grouped = []
        for i in range(8):
            risk_accidents_grouped.append([])
        for i in range(len(accident_occurence)):
            # print('i = {}, and value = {}'.format(i,close_occurence[i]))
            risk_accidents_grouped[int(accident_occurence[i])].append(
                average_score[i])

        # Risk close riding
        close_occurence = self.merge_data['how_often_do_you_do_the_following_driving_so_close_to_the_car_in_front_that_it_would_be_difficult_to_stop_in_an_emergency']

        # Filter no responses
        close_occurence = [
            0 if value == 'i_prefer_not_to_respond' else value for value in close_occurence]
        close_occurence = [
            1 if value == '0_times_per_month' else value for value in close_occurence]
        close_occurence = [
            2 if value == '1_to_3_times_per_month' else value for value in close_occurence]
        close_occurence = [
            3 if value == '4_to_6_times_per_month' else value for value in close_occurence]
        close_occurence = [
            4 if value == '7_to_9_times_per_month' else value for value in close_occurence]
        close_occurence = [
            5 if value == '10_or_more_times_per_month' else value for value in close_occurence]
        close_occurence = [value if value == np.nan else float(
            value) for value in close_occurence]

        close_occurence_grouped = []
        for i in range(6):
            close_occurence_grouped.append([])
        for i in range(len(close_occurence)):
            close_occurence_grouped[int(close_occurence[i])].append(
                average_score[i])

        r = np.corrcoef(close_occurence, average_score)[0, 1]
        self.plot_correlation(pd.Series(close_occurence), pd.Series(
            average_score), name2='Close driving', name1='Average score', parameter='close_score', r=round(r, 5))

        # Disregard speedlimit
        speed_occurence = self.merge_data['how_often_do_you_do_the_following_disregarding_the_speed_limit_on_a_residential_road']

        # Filter no response
        speed_occurence = [
            0 if value == 'i_prefer_not_to_respond' else value for value in speed_occurence]
        speed_occurence = [
            1 if value == '0_times_per_month' else value for value in speed_occurence]
        speed_occurence = [
            2 if value == '1_to_3_times_per_month' else value for value in speed_occurence]
        speed_occurence = [
            3 if value == '4_to_6_times_per_month' else value for value in speed_occurence]
        speed_occurence = [
            4 if value == '7_to_9_times_per_month' else value for value in speed_occurence]
        speed_occurence = [
            5 if value == '10_or_more_times_per_month' else value for value in speed_occurence]
        speed_occurence = [value if value == np.nan else float(
            value) for value in speed_occurence]

        speed_occurence_grouped = []
        for i in range(6):
            speed_occurence_grouped.append([])
        for i in range(len(speed_occurence)):
            speed_occurence_grouped[int(speed_occurence[i])].append(
                average_score[i])

        r = np.corrcoef(speed_occurence, average_score)[0, 1]
        self.plot_correlation(pd.Series(speed_occurence), pd.Series(
            average_score), name2='Speeding', name1='Average score', parameter='speed_score', r=round(r, 5))

        # Disregard phone
        phone_occurence = self.merge_data['how_often_do_you_do_the_following_using_a_mobile_phone_without_a_hands_free_kit']

        # Filter no response
        phone_occurence = [
            0 if value == 'i_prefer_not_to_respond' else value for value in phone_occurence]
        phone_occurence = [
            1 if value == '0_times_per_month' else value for value in phone_occurence]
        phone_occurence = [
            2 if value == '1_to_3_times_per_month' else value for value in phone_occurence]
        phone_occurence = [
            3 if value == '4_to_6_times_per_month' else value for value in phone_occurence]
        phone_occurence = [
            4 if value == '7_to_9_times_per_month' else value for value in phone_occurence]
        phone_occurence = [
            5 if value == '10_or_more_times_per_month' else value for value in phone_occurence]
        phone_occurence = [value if value == np.nan else float(
            value) for value in phone_occurence]

        phone_occurence_grouped = []
        for i in range(6):
            phone_occurence_grouped.append([])
        for i in range(len(phone_occurence)):
            phone_occurence_grouped[int(phone_occurence[i])].append(
                average_score[i])

        r = np.corrcoef(phone_occurence, average_score)[0, 1]
        self.plot_correlation(pd.Series(phone_occurence), pd.Series(
            average_score), name2='Phone driving', name1='Average score', parameter='phone_score', r=round(r, 5))
        
        
        # Result correlation matrix
        inter_group_df = pd.DataFrame([speed_occurence, phone_occurence, accident_occurence, close_occurence])
        inter_group_df = inter_group_df.transpose()
        inter_group_df.columns = ['speed_occurence', 'phone_occurence', 'accident_occurence', 'close_occurence']
        inter_group_corr = inter_group_df.corr()
        
        plt.clf()
        sn.heatmap(inter_group_corr, vmax=1, vmin=-1,
                    cmap='RdBu_r', linewidths=.5, annot=True)
        plt.show()
            
        
        
        r = np.corrcoef(speed_occurence, accident_occurence)[0, 1]
        self.plot_correlation(pd.Series(speed_occurence), pd.Series(
            accident_occurence), name2='Speed driving', name1='Accidents', parameter='speed_accident', r=round(r, 5))
        
        r = np.corrcoef(phone_occurence, accident_occurence)[0, 1]
        self.plot_correlation(pd.Series(phone_occurence), pd.Series(
            average_score), name2='Phone driving', name1='Accidents', parameter='phone_accident', r=round(r, 5))
        
        r = np.corrcoef(close_occurence, accident_occurence)[0, 1]
        self.plot_correlation(pd.Series(close_occurence), pd.Series(
            accident_occurence), name2='close driving', name1='Accidents', parameter='close_accident', r=round(r, 5))

        # print(survey_results)


        if plotBool:
            plt.figure()
            plt.boxplot(risk_accidents_grouped, labels=[
                        'no reply', '0', '1', '2', '3', '4', '5', 'more than 5'])
            plt.title("Accident risk")
            plt.xlabel("Accident occurence")
            plt.ylabel("Risk score")
            plt.savefig(self.results_folder + 'survey_images/' +
                        'risk_accidents_box' + '.png')

            plt.figure()
            plt.boxplot(close_occurence_grouped, labels=[
                        'no reply', '0', '1-3', '4-6', '7-9', '10 or more'])
            plt.title("Keeping distance driving risk")
            plt.xlabel("Disregard occurence")
            plt.ylabel("Risk score")
            plt.savefig(self.results_folder + 'survey_images/' +
                        'close_occurence_box' + '.png')

            plt.figure()
            plt.boxplot(speed_occurence_grouped, labels=[
                        'no reply', '0', '1-3', '4-6', '7-9', '10 or more'])
            plt.title("Speed disregard driving risk")
            plt.xlabel("Disregard occurence")
            plt.ylabel("Risk score")
            plt.savefig(self.results_folder + 'survey_images/' +
                        'speed_occurence_box' + '.png')

            plt.figure()
            plt.boxplot(phone_occurence_grouped, labels=[
                        'no reply', '0', '1-3', '4-6', '7-9', '10 or more'])
            plt.title("Handsfree disregard driving risk")
            plt.xlabel("Disregard occurence")
            plt.ylabel("Risk score")
            plt.savefig(self.results_folder + 'survey_images/' +
                        'phone_occurence_box' + '.png')

            plt.figure()
            plt.hist(average_score, bins=n_bins, rwidth=0.9)
            plt.title("Average score responses")
            plt.xlabel("Average risk score")
            plt.ylabel("Occurences")
            plt.savefig(self.results_folder + 'survey_images/' +
                        'avg_response' + '.png')

            # plt.show()

            # plt.figure()
            # plt.hist2d(accident_occurence,average_score,alpha=0.9)
            # plt.title("Accident risk")
            # plt.xlabel("Accident occurence")
            # plt.ylabel("Risk score")
        print('Saved survey images')

    def cronbach_alpha(self, df):
        # 1. Transform the df into a correlation matrix
        df_corr = df.corr()

        # 2.1 Calculate N
        # The number of variables equals the number of columns in the df
        N = df.shape[1]

        # 2.2 Calculate R
        rs = np.array([])
        for i, col in enumerate(df_corr.columns):
            sum_ = df_corr[col][i+1:].values
            rs = np.append(sum_, rs)
        mean_r = np.mean(rs)

        # 3. Use the formula to calculate Cronbach's Alpha
        cronbach_alpha = (N * mean_r) / (1 + (N - 1) * mean_r)
        # print(rs)
        print('Cronbach alpha: {}'.format(cronbach_alpha))
        print('Calculated cronbach alpha')

    def plot_correlation(self, series1, series2,
                         std1=None,
                         std2=None,
                         name1='Series 1', name2='Series 2',
                         parameter='Parameter', r2=np.nan, r = np.nan):

        # Plot errobar figure
        plt.clf()
        plt.errorbar(series1, series2, std2, std1, linestyle='None',
                     marker='.', markeredgecolor='green')
        if(not isnan(r2)):
            plt.title(name1 + ' vs. ' + name2 + " | R^2 = %s" % r2)
        elif(not isnan(r)):
            plt.title(name1 + ' vs. ' + name2 + " | r = %s" % r)
        else:
            plt.title(name1 + ' vs. ' + name2)
        plt.xlabel(name1)
        plt.ylabel(name2)

        # Create linear fit of model and responses
        linear_model = np.polyfit(series1, series2, 1)
        linear_model_fn = np.poly1d(linear_model)
        x_s = np.arange(series1.min(), series1.max(),
                        ((series1.max()-series1.min())/1000))

        # Plot linear fit
        plt.plot(x_s, linear_model_fn(x_s), color="red")

        # Save figure
        plt.savefig(self.results_folder +
                    'correlation_images/' + parameter + '.png')


if __name__ == "__main__":
    dataPath = '/home/jim/HDDocuments/university/master/thesis/ROS/data/2011_09_26'
    drive = '/test_images'
    resultsFolder = '/home/jim/HDDocuments/university/master/thesis/results/'
    mergedDataFile = '/home/jim/HDDocuments/university/master/thesis/results/filtered_responses/merged_data.csv'
    modelDataFile = 'model_results.csv'

    analyse = analyse(dataPath, drive, mergedDataFile,
                      modelDataFile, resultsFolder)
    analyse.get_responses()

    analyse.info()
    # analyse.find_outliers(10)
    analyse.split()
    # analyse.risk_accidents()
    analyse.model(plotBool=False)
    # analyse.risky_images(model=False)
    # analyse.risk_accidents(plotBool=False)
    # analyse.risk_ranking()
    analyse.PCA()
    # analyse.multivariate_regression(pred='sig')
    # analyse.plot_correlation(analyse.model_data['road_road'],analyse.model_data['general_velocity'])

    # analyse.cronbach_alpha(analyse.response_data)
