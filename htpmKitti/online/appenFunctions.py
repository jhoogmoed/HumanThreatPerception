#!/usr/bin/env python3

import pandas as pd


class appen:
    def __init__(self, csvFile, results_folder):
        self.csvFile = csvFile
        self.results_folder = results_folder
        self.appen_data = pd.read_csv(
            self.results_folder + 'online_data/' + csvFile)

    def find_cheaters(self, column_type='standard'):
        code_name = 'type_the_code_that_you_received_at_the_end_of_the_experiment'
        self.unique_appen_data = self.appen_data.drop_duplicates(subset=[
                                                                 code_name])
        if column_type == 'standard':
            self.cheater_appen_data = self.appen_data.drop(
                self.unique_appen_data.index)

        elif column_type == 'daniel':
            self.cheater_appen_data = self.appen_data.drop(
                self.unique_appen_data.index)
            self.cheater_appen_data = self.cheater_appen_data[['_id', '_worker_id', code_name, '_ip', '_started_at', '_created_at', '_country', 'what_is_your_gender', 'what_is_your_age', 'have_you_read_and_understood_the_above_instructions', 'at_which_age_did_you_obtain_your_first_license_for_driving_a_car_or_motorcycle', 'what_is_your_primary_mode_of_transportation',
                                                               'on_average_how_often_did_you_drive_a_vehicle_in_the_last_12_months', 'about_how_many_kilometers_miles_did_you_drive_in_the_last_12_months', 'how_many_accidents_were_you_involved_in_when_driving_a_car_in_the_last_3_years_please_include_all_accidents_regardless_of_how_they_were_caused_how_slight_they_were_or_where_they_happened']]
            daniels_headers = ['ID', 'worker_id', 'worker_code', 'ip_address', 'start_time', 'end_time', 'country', 'gender',
                               'age', 'read_instructions', 'license_age', 'primary_transport', 'avg_vehicle_time', 'avg_mileage', 'accidents']
            self.cheater_appen_data.columns = daniels_headers

        else:
            pass
        print('There are %s cheaters detected, giving %s unreliable results.' % (len(
            self.cheater_appen_data['worker_code'].unique()), len(self.cheater_appen_data['worker_code'])))

    def makeCSV(self):
        appen_name = self.csvFile.split('.')[0]
        self.unique_appen_data.to_csv(self.results_folder +
                                      'filtered_responses/'+
                                      appen_name+
                                      '_unique.csv')
        self.cheater_appen_data.to_csv(
            self.results_folder + 'filtered_responses/' + appen_name + '_cheaters.csv')
        return self.results_folder + 'filtered_responses/' + appen_name + '_unique.csv', appen_name + '_cheaters.csv'


if __name__ == "__main__":
    results_folder = '/home/jim/HDDocuments/university/master/thesis/results/'
    appenFile = 'f1669822.csv'
    a = appen(appenFile, results_folder)
    a.find_cheaters('daniel')
    a.makeCSV()
