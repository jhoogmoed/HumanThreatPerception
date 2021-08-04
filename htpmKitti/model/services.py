#!/usr/bin/env python3
import os
import numpy as np
# import kitti.tracklet_parser as tracklet_parser
# import kitti.tracklet_parser as tracklet_parser
import collections
import pandas as pd

KittiObject = collections.namedtuple('KittiObject', ['type',
                                                     'truncated',
                                                     'occluded',
                                                     'alpha',
                                                     'bbox',
                                                     'dimensions',
                                                     'location',
                                                     'location_y'])
KittiImu = collections.namedtuple(
    'KittiImu', ['location', 'linear_velocity', 'linear_acceleration'])


class kitti_parser:
    def __init__(self,dataPath,drive,resultsFolder):
        # Set base paths
        self.dataPath = dataPath 
        self.drive = drive
        self.resultsFolder = resultsFolder

        # Check if exists
        if(not os.path.exists(self.dataPath+self.drive)):
            print("Drive does not exist")
            raise SystemExit

        # Image pathsdataPath,drive,resultsFolder
        try:
            self.left_color_image_list = sorted(os.listdir(
                self.dataPath + self.drive + '/image_02/data'), key=self.sorter)
        except:
            print("No image data")
            raise SystemExit

        # Imu paths
        try:
            self.imuFileList = sorted(os.listdir(
                self.dataPath + self.drive + '/oxts/data/'), key=self.sorter)
        except:
            print("No oxts data")
            raise SystemExit

        # Object paths
        try:
            self.objectFileList = sorted(os.listdir(
                self.dataPath + self.drive + '/label_2'), key=self.sorter)
        except:
            print("No object data, create from xml...")
            try:
                tracklet_parser.main(self.dataPath, self.drive)
                self.objects_list = sorted(os.listdir(
                    self.dataPath + self.drive + '/label_2'), key=self.sorter)
            except:
                print("No object xml")
                raise SystemExit

        # Check variables
        self.frame = 0
        self.done = 0

        # Setup data acquisition
        try:
            os.remove(os.path.join(self.resultsFolder,
                                   'model_responses/model_results.csv'))
        except:
            pass

        # Get information
        self.get_road()
        self.get_objects()
        self.get_imu()
        self.get_manual()

    def get_road(self):
        self.par_city = []
        self.par_residential = []
        self.par_road = []
        
        road_file = open(self.dataPath + self.drive +
                         '/uniform_image_list.txt', "r")
        lines = road_file.readlines()
        self.road_types = []
        for i in range(len(lines)):
            road = lines[i].split('/')[0]
            self.road_types.append(road)
            self.par_city.append((road == 'city')*1)
            self.par_residential.append((road == 'residential')*1)
            self.par_road.append((road == 'road')*1)          

    def get_objects(self):
        self.objectsList = []
        for i in range(len(self.objectFileList)):
            # Open file
            self.object_file = open(
                self.dataPath + self.drive + '/label_2/' + self.objectFileList[i], "r")

            # Setup object per frame
            objects = []
            # Read next line
            lines = self.object_file.readlines()
            for object in lines:
                oArgs = object.split(' ')
                type = oArgs[0]
                truncated = float(oArgs[1])
                occluded = int(oArgs[2])
                alpha = float(oArgs[3])
                bbox = [float(oArgs[4]),
                        float(oArgs[5]),
                        float(oArgs[6]),
                        float(oArgs[7])]
                dimensions = [float(oArgs[8]),
                              float(oArgs[9]),
                              float(oArgs[10])]
                location = [float(oArgs[11]),
                            float(oArgs[12]),
                            float(oArgs[13])]
                location_y = float(oArgs[14])

                # Append object list of frame
                objects.append(KittiObject(type,
                                           truncated,
                                           occluded,
                                           alpha,
                                           bbox,
                                           dimensions,
                                           location,
                                           location_y))

            # Close file
            self.object_file.close
            self.objectsList.append(objects)

    def get_imu(self):
        self.imuList = []
        for file in self.imuFileList:
            # Open file
            imu_file = open(
                self.dataPath + self.drive + '/oxts/data/' + file, "r")

            # Create new imu msg
            # imuObject = KittiImu

            # Get imu data from file
            line = imu_file.readline()
            imuArgs = line.split(' ')

            # Fill new object
            location = [
                float(imuArgs[0]),
                float(imuArgs[1]),
                float(imuArgs[2]),
                float(imuArgs[5])]
            linear_velocity = [
                float(imuArgs[8]),
                float(imuArgs[9]),
                float(imuArgs[10])]
            linear_acceleration = [
                float(imuArgs[11]),
                float(imuArgs[12]),
                float(imuArgs[13])]
            self.imuList.append(
                KittiImu(location, linear_velocity, linear_acceleration))
            # Close file
            imu_file.close

    def get_manual(self):
        self.manual_data = pd.read_csv(
            self.dataPath + self.drive + '/manual_data.csv')

    def sorter(self, name):
        frame = int(name.split('.')[0])
        return frame

    def typeSwitch(self, objType, parameters):
        # Switch to type to assign weight based on...
        typeSwitch = {
            'Car': parameters[0],
            'Van': parameters[1],
            'Truck': parameters[2],
            'Pedestrian': parameters[3],
            'Person_sitting': parameters[4],
            'Cyclist': parameters[5],
            'Tram': parameters[6],
            'Misc': parameters[7],
            'DontCare': parameters[8],
        }
        return typeSwitch.get(objType, "Invalid object type")

    def roadSwitch(self, roadType, parameters):
        # Switch to type to assign weight based on...
        roadSwitch = {
            'city': parameters[9],
            'residential': parameters[10],
            'road': parameters[11],
        }
        return roadSwitch.get(roadType, "Invalid object type")

    def fast_type(self, x):
        par_type = []
        par_alpha = []
        par_occluded = []
        par_truncated = []
        par_size = []
        for frame_objects in self.objectsList:
            types = []
            alpha = []
            occluded = []
            truncated = []
            size = []
            for object in frame_objects:
                types.append(self.typeSwitch(object.type, x))
                alpha.append(abs(object.alpha))
                occluded.append(object.occluded)
                truncated.append(object.truncated)
                size.append(np.prod(object.dimensions))
            par_alpha.append(alpha)
            par_type.append(sum(types))
            par_occluded.append(occluded)
            par_truncated.append(truncated)
            par_size.append(size)
        return par_type, par_alpha, par_occluded, par_truncated,par_size

    def fast_imm(self, x):
        # Get variables from arguments
        a = x[12]
        b = x[13]

        # Create empty return lists
        par_total_distance = []
        par_velocity = []
        par_imm = []

        # Get object and ego vehicle data per frame
        for frame in range(len(self.imuFileList)):
            # Get ego velocity
            velocity = np.linalg.norm(self.imuList[frame].linear_velocity, 2)

            # Construct save variables
            all_imminence = []
            all_distance = []

            # Get object data per object in frame
            for object in self.objectsList[frame]:
                distance = np.linalg.norm(object.location, 2)

                # Linear imminence parameter
                # imm =  a * distance/velocity + b

                # Quadratic imminence parameter
                if b == 0:
                    imm = np.nan
                else:
                    imm = a*(distance/velocity)**(1/b)
                    if imm>50:
                        imm = 50

                # Save paremeter per object
                all_imminence.append(imm)
                all_distance.append(distance)

            # Save parameter values per frame
            par_imm.append(sum(all_imminence))
            par_velocity.append(velocity)
            par_total_distance.append(all_distance)
            frame += 1
        return par_imm, par_velocity, par_total_distance

    def fast_prob(self, x):
        probability_par = []
        for road in self.road_types:
            probability_par.append(self.roadSwitch(road, x))
        
        return probability_par

    def get_model(self, x):
        # Get individual model results
        par_all_imminence, par_velocity, par_all_distance = self.fast_imm(x)
        par_type, par_alpha, par_occluded, par_truncated,par_size = self.fast_type(x)
        par_probability = self.fast_prob(x)

        # Construct empty lists for itereation
        par_combi = []
        
        number_objects = []
        
        sum_distance = []
        min_distance = []
        mean_distance = []
        
        min_alpha = []
        mean_alpha = []
        max_alpha = []
        
        mean_par_occluded = []
        sum_par_occluded = []
        
        mean_par_truncated = []
        sum_par_truncated = []
        
        mean_par_size = []
        max_par_size = []
        min_par_size = []
        sum_par_size = []
        

        # Get combined model results
        for frame in range(len(par_all_imminence)):
            sum_distance.append(sum(par_all_distance[frame]))
            min_distance.append(min(par_all_distance[frame], default=0))
            
            # Check for objects present
            if len(par_all_distance[frame]) != 0:
                number_objects.append(len(par_all_distance[frame]))
                
                mean_distance.append(
                    sum(par_all_distance[frame])/len(par_all_distance[frame]))
                
                min_alpha.append(min(par_alpha[frame]))
                mean_alpha.append(
                    sum(par_alpha[frame])/len(par_alpha[frame]))
                max_alpha.append(max(par_alpha[frame]))
                
                mean_par_occluded.append(sum(par_occluded[frame])/len(par_occluded[frame]))
                sum_par_occluded.append(sum(par_occluded[frame]))
                
                mean_par_truncated.append(sum(par_truncated[frame])/len(par_truncated[frame]))
                sum_par_truncated.append(sum(par_truncated[frame]))
                
                mean_par_size.append(sum(par_size[frame])/len(par_size[frame]))
                max_par_size.append(max(par_size[frame]))
                min_par_size.append(min(par_size[frame]))
                sum_par_size.append(sum(par_size[frame]))
                
            else:
                number_objects.append(0.0)
                mean_distance.append(0.0)
                
                min_alpha.append(0.0)
                mean_alpha.append(0.0)
                max_alpha.append(0.0)
                
                mean_par_occluded.append(0.0)
                sum_par_occluded.append(0.0)
                
                mean_par_truncated.append(0.0)
                sum_par_truncated.append(0.0)
                
                mean_par_size.append(0.0)
                max_par_size.append(0.0)
                min_par_size.append(0.0)
                sum_par_size.append(0.0)
            par_combi.append(par_all_imminence[frame] +
                            par_type[frame] + par_probability[frame])
                
        # Create empty dict
        results = {}

        # Add items to dict
        results['general_frame_number'] = range(
            len(self.left_color_image_list))
        results['model_combination'] = par_combi
        results['model_type'] = par_type
        results['model_imminence'] = par_all_imminence
        results['model_probability'] = par_probability
        results['general_velocity'] = par_velocity
        results['general_distance_sum'] = sum_distance
        results['general_distance_min'] = min_distance
        results['general_distance_mean'] = mean_distance
        results['general_number_bjects'] = number_objects
        results['manual_car_toward'] = self.manual_data.CarToward
        results['manual_car_away'] = self.manual_data.CarAway
        results['manual_breaklight'] = self.manual_data.Breaklight
        results['alpha_min'] = min_alpha
        results['alpha_mean'] = mean_alpha
        results['alpha_max'] = max_alpha
        results['occluded_mean'] = mean_par_occluded
        results['occluded_sum'] = sum_par_occluded
        results['truncated_mean'] = mean_par_truncated
        results['truncated_sum'] = sum_par_truncated
        results['size_mean'] = mean_par_size
        results['size_max'] = max_par_size
        results['size_min'] = min_par_size
        results['size_sum'] = sum_par_size
        results['road_road']= self.par_road
        results['road_residential'] = self.par_residential
        results['road_city'] = self.par_city

        return results

    def save_model(self, x,modelFile = 'model_results.csv'):
        # Get model response
        results = self.get_model(x)

        # Create dataframe from dict
        resultsDF = pd.DataFrame.from_dict(results)

        # save dataframe as csv file
        resultsDF.to_csv(os.path.join(self.resultsFolder,
                                      'model_responses',modelFile), index=False)


if __name__ == "__main__":
    # Example input
    dataPath       = '/home/jim/HDDocuments/university/master/thesis/ROS/data/2011_09_26'
    drive          = '/test_images'
    resultsFolder  = '/home/jim/HDDocuments/university/master/thesis/results'
    
    # Construct parser class
    kp = kitti_parser(dataPath,drive,resultsFolder)
    
    # Example parameters
    # x = [0., 1.458974, 2.63547244, 0.96564807, 2.21222542, 1.65225034, 0., 0., 1.,
    #      2.20176468, 2.40070779, 0.1750559,
    #      0.20347586, 6.54656438]
    
    x = [0.2, 0.4, 0.6, 1., 0.2, 1., 0.6, 0.2, 0., 
              3., 1.5, 0., 
              1., 0.1]
    
    # Get model results
    results = kp.get_model(x)
    
    # Save model results
    kp.save_model(x)
