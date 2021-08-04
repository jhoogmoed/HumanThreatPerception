#!/usr/bin/env python

# Import own modules
import model.services as services
import model.optimisation as optimisation
import model.compare as compare
import online.dataCombiner as online
from online.herokuFunctions import heroku
from online.appenFunctions import appen

# Set name of heroku and appen file
herokuFile     = 'entries_1.json'
appenFile      = 'f1669822.csv'
resultsFolder  ='/media/jim/HDD/university/master/thesis/results/'

# Set kitti data folder
dataPath       = '/media/jim/HDD/university/master/thesis/ROS/data/2011_09_26'
drive          = '/test_images'

# Set name of model results file
resultsFile = 'model_results.csv'

# Prepare data from online survey
oc = online.combiner(herokuFile,appenFile,resultsFolder)
appenUniquePath, herokuResponsePath,mergedDataFile = oc.combine()

# Construct kitti parser
kp = services.kitti_parser(dataPath,drive,resultsFolder)

# Start parameter weights
x = [0.2, 0.4, 0.6, 1., 0.2, 1., 0.6, 0.2, 0., 
              0.32, 0.32, 1.43, 
              1., 0.5]

optimisation()

# Example parameter weights
# x = [0., 1.458974, 2.63547244, 0.96564807, 2.21222542, 1.65225034, 0., 0., 1.,
#      2.20176468, 2.40070779, 0.1750559,
#      0.20347586, 6.54656438]

# Save model results
kp.save_model(x,resultsFile)


analyse = compare.analyse(dataPath,drive,mergedDataFile,resultsFile,resultsFolder)
analyse.get_responses()
analyse.info()
analyse.split()
analyse.model()
# analyse.risky_images()
analyse.risk_ranking()
