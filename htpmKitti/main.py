#!/usr/bin/env python3

# Import own modules
import kitti
import model.services as model
import online.dataCombiner as online
from online.herokuFunctions import heroku
from online.appenFunctions import appen

# Set name of heroku and appen file
herokuFile  = 'entries_1.json'
appenFile   = 'f1669822.csv'
resultsFolder   ='/home/jim/HDDocuments/university/master/thesis/results/'
        
# Prepare data from online survey
# oc = online.combiner(herokuFile,appenFile,resultsFolder)
# oc.combine()



kp = model.kitti_parser()
x = [0., 1.458974, 2.63547244, 0.96564807, 2.21222542, 1.65225034, 0., 0., 1.,
     2.20176468, 2.40070779, 0.1750559,
     0.20347586, 6.54656438]
kp.save_model(x)
