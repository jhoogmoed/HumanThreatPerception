#!/usr/bin/env python
from scipy import optimize
import rospy
import numpy as np
import pandas as pd


from compare_res_vs_model import analyse 
from parameter_services import kitti_parser

from scipy.optimize import minimize

# Set data folder paths
dataPath        = '/home/jim/HDDocuments/university/master/thesis/ROS/data/2011_09_26'
drive           = '/test_images'
results_folder  = '/home/jim/HDDocuments/university/master/thesis/results/'
mergedDataPath  = 'merged_data.csv'
# firstDataPath   = 
# lastDataPath    =
modelDataPath   = 'model_results.csv'

# Get responses from participants
data = analyse(dataPath,drive,mergedDataPath,modelDataPath,results_folder)
data.get_responses()
data.split()


# Start values
type_params = ['type/car',
               'type/van',
               'type/truck',
               'type/ped',
               'type/pedsit',
               'type/cycl',
               'type/tram',
               'type/misc',
               'type/dc']
type_params_start = [0.2, 0.4, 0.6, 1, 0.2, 1, 0.4, 0, 0]

prob_params = ['prob/road',
               'prob/city',
               'prob/residential']
prob_params_start = [0.0,2.0,2.0]

imm_params = ['imm/gain', 
              'imm/bias',
              'imm/max']
imm_params_start = [0.5,0]

param_names = type_params + prob_params +imm_params
i = 0
kp = kitti_parser()

def get_correlation(x):
    global i 
    results = pd.DataFrame(kp.parameter_server(x))
    
    print("Iteration %s" %i)
    rc =  results['Combination parameter'].corr(data.response_mean_first)
    rt =  results['Type parameter'].corr(data.response_mean_first)
    rp =  results['Probability parameter'].corr(data.response_mean_first)
    ri =  results['Imminence parameter'].corr(data.response_mean_first)
    
    
    print("Combination corr : %s" %(rc*rc))
    print("Type corr        : %s" %(rt*rt))
    print("Probability corr : %s" %(rp*rp))
    print("Imminence corr   : %s" %(ri*ri))
    
    i+=1
    return 1-(rc * rc)

# Random first guess
# x0 = [1,1,1,1,1,1,1,1,1,1,1,1,1,1]
# x0 = type_params_start +prob_params_start +imm_params_start

# First optimisation results starting from all values 1
# x0 = [ 0.78603735,  0.89275405,  0.97970909,  0.68928311,  0.64975938,
#         0.83689267,  0.62785914,  0.74671845,  1.        ,  1.06649842,
#         1.03987374,  0.89361434,  0.3830448 , -0.79077126]

# Individual optimisation results
x0 = [0.24441526,  1.50570518,  2.52707192,  1.17821516,  1.83390363,  1.84680202,
      -1.2130232,   0.01693858,  1.,
      0.11087554,  0.09018137, -0.00105709,
      0.15939724, 0.01049802]


#Broyden-Fletcher-Goldfarb-Shanno method
# res = minimize(get_correlation, x0, method='BFGS',options={'disp': True,'xtol': 0.00000001})

# Basinhopper method
minimizer_kwargs = {"method": "BFGS"}
res = optimize.basinhopping(get_correlation, x0,minimizer_kwargs=minimizer_kwargs)

# Nelde Mead method
# res = minimize(get_correlation,x0 ,method='Nelder-Mead',options={'disp': True})

print("Parameters after optimisation")
print(res.x)
final_results = pd.DataFrame(kp.parameter_server(res.x))
rc =  final_results['Combination parameter'].corr(data.response_mean_first)
rt =  final_results['Type parameter'].corr(data.response_mean_first)
rp =  final_results['Probability parameter'].corr(data.response_mean_first)
ri =  final_results['Imminence parameter'].corr(data.response_mean_first)

print("Correlation after optimisation")
print("Combination corr : %s" %(rc*rc))
print("Type corr        : %s" %(rt*rt))
print("Probability corr : %s" %(rp*rp))
print("Imminence corr   : %s" %(ri*ri))

second_results = pd.DataFrame(kp.parameter_server(res.x))
rc =  second_results['Combination parameter'].corr(data.response_mean_last)
rt =  second_results['Type parameter'].corr(data.response_mean_last)
rp =  second_results['Probability parameter'].corr(data.response_mean_last)
ri =  second_results['Imminence parameter'].corr(data.response_mean_last)

print("Correlation with second half")
print("Combination corr : %s" %(rc*rc))
print("Type corr        : %s" %(rt*rt))
print("Probability corr : %s" %(rp*rp))
print("Imminence corr   : %s" %(ri*ri))
