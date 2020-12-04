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
modelDataPath   = 'model_results.csv'

# Get responses from participants
data = analyse(dataPath,drive,mergedDataPath,modelDataPath,results_folder)
data.get_responses()

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
# type_params_start = [0.2, 0.4, 0.6, 1, 0.2, 1, 0.4, 0, 0]
type_params_start = [0.15685902,
                     1.2939467,
                     1.82245876,
                     0.99936893,
                     1.6379139,
                     1.0,
                     0.0,
                     1.0,
                     5.17585792]

prob_params = ['prob/road',
               'prob/city',
               'prob/residential']
prob_params_start = [0.0,2.0,2.0]

imm_params = ['imm/gain', 
              'imm/bias',
              'imm/max']
imm_params_start = [1,0,1]

param_names = type_params + prob_params +imm_params
i = 0
kp = kitti_parser()

def get_correlation(x):
    global i 
    results = pd.DataFrame(kp.parameter_server(x))
    
    print("Iteration %s" %i)
    rc =  results['Combination parameter'].corr(data.response_mean)
    rt =  results['Type parameter'].corr(data.response_mean)
    rp =  results['Probability parameter'].corr(data.response_mean)
    ri =  results['Imminence parameter'].corr(data.response_mean)
    
    
    print("Combination corr : %s" %(rc*rc))
    print("Type corr        : %s" %(rt*rt))
    print("Probability corr : %s" %(rp*rp))
    print("Imminence corr   : %s" %(ri*ri))
    
    i+=1
    return 1-(rc * rc)


x0 = type_params_start +prob_params_start +imm_params_start
x0 = np.array(x0)
# x0 = [ 0.44280227,  3.34705824,  5.76650168,  1.71872993, -1.05546119,
#         4.27986996, -3.84714484, -0.67921923, 18.11550273,  4.97899823,
#         4.1454334 ,  0.58620334, 19.83541916, 12.93964481, 13.93964481]
    
res = minimize(get_correlation, x0, method='BFGS',options={'disp': True,'xtol': 0.000001})
# res = optimize.basinhopping(get_correlation, x0)
# res = minimize(get_correlation,x0 ,method='Nelder-Mead',options={'disp': True})

print(res)


    

