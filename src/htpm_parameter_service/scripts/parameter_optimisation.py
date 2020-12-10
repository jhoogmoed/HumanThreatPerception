#!/usr/bin/env python
from scipy import optimize
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
data.split()

# Start values
params = {'type/car': 0.2,
          'type/van': 0.4,
          'type/truck' : 0.6,
          'type/ped' : 1,
          'type/pedsit': 0.2,
          'type/cycl' : 1,
          'type/tram': 0.4,
          'type/misc' : 0,
          'type/dc' : 0,
          'prob/road' : 0,
          'prob/city' : 2,
          'prob/residential' : 2,
          'imm/gain' : -0.5, 
          'imm/bias' : 0}

i = 0
kp = kitti_parser()

def get_correlation(x):
    global i 
    results = pd.DataFrame(kp.get_model(x))
    
    # print("Iteration %s" %i)
    r = get_corr(results,'first',False)
    
    i+=1
    return 1-(r['c']**2)

def get_corr(results,data_range,print_bool):
    ranges = {'first': data.response_mean_first,
              'last': data.response_mean_last}
    r = {}
    r['c'] =  results['Combination parameter'].corr(ranges[data_range])
    r['t'] =  results['Type parameter'].corr(ranges[data_range])
    r['p'] =  results['Probability parameter'].corr(ranges[data_range])
    r['i'] =  results['Imminence parameter'].corr(ranges[data_range])
    
    if print_bool == True:
        print("Combination corr : %s" %(r['c']**2))
        print("Type corr        : %s" %(r['t']**2))
        print("Probability corr : %s" %(r['p']**2))
        print("Imminence corr   : %s" %(r['i']**2))
    return r 
    
# Random first guess
x0 = [1,1,1,1,1,1,1,1,1,1,1,1,1,1]
# x0 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]

bnds  = ((0, 10),(0, 10),(0,10),(0, 10),(0, 10),
           (0, 10),(0, 10),(0, 10),(0, 10),
           (0, 10),(0, 10),(0, 10),
           (0, 10),(None, None))

# bnds  = ((0, 1),(0, 1),(0,1),(0, 1),(0, 1),
#            (0, 1),(0, 1),(0, 1),(0, 1),
#            (0, 1),(0, 1),(0, 1),
#            (0, 1),(-1, 1))


#Broyden-Fletcher-Goldfarb-Shanno method
res = minimize(get_correlation, params.values(), method='TNC',bounds=bnds,options={'disp': True, 'maxfun': 1000}) # 

# Basinhopper method
# class MyBounds(object):

#     def __init__(self, xmax=[10,10,10,10,10,10,10,10,10,10,10,10,10,10],
#                  xmin=[0,0,0,0,0,0,0,0,0,0,0,0,-10,-10]):
#         self.xmax = np.array(xmax)
#         self.xmin = np.array(xmin)

#     def __call__(self, **kwargs):
#         x = kwargs["x_new"]
#         tmax = bool(np.all(x <= self.xmax))
#         tmin = bool(np.all(x >= self.xmin))
#         return tmax and tmin
    
# mybounds = MyBounds()
# minimizer_kwargs = {"method": "L-BFGS-B"}
# res = optimize.basinhopping(get_correlation, params.values(),accept_test=mybounds,disp=True)#minimizer_kwargs=minimizer_kwargs,

# Nelde Mead method
# res = minimize(get_correlation,params.values() ,method='Nelder-Mead',options={'disp': True})

# SHGO
# minimizer_kwargs = {"method": "L-BFGS-B"}
# res = optimize.shgo(get_correlation,bnds,minimizer_kwargs=minimizer_kwargs,options={'disp': True})

# Print parameters
print("Parameters after optimisation")
print(res.x)

# Save Model
kp.save_model(res.x)

# Print correlations
print("Correlation first half (optimisation)")
final_results = pd.DataFrame(kp.get_model(res.x))
get_corr(final_results,'first',True)

print("Correlation second half (evaluation)")
get_corr(final_results,'last',True)

# Get images
data.model()
