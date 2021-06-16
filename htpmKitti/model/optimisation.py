#!/usr/bin/env python3
from scipy import optimize
import numpy as np
import pandas as pd


from compare import analyse
from services import kitti_parser

from scipy.optimize import minimize


# Set data folder paths
dataPath = '/home/jim/HDDocuments/university/master/thesis/ROS/data/2011_09_26'
drive = '/test_images'
results_folder = '/home/jim/HDDocuments/university/master/thesis/results/'
mergedDataPath = '/home/jim/HDDocuments/university/master/thesis/results/filtered_responses/merged_data.csv'
modelDataPath = 'model_results.csv'

# Get responses from participants
data = analyse(dataPath, drive, mergedDataPath, modelDataPath, results_folder)
data.get_responses()
data.split()

# Start values
params = {'type/car': 0.2,
          'type/van': 0.4,
          'type/truck': 0.6,
          'type/ped': 1,
          'type/pedsit': 0.2,
          'type/cycl': 1,
          'type/tram': 0.6,
          'type/misc': 0.2,
          'type/dc': 0,
          'prob/city': 0.32,
          'prob/residential': 0.32,
          'prob/road': 1.43,
          'imm/gain': 1, 
          'imm/bias': 0.5} #a*(distance/velocity)**(1/b)

i = 0
kp = kitti_parser(dataPath,drive,results_folder)


def get_correlation(x):
    global i
    results = pd.DataFrame(kp.get_model(x))

    # print("Iteration %s" %i)
    r = get_corr(results, 'training', False)

    i += 1
    return 1-(r['c'])

def get_corr(result, data_range, print_bool = False):
    ranges = {'training': data.data_training.mean(skipna = True),
              'testing': data.data_testing.mean(skipna = True)}
    middle_frame = int(round(result.shape[0])/2)-1
    # print("middle frame {}".format(middle_frame))
    
    results = {'training': result.loc[0:middle_frame,:],
              'testing': result.loc[middle_frame:result.shape[0],:]}
    # print(results[data_range])
    # print(ranges[data_range])
    r = {}
    r['c'] = results[data_range]['model_combination'].corr(ranges[data_range],'pearson')
    r['t'] = results[data_range]['model_type'].corr(ranges[data_range])
    r['p'] = results[data_range]['model_probability'].corr(ranges[data_range])
    r['i'] = results[data_range]['model_imminence'].corr(ranges[data_range])

    if print_bool == True:
        print("Combination corr : %s" % (r['c']))
        print("Type corr        : %s" % (r['t']))
        print("Probability corr : %s" % (r['p']))
        print("Imminence corr   : %s" % (r['i']))
    return r


# Random first guess
# x0 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# x0 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
x0 = list(params.values())

# bnds = ((0, 10), (0, 10), (0, 10), (0, 10), (0, 10),
#         (0, 10), (0, 10), (0, 10), (0, 10),
#         (0, 10), (0, 10), (0, 10),
#         (0, 10), (0, 10))

bnds = ((-100, 100), (-100, 100), (-100, 100), (-100, 100), (-100, 100),
        (-100, 100), (-100, 100), (-100, 100), (-100, 100),
        (-100, 100), (-100, 100), (-100, 100),
        (-100, 100), (0, 100))


# Broyden-Fletcher-Goldfarb-Shanno method
# res = minimize(get_correlation, x0, method='TNC', bounds=bnds, options={
#                'disp': True, 'maxfun': None, 'ftol': 1e-9, 'gtol': 1e-9})
# res = minimize(get_correlation, list(params.values()), method='L-BFGS-B',bounds=bnds,options={'disp': True}) #


# Basinhopper method
# minimizer_kwargs = {"method": "L-BFGS-B",'options':{'maxfun': 100000,'ftol': 1e-8,'gtol': 1e-8}}
# res = optimize.basinhopping(get_correlation, x0,minimizer_kwargs=minimizer_kwargs,disp=True)#,accept_test=mybounds

# Differential evolution method
res = optimize.differential_evolution(get_correlation,bnds,disp=True,strategy="best1exp",popsize=20)
# res = optimize.shgo(get_correlation,bnds,options={'disp':True})

# Nelde Mead method
# res = minimize(get_correlation,x0 ,method='Nelder-Mead',options={'disp': True})

# SHGO method
# minimizer_kwargs = {"method": "L-BFGS-B"}
# res = optimize.shgo(get_correlation,bnds, n=1000,minimizer_kwargs=minimizer_kwargs,sampling_method='simplicial',options={'disp': True})

# Print parameters
print("Parameters after optimisation")
print(res.x)

# Save Model
kp.save_model(res.x)

# Print correlations
print("Correlation first half (optimisation)")
final_results = pd.DataFrame(kp.get_model(res.x))
get_corr(final_results, 'training', True)

print("Correlation second half (evaluation)")
get_corr(final_results, 'testing', True)

data.model()
# Get images
# data.model()
