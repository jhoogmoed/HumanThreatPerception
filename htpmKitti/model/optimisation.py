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
          'type/tram': 0.4,
          'type/misc': 0,
          'type/dc': 0.1,
          'prob/road': 0.1,
          'prob/city': 2,
          'prob/residential': 2,
          'imm/gain': 0.5,
          'imm/bias': 5}

i = 0
kp = kitti_parser(dataPath,drive,results_folder)


def get_correlation(x):
    global i
    results = pd.DataFrame(kp.get_model(x))

    # print("Iteration %s" %i)
    r = get_corr(results, 'first', False)

    i += 1
    return 1-(r['c']**2)

def get_corr(results, data_range, print_bool = False):
    ranges = {'first': data.response_mean_first,
              'last': data.response_mean_last}
    r = {}
    r['c'] = results['model_combination'].corr(ranges[data_range])
    r['t'] = results['model_type'].corr(ranges[data_range])
    r['p'] = results['model_probability'].corr(ranges[data_range])
    r['i'] = results['model_imminence'].corr(ranges[data_range])

    if print_bool == True:
        print("Combination corr : %s" % (r['c']**2))
        print("Type corr        : %s" % (r['t']**2))
        print("Probability corr : %s" % (r['p']**2))
        print("Imminence corr   : %s" % (r['i']**2))
    return r


# Random first guess
x0 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# x0 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]

bnds = ((0, 10), (0, 10), (0, 10), (0, 10), (0, 10),
        (0, 10), (0, 10), (0, 10), (0, 10),
        (0, 10), (0, 10), (0, 10),
        (0, 10), (0, 10))

# bnds = ((-100, 100), (-100, 100), (-100, 100), (-100, 100), (-100, 100),
#         (-100, 100), (-100, 100), (-100, 100), (-100, 100),
#         (-100, 100), (-100, 100), (-100, 100),
#         (-100, 100), (0, 100))


# Broyden-Fletcher-Goldfarb-Shanno method
# res = minimize(get_correlation, x0, method='TNC', bounds=bnds, options={
#                'disp': True, 'maxfun': None, 'ftol': 1e-9, 'gtol': 1e-9})

# res = minimize(get_correlation, list(params.values()), method='L-BFGS-B',bounds=bnds,options={'disp': True}) #
# res = minimize(get_correlation, x0, method='BFGS',options={'disp': True}) #

# Basinhopper method
# minimizer_kwargs = {"method": "L-BFGS-B",'bounds':bnds,'options':{'maxfun': 100000,'ftol': 1e-8,'gtol': 1e-8}}
# minimizer_kwargs = {"method" : "Nelder-Mead",'options':{'disp':True}}
# res = optimize.basinhopping(get_correlation, x0,minimizer_kwargs=minimizer_kwargs,disp=True)#,accept_test=mybounds
res = optimize.differential_evolution(get_correlation,bnds,disp=True)
# res = optimize.shgo(get_correlation,bnds,options={'disp':True})

# Nelde Mead method
# res = minimize(get_correlation,x0 ,method='Nelder-Mead',options={'disp': True})


# SHGO
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
get_corr(final_results, 'first', True)

print("Correlation second half (evaluation)")
get_corr(final_results, 'last', True)

data.model()
# Get images
# data.model()
