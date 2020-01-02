"""

SUPPLEMENTARY CODE FOR BOE SWP 848: 
Credit Growth, the Yield Curve and Financial Crisis Prediction: Evidence from a Machine Learning Approach 

This script produces performance results and the Shapley vlaue decomposition for the cross-validation experiment.

"""


import os, sys
import multiprocessing

# Set the main directory of the project
main_directory = "\\\\mafp-nwsrv\\data\\Advanced Analytics\\_People\Marcus Buckmann\\crisis_prediction\\public_code\\"
os.chdir(main_directory)
# main_directory = "your/path"
sys.path.insert(1, main_directory +'\\scripts')
# import models
from configure import *
from procedure import *

"""
We first create an instance of the Config class. This instance contains all the 
parameters of the empirical experiments, such as the proportion of the sample used 
for training or the names of the algorithms that are tested. Each parameter 
has a default value (see scripts/configure.py) - in the following we manually 
change a few parameters for our experiment.
"""

config = Config()
# Here, we overwrite some of the default parameter settings
config.data_horizon = 2  # Horizon of percentage and ratio changes (in years)
# algorithms with prefix r_ are trained in R
# set of algorithms tested in the paper:
# config.exp_algos = ["r_c50", "logreg", "extree", "forest", "svm_multi", "nnet_multi"]  
# select only few algorithms to reduce computation time
config.exp_algos = ["extree", "logreg"]

""" Here, we choose the predictors. We use the following convention.
 _pdiff indicators precentage changes
 _rdiff indicators X/GDP ratio changes
 """

config.data_predictors = ["drate", "cpi_pdiff", "bmon_gdp_rdiff", "stock_pdiff",
                                 "cons_pdiff" ,
                                 "pdebt_gdp_rdiff", "inv_gdp_rdiff", "ca_gdp_rdiff",
                                 "tloan_gdp_rdiff",
                                 "tdbtserv_gdp_rdiff", "global_loan", "global_drate"]

config.exp_do_shapley = True  # whether we want to estimate Shapley values
config.exp_n_kernels = multiprocessing.cpu_count() - 1 # number of CPU kernels used in parallel

df = create_data(config)
for i in np.arange(10): # repeat experiment 10 times
    o = Procedure(config, df_in=df, folder="results/baseline/")

