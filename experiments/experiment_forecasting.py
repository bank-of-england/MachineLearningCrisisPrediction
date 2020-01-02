import os, sys
import multiprocessing
# Set the main directory of the project
# os.chdir("\\\\mafp-nwsrv\\data\\Advanced Analytics\\_People\Marcus Buckmann\\crisis_prediction\\public_code\\")
main_directory = "your/path"
os.chdir(main_directory)
sys.path.insert(1, main_directory +'\\scripts')


sys.path.insert(1, 'scripts')

# import models
from configure import *
from procedure import *

"""
We first create an instance of the Config class. This instance contains all
 the parameters of the empirical experiments, such as the proportion of the 
 sample used for training or the names of the algorithms that are tested.
 Each parameter has a default value, in the following we manually change a 
 few parameters for our experiment.
"""


config = Config()
# Here, we overwrite some of the default parameter settings
config.data_horizon = 2  # Horizon of percentage and ratio changes (in years)
# algorithms with prefix r_ are trained in R
# set of algorithms tested in the paper
config.exp_algos = ["r_c50", "logreg", "extree", "forest", "svm_single", "nnet_single"]  
# select only few algorithms to reduce computation time
config.exp_algos = ["extree", "forest"]


""" Here we choose the predictors. We use the following convention.
 _pdiff indicators precentage changes
 _rdiff indicators X/GDP ratio changes
 """

# choose predictors
config.data_predictors = ["drate", "cpi_pdiff", "bmon_gdp_rdiff", "stock_pdiff",
                                 "cons_pdiff" ,
                                 "pdebt_gdp_rdiff", "inv_gdp_rdiff", "ca_gdp_rdiff",
                                 "tloan_gdp_rdiff",
                                 "tdbtserv_gdp_rdiff", "global_loan", "global_drate"]

config.exp_do_shapley = False  # whether we want to estimate Shapley values
config.exp_bootstrap = "up"  # we upsample all training sets
config.exp_n_kernels = multiprocessing.cpu_count() - 1 # number of CPU kernels used in parallel

df = create_data(config)  # create dataset

min_crisis = 20  # minimum number of crises observations in the training set, before the forecasting starts
f, yearsplits = create_forecasting_folds(df["crisis"], df["year"], min_crisis=min_crisis)
# yearsplits = [y for y in yearsplits if y > 1990]

for year in yearsplits: # the year reflects the last year in the training set
    config.exp_year_split = year
    for i in np.arange(5): # repeat experiment 5 times
        o = Procedure(config, df_in=df,
                       folder="results/forecasting_" + config.exp_bootstrap + "/",
                       save_data=max(yearsplits) == year)

