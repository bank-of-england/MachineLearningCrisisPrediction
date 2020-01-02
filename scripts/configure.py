"""

SUPPLEMENTARY CODE FOR BOE SWP 848: 
Credit Growth, the Yield Curve and Financial Crisis Prediction: Evidence from a Machine Learning Approach 

"""

import numpy as np
import hashlib

class Config:
    """ Creates a config object that specifies how the data is processed and how the experiment is run.
        The default values assigned here can be altered by the user in the experiment files (see experiments folder)
     """
    

    def __init__(self):

         # Path of R is needed as the decision tree is trained in R.
        self.r_path = None # e.g. 'C:\\Program Files\\R\\R-3.5.1\\bin\\x64\\Rscript'
        
        #### The following parameters determine how the data is processed ####
        self.data_predictors = ["drate", "cpi_pdiff" , "bmon_gdp_rdiff", "stock_pdiff",
                                 "cons_pdiff" ,
                                 "pdebt_gdp_rdiff", "inv_gdp_rdiff", "ca_gdp_rdiff",
                                 "tloan_gdp_rdiff",
                                 "tdbtserv_gdp_rdiff", "global_loan"]  # Names of the indicators used as predictors
        # pdiff: percentage change
        # rdiff ratio change. Change of the variable relative to the change in GDP
        self.data_horizon = 2  # Horizon of percentage and ratio changes (in years)

        self.data_period = 'all'  # The time frame investigate. Either 'all' observations,
        # or 'pre-ww2' or 'post-ww2'.
        self.data_exclude_extreme_period = True  # Whether to exclude WW1, WW2 and 
        # the Great Depression
        self.data_include_crisis_year = False  # Whether to exclude the actual crisis
        # observation and only predict years a head of a crisis
        self.data_years_pre_crisis = 2  # number of years before a crisis for
        # which outcome is set positive
        self.data_post_crisis = 4  #  How many observations (in years) after the 
        # crisis should be deleted to avoid post-crisis bias
        
        
        
        #### The following parameters determine experimental details ####

        self.exp_n_kernels = 1  # The number of kernels of the CPU used in parallel
        self.exp_nfolds = 5  # Number of folds in the cross-validation experiment.

        self.exp_algos = ['extree', "log"]  # list of algorithms that are tested in the experiment

        self.exp_year_split = None  # If 'None' the cross-validation experiment is run.
        # If it is a year y all instances up to that year are used for training and 
        # the following observations for testing the model. The latter option is used for forecasting

        self.exp_id = "crisis"
        # This variable specifies constraints for the cross-validation
        # 'no': no constraint used
        # 'crisis': the observation of a crisis (by default 1-2 years before crisis)
        #   are assigned to the same fold
        # 'year': all observations of a certain year are assigned to the same fold
        # 'year_and_crisis' combination of the two constraints above

        # Hyperparameter search
        self.exp_verbose = 0  # Determines how verbose the output of the hyperparameter search is. 
        self.exp_hyper_folds = 5  # Number of folds in the cross-validation of the hyperparameters
        self.exp_rep_cv = 1  # How often the cross-validation of the hyperparameters is repeated.
        self.exp_search = "grid"  # Either we use full 'grid' search or 'random' search
        self.exp_n_iter_rsearch = 250  # How many hyperparamter combinations are tested in the random search
        self.exp_optimization_metric = 'roc_auc'  # Metric that is optimized in the hyperparameter search

        # Shapley
        self.exp_do_shapley = True  # Whether Shapley values are computed
        self.exp_shap_background = 50  # Number of background samples used by the Shapley Kernel explainer
        self.exp_shapley_interaction = False  # Whether interactions of Shapley values are computed

        self.exp_error_costs = "0.5"  # cost associated with the false positive 
        # and false negative error. If set to '0.5', both errors are treated as equally important
        #  If set to 'balanced' the error of the minority classes is upweighted 
        # such that the product of the error-weight and the proportion of objects
        # in the class is equivalent for both classes.

        # The error costs can also be set to arbitrary values using a dictionary,
        # e.g. {0: 0.1, 1: 0.9}. This means that the error in the positive class
        # are 9 times more important than the error in the negative class.

        self.exp_do_upsample = False  # whether the minority class is upsampled 
        # according to the error costs (see above). If False, the objects are weighted
        # according to the error costs. Note that the weighting of objects is not
        # supported by all algorithms.

        self.exp_bootstrap = "no" # bootstrapping the training set with the options
        # no (no bootstrappoing), up (upsampling), down (downsampling)
        self.exp_bootstrap_replace = "no" # # whether to resample the minority class by replacement as well


    def _make_name(self, name_appx=""):
        """Creates a descriptive name according to the configuration.
        This name is used when saving the files in the results folder.
        It is based on some of the experiments parameters but the user
        can also add a suffix to the name with the name_appx argument
        """
        name_data = name_appx +  str(self.data_horizon) + "_" + str(self.exp_id)

        if self.exp_year_split is None:
            expName = "CV"
        else:
            expName = "year" + str(int(self.exp_year_split))
        name = self.data_period + "_" + str(expName) + "_" + str(name_data)

        if self.data_include_crisis_year:
            name = name + "crsIncl_"

        if self.exp_do_shapley:
            name = name + "SHAP_"


        return name
