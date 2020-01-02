"""
SUPPLEMENTARY CODE FOR BOE SWP 848: 
Credit Growth, the Yield Curve and Financial Crisis Prediction: Evidence from a Machine Learning Approach 

This script contains the functions that conduct the cross-validation and forecasting experiments.
"""

import sys
import random
import xarray as xr
import time
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from ml_functions import *



def train_and_test(df, config):

    """ Low level experiment function
    It samples the training and test data, trains and test the prediction models 
    (their function are in ml_functions.py)
    either with our without the computation of Shapley values
    
     :param pd.DataFrame df: Data set with an arbitrary number of predictors and the columns
        crisis, crisis_id, year, and iso.
     :param Config config: Configuration file that specifies the experimental setup
    """

    algo_names = config.exp_algos
    nfolds = config.exp_nfolds
    hyper_folds = config.exp_hyper_folds
    rep_cv = config.exp_rep_cv

    X = df.copy()
    Y = X['crisis'].values.astype(int)
    crisis_id = X['crisis_id'].values.astype(int)
    years = X.year.values

    X = X.drop(columns=['crisis', 'crisis_id', 'year', 'iso'])
    feature_names = X.columns.values
    X = np.array(X)

    output_ypred = pd.DataFrame(index=np.arange(len(X)), columns=algo_names)

    # prepare a list that contains all the results
    model_out = {x: list() for x in algo_names}
    fits_out = {x: list() for x in algo_names}
    time_out = {x: list() for x in algo_names}
    params_out = {x: list() for x in algo_names}

    if config.exp_do_shapley:
        output_shapley = xr.DataArray(np.zeros((len(algo_names),
                                                X.shape[0],
                                                X.shape[1])) * float("nan"),
                                      [('algorithms', algo_names),
                                       ("instances", np.arange(X.shape[0])),
                                       ("features", feature_names)])
        output_shapley_fnull = pd.DataFrame(index=np.arange(len(X)),
                                            columns=algo_names)
    else:
        output_shapley = None
        output_shapley_fnull = None

    if config.exp_shapley_interaction:
        inter_algos = list(set(algo_names).intersection(set(["extree", "forest"])))
        output_shapley_inter = xr.DataArray(np.zeros((len(inter_algos),
                                                X.shape[0],
                                                X.shape[1], X.shape[1])) * float("nan"),
                                      [('algorithms', inter_algos),
                                       ("instances", np.arange(X.shape[0])),
                                       ("features1", feature_names),
                                       ("features2", feature_names)])
    else:
        output_shapley_inter = None

    results = {'predictions': output_ypred,
               "fits": fits_out,
               'ix_test': [],
               'ix_train': [], 
               'models': model_out,
               'parameters': params_out, "time": time_out,
               'shapley': output_shapley,
               "shapley_fnull": output_shapley_fnull,
               "data": [],
               'shapley_inter': output_shapley_inter}


    if config.exp_year_split is None:
        # Create the cross-validation folds
        if (config.exp_id == "none"):
            folds, _ = create_grouped_folds(y=Y, y_group=np.arange(Y.shape[0]),
                                          nfolds=nfolds, reps=1)

        if(config.exp_id == "crisis"):
            folds, _ = create_grouped_folds(y=Y, y_group=crisis_id, nfolds=nfolds,
                                          reps=1)

        if (config.exp_id == "year_and_crisis"):
            folds, _ = create_grouped_folds(y=Y, y_group=crisis_id, y_group_2=years,
                                          nfolds=nfolds, reps=1)
        if (config.exp_id == "year"):
            folds, _ = create_grouped_folds(y=Y, y_group=years, nfolds=nfolds,
                                          reps=1)


    else:
        # If we have a year splitting training and test set:
        nfolds = 1

    # run through the folds
    for f in np.arange(nfolds):
        sys.stdout.write('.')

        if config.exp_year_split is None:
            # obtain training and test set from the previously defined folds
            ix_train = list(folds[f][0])
            ix_test = list(folds[f][1])
        else:
            # observations before splitting year are used for training, 
            # the remaining observations for testing
            ix_train = list(np.where(years <= config.exp_year_split)[0])
            ix_test = list(np.where(years > config.exp_year_split)[0])

        # a random shuffle of the order of observations.
        ix_train = np.array(random.sample(ix_train, len(ix_train)))
        ix_test = np.array(random.sample(ix_test, len(ix_test)))

        if config.exp_bootstrap == "naive":
            ix_train = np.random.choice(ix_train, size=len(ix_train), replace=True)

        if config.exp_bootstrap in ["up", "down"]:
            ix_pos = ix_train[Y[ix_train] == 1]
            ix_neg = ix_train[Y[ix_train] == 0]
            replacer = False
            if config.exp_bootstrap_replace == "yes":
                replacer = True # whether to sample the minoritty class by replacement as well

            if config.exp_bootstrap == "up":
                if len(ix_neg) > len(ix_pos):
                    ix_train = np.concatenate((np.random.choice(ix_neg,
                                                                size=len(ix_neg), 
                                                                replace=replacer),
                                               np.random.choice(ix_pos,
                                                                size=len(ix_neg),
                                                                replace=True)))

                if len(ix_pos) > len(ix_neg):
                    ix_train = np.concatenate((np.random.choice(ix_pos,
                                                                size=len(ix_pos),
                                                                replace=replacer),
                                               np.random.choice(ix_neg,
                                                                size=len(ix_pos), 
                                                                replace=True)))

            if config.exp_bootstrap == "down":
                if len(ix_neg) > len(ix_pos):
                    ix_train = np.concatenate((np.random.choice(ix_pos
                                                                , size=len(ix_pos),
                                                                replace=replacer),
                                               np.random.choice(ix_neg,
                                                                size=len(ix_pos),
                                                                replace=False)))

                if len(ix_pos) > len(ix_neg):
                    ix_train = np.concatenate((np.random.choice(ix_neg,
                                                                size=len(ix_neg),
                                                                replace=replacer),
                                               np.random.choice(ix_pos,
                                                                size=len(ix_neg), 
                                                                replace=False)))

        results["ix_train"].append(ix_train)
        results["ix_test"].append(ix_test)

        dat = dict(train_x=X[ix_train, :],
                   test_x=X[ix_test,: ],
                   train_y=Y[ix_train],
                   test_y=Y[ix_test],
                   train_crisis_id=crisis_id[ix_train])

        # The error costs (false positve, false negative) determine how
        # the instances are weighted in the training set
        if isinstance(config.exp_error_costs, dict):
            class_costs = config.exp_error_costs
        elif config.exp_error_costs == "balanced": # objects are weighted,
            # such that the weighted proportion of objects
            #  contribute equally to the training set
            class_costs = {0: dat["train_y"].mean(), 1: 1 - dat["train_y"].mean()}
        elif config.exp_error_costs == "0.5":
            class_costs = {0: 0.5, 1: 0.5} # each object has the same weight.

        if config.exp_do_upsample: # upsample training set
            dat["train_x"], dat["train_y"], 
            group = upsample(dat["train_x"],
                             dat["train_y"],
                             group=dat["train_crisis_id"], costs=class_costs)
            class_costs_use = {0: 0.5, 1: 0.5}
            sample_weight = compute_sample_weight(class_costs_use, dat["train_y"])
            cv_hyper, cv_fold_vector = create_grouped_folds(dat['train_y'],
                                                          group, nfolds=hyper_folds,
                                                          reps=rep_cv)
        else: # create folds for the hyperparater search. (Nested cross-validation)
            group = dat["train_crisis_id"]
            class_costs_use = class_costs
            class_weight = weights_from_costs(class_costs_use, dat["train_y"])
            sample_weight = compute_sample_weight(class_weight, dat["train_y"])
            cv_hyper, cv_fold_vector = create_grouped_folds(dat['train_y'],
                                                          dat["train_crisis_id"],
                                                          nfolds=hyper_folds,
                                                          reps=rep_cv)

        # rescale all variables according to the training set
        scaler = StandardScaler()
        dat['train_x_scaled'] = scaler.fit_transform(dat['train_x'])
        dat['test_x_scaled'] = scaler.transform(dat['test_x'])

        results["data"].append(dat)

        python_algos = [x for x in algo_names if x[0:2] != "r_"]
        # Train and test PYTHON prediction models 
        data = {"trainx": dat['train_x_scaled'] ,
                "trainy": dat['train_y'],
                "testx": dat['test_x_scaled']
                }
        
        for algo in python_algos:
            out = globals()[algo](data,
                        config=config,
                        cv_hyper=cv_hyper,
                        group=group,
                        sample_weight=sample_weight,
                        do_cv = False, name = algo)
        
            append_results(results, out)


        # Some models are trained in R, we call the R script from here.
        # The R script loads the  training and test set as a csv,
        # We save them here and then save the results as a csv as well.
        r_algos = [x for x in algo_names if x[0:2] == "r_"]
        
        if len(r_algos) > 0:
            try:
                os.makedirs("r_data")
            except:
                pass

            train_r = pd.DataFrame(dat['train_y'],
                                   columns=["y"]).join(pd.DataFrame(dat['train_x']))
            test_r = pd.DataFrame(dat['test_y'],
                                  columns=["y"]).join(pd.DataFrame(dat['test_x']))
            ident = np.random.randint(100000000) # random name_suffix as a unique
            # identifier for the experiment
           
            train_r.to_csv('r_data/train_in' + str(ident) + '.csv', sep='\t')
            test_r.to_csv('r_data/test_in' + str(ident) + '.csv', sep='\t')
            pd.Series(cv_fold_vector).to_csv('r_data/cv_fold_vector' \
                     + str(ident) + '.csv', sep='\t', header = False)
               
    
            for r_algo in r_algos:
                out = rmodel(r_algo, class_costs_use, config, ident, 5)
                append_results(results, out)
                
            while True: 
                try:    
                    remove_file('r_data/out_to_python_' + str(r_algo) + str(ident) + '.csv') # remove R file
                    remove_file('r_data/train_in' + str(ident) + '.csv')
                    remove_file('r_data/test_in' + str(ident) + '.csv')
                    remove_file('r_data/cv_fold_vector' + str(ident) + '.csv')
                    break
                except:
                    pass
                
    return results

def append_results(results, add):

    """Appends the results obtained for a single fold to the previous results"""
    name = add["name"]
    ix_test = results["ix_test"][-1] # last element in list

    results['predictions'].loc[ix_test, name] = add["pred"]
    results['fits'][name].append(add["fit"])

    results["models"][name].append(add["model"])
    results["parameters"][name].append(add["hyper_params"])
    results["time"][name].append(add["time"])
    if not add["shapley"] is None:
        results["shapley"].loc[name, ix_test, :] = add["shapley"]

        if "shapley_inter" in add.keys():
            if not add["shapley_inter"] is None:
                results["shapley_inter"].loc[name, ix_test, :, :] = add["shapley_inter"]