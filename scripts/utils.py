"""
SUPPLEMENTARY CODE FOR BOE SWP 848: 
Credit Growth, the Yield Curve and Financial Crisis Prediction: Evidence from a Machine Learning Approach 

This script provides utility functions that are used in the experiments
"""

import sys
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy.stats as st
import shap
import os
from operator import itemgetter
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from scipy.stats import mstats
import statsmodels.formula.api as smf


def shapley_kernel_wrapper(model, trainx, testx, config):
    """ This function is called by the prediction algorithms (ml_functions) 
    to compute the Shapley values. Note that the decision tree based models
    such as random forest do provide faster and exact (non-approximated)
     Shapley values with the TreeShapExplainer"""
    if config.exp_shap_background >= trainx.shape[0]:
        background = trainx
    else:
        background = shap.kmeans(trainx, config.exp_shap_background)

        # random sample of background values
        # ixx = np.random.choice(trainx.shape[0], config.exp_shap_background, replace=False)
        # background = trainx[ixx, :]
        # print(background.shape)

        explainer = shap.KernelExplainer(model.predict_proba, background)
        if isinstance(model, LogisticRegression):  # one background instance is enough if we use a linear model
            background = shap.kmeans(trainx, 1)
            backup_base = explainer.fnull
            explainer = shap.KernelExplainer(model.predict_proba, background)
            explainer.fnull = backup_base

    fnull_save = explainer.fnull[1]
    out = [explainer.shap_values(testx[i, :], l1_reg=0.0)[1] for i in np.arange(len(testx))]
    return np.vstack(out)

def exclude_periods(data, config):
    """ the function sets all cue values on the excluded periods to NA and returns
     an index of all the objects that should later be deleted.
     This way of processing ist best because all the preprocessing functions do not need
     consider cases where years are already missing """

    exclude_ix = np.zeros(len(data)) > 1

    if config.data_exclude_extreme_period:
        # exclude great depression | but NOT the beginnig of this crisis
        exclude_ix = exclude_ix | (np.array(data["year"] > 1933) & np.array(data["year"] < 1939))
        # exclude WW1
        exclude_ix = exclude_ix | (np.array(data["year"] > 1913) & np.array(data["year"] < 1919))
        # exclude WW2
        exclude_ix = exclude_ix | (np.array(data["year"] > 1938) & np.array(data["year"] < 1946))

    if not config.data_period in ['all', 'pre-ww2', 'post-ww2']:
        raise ValueError("time split is either 'all', 'pre-ww2', or 'post-ww2'")

    elif config.data_period  == 'pre-ww2':
        exclude_ix = exclude_ix | np.array(data["year"] > 1939)

    elif config.data_period == 'post-ww2':
        exclude_ix = exclude_ix | np.array(data["year"] < 1946)

    feature_names = set(data.columns.values).difference(set(['year', 'country', 'iso', 'crisis_id',
                                                             'crisis']))
    # set all feature values to NA in the excluded periods
    data.loc[exclude_ix, feature_names] = np.nan

    return data, exclude_ix




def create_grouped_folds(y, y_group, y_group_2=None, nfolds=10, reps=1, balance=True):
    """Create folds such that all objects in the same y_group and in the same
    y_group_2 (if not none) are assigned
    to the same fold
    :param np.array y: Binary outcome variable
    :param np.array y_group: Grouping variable, e.g crisis indicator
    :param np.array y: Second grouping variable (optional)
    :param int nfolds: Number of folds
    :param int reps: Number of replications of the n-fold cross-validation
    :param bool balance: If true, the outcome y is balanced as much as possible,
        i.e. that there are an equal number of
        positive observations in each fold
    """
    no = y.size
    iterable = list()
    for r in np.arange(reps):
        placed = np.zeros(no, dtype=np.int64)
        out = np.zeros(no, dtype=np.int64)*np.nan
        pos_counter = np.zeros(nfolds, dtype=np.int64)
        neg_counter = np.zeros(nfolds, dtype=np.int64)

        # go through objects in random order
        oo = np.random.choice(np.arange(no), no, replace=False)
        for i in oo:
            if placed[i] == 0:
                if not y_group_2 is None: # no verlap in year AND crisis_id
                    ix = np.where((y_group[i] == y_group) | (y_group_2[i] == y_group_2))[0]
                    for i in np.arange(25):
                        ix = np.where(np.in1d(y_group, y_group[ix]) | np.in1d(y_group_2, y_group_2[ix]))[0]
                else: # no overlap in crisis_id
                    ix = np.where(y_group[i] == y_group)[0]

                placed[ix] = 1

                if balance:
                    if y[i] == 1:
                        rf = np.random.choice(np.where(pos_counter == pos_counter.min())[0])
                        pos_counter[rf] += ix.size
                    else:
                        rf = np.random.choice(np.where(neg_counter == neg_counter.min())[0])
                        neg_counter[rf] += ix.size
                else:
                    rf = int(np.random.randint(0, nfolds, 1))

                out[ix] = rf

        for f in np.arange(nfolds):
            ix_train = np.where(out != f)[0]
            ix_test = np.where(out == f)[0]
            # make sure that test set contains both classes
            if (not (y[ix_test].mean() == 0)) & (not (y[ix_test].mean() == 1)):
                iterable.append((ix_train, ix_test))

    if len(iterable) < nfolds*reps:
        print("Repeat folding, some test set had zero variance in criterion")
        return create_grouped_folds(y, y_group, y_group_2=y_group_2,
                                  nfolds=nfolds, reps=reps, balance=balance)
    else:
        return iterable, out

def create_forecasting_folds(y, year, min_crisis=20, temp_resolution=1):
    """ Create folds for the forecasting experiment
     :param np.array y: Binary outcome variable
     :param np.array year: Time stamp for each observation
     :param int min_crisis: Minimum number of crisis observations in the training set.
     :param int temp_resolution: After how many years a new model should be trained.
     The default is 1, meaning, a new model is trained for every year.
    """
    iterable = list()
    last_train_year = list()
    uni_years = sorted(year.unique())
    del uni_years[-1]
    for i in np.arange(len(uni_years)):
        n_crisis = y[year <= uni_years[i]].sum()
        if (n_crisis >= min_crisis) & ((uni_years[i] % temp_resolution) == 0):
            ix_train = np.where(year <= uni_years[i-1])[0]
            ix_test = np.where(year > uni_years[i - 1])[0]

            if (len(ix_train) > 0) & (len(ix_test) > 0):
                iterable.append((ix_train, ix_test))
                last_train_year.append(uni_years[i])
    return iterable, last_train_year


def hyperparam_search(model, parameters, use='grid', n_jobs=1, cv=None, scoring=None,
                  n_iter=250, verbose=False):
    """Create a Grid or random search object that can be processed by Scikit-learn"""
    if isinstance(cv, int):
        raise ValueError("The argument cv should not be a number because the GridSearch algorithms"
                         " in sklearn do always create the same folds even with differnt random seeds."
                         " Rather you should pass folds that were created by our own function create_grouped_folds")


    if np.cumprod([len(parameters[x]) for x in parameters.keys()]).max() <= n_iter: # do use gridsearch if less than n_iter
        use = "grid" # combinations are tested
    if use == 'grid':
        model_out = GridSearchCV(model, parameters, n_jobs=n_jobs, cv=cv,
                                 scoring=scoring, verbose=verbose, iid = True)
    if use == 'random':
        model_out = RandomizedSearchCV(model, parameters, n_jobs=n_jobs, 
                                       cv=cv, n_iter=n_iter, scoring=scoring,

                                       verbose=verbose)
    return model_out


def write_file(data, file_name, path = '../results/', round=3, format=".txt",
              short_name=6, append=False, shorten = True):
      
    """ Writes a table as a text file to the hard drive """
    out = data.round(round)
    if not os.path.exists(path):
        os.mkdir(path)
    if isinstance(data, pd.core.frame.DataFrame):
        if shorten:
            out.columns = [x.replace("_" , "")[0:short_name] for x in out.columns.values]
            out.index = [str(x).replace('_', ' ')[0:short_name] for x in out.index.values]
        if append:
            out.to_csv(path + file_name + format , sep='\t', mode = 'a', header=True)
        else:
            out.to_csv(path + file_name + format, sep='\t', header=True)


def weights_from_costs(costs, Y):
    """  Weights observations according to the costs of the errors (as speceificed by the user) of the two classes.
    For example if the cost vector is [0.5, 0.5] and class A is twice as prevalent as class B,
    objects in class B will get twice the weight as objects in class A. """
    p1 = Y.mean()
    weights = {}
    weights[1] = costs[1] / (p1 * costs[1] + (1-p1) * costs[0])
    weights[0] = costs[0] / (p1 * costs[1] + (1 - p1) * costs[0])
    return weights


def downsample(X, Y, costs={0:0.5, 1: 0.5}, group=None):
    """ downsample the majority class according to the costs of the errors. """
    if group is None:
        group = np.arange(len(Y))
    weights = weights_from_costs(costs, Y)

    ix_pos = np.where(Y == 1)[0]
    n_pos = ix_pos.size
    ix_neg = np.where(Y == 0)[0]
    n_neg = ix_neg.size
    norm_w = min(weights.values())
    weights[0] = weights[0] / norm_w
    weights[1] = weights[1] / norm_w

    if weights[0] > weights[1]:
      ix_pos = np.random.choice(ix_pos, size=int(round(n_pos/weights[0])), replace=True)
    else:
      ix_neg = np.random.choice(ix_neg, size=int(round(n_neg/weights[1])), replace=True)
    ixses = np.concatenate((ix_pos, ix_neg))
    ixses = np.random.choice(ixses, size=ixses.size, replace=False)
    return X[ixses, :], Y[ixses], group[ixses]

def upsample(X, Y, group, costs):
    """ upsamples the minority class """
    weights = weights_from_costs(costs, Y)

    ix_pos = np.where(Y == 1)[0]
    n_pos = ix_pos.size
    ix_neg = np.where(Y == 0)[0]
    n_neg = ix_neg.size
    norm_w = min(weights.values())
    weights[0] = weights[0] / norm_w
    weights[1] = weights[1] / norm_w

    if weights[1] > weights[0]:
      ix_pos = np.random.choice(ix_pos, size=int(round(weights[1] * n_pos)), replace=True)
    else:
      ix_neg = np.random.choice(ix_neg, size=int(round(weights[0] * n_neg)), replace=True)
    ixses = np.concatenate((ix_pos, ix_neg))
    ixses = np.random.choice(ixses, size=ixses.size, replace=False)

    return X[ixses, :], Y[ixses], group[ixses]


# UTILITIES FOR TRANSFORMING VARIABLES #

def make_ratio(data_input, variables, denominator="gdp"):
    """ Computes the ratio of two variables. By detault the denominator is GDP. """

    names_out = []
    if isinstance(variables, str):
        variables = [variables]
    data = data_input.copy()
    for var in variables:
        varname = var + '_' + denominator
        data[varname] = data[var] / data[denominator]
        names_out.append(varname)
    return data, names_out

def make_shift(data_input, variables, type, horizon=5):
    """ Computes the change of a variable with respect to a certain horizon.
     :param pd.dDtaFrame data_input: Dataset. The tranformed variable will be appended to that data
     :param list of str variables : Name of the variables in data_input that will be transformed
     :param str type: Type of transformation. Either "absolute" (change) or "percentage" (change).
    """
    
    names_out = []
    data = data_input.copy()
    data_group = data.groupby('iso')
    if isinstance(variables, str):
        variables = [variables ]
    for var in variables:
        if type == "absolute":
            varname = var + '_rdiff' + str(horizon)
            data[varname] = data_group[var].diff(horizon)
        elif type == "percentage":
            varname = var + '_pdiff' + str(horizon)
            # attention objects must be ordered by year and country as they are in the original data
            data[varname] = data_group[var].apply(lambda x: lag_pct_change(x, h=horizon))
            #data[varname] = data_group[var].pct_change(horizon)

        names_out.append(varname)
    return data, names_out

def lag_pct_change(x, h):
    """ Computes percentage changes """
    lag = np.array(pd.Series(x).shift(h))
    return (x - lag) / lag



def make_level_change(data_input, variables, type, horizon=10):
    """ Computes the hamilton filter or difference from moving average
     :param pd.dDtaFrame data_input: Dataset. The tranformed variable will be appended to that data
     :param list of str variables: Name of the variables in data_input that will be transformed
     :param str type: Type of transformation. Either "ham" (hamilton filter) or "mad" (movgin average difference).
    """
    names_out = []
    data = data_input.copy()
    data_grouped = data.groupby('iso')
    if isinstance(variables, str):
        variables = [variables]
    for var in variables:
        if type == "mad":
            varname = var + '_mad'
            data[varname] = np.nan
            data_mad = pd.DataFrame(data_grouped.apply(mov_ave_diff, var, horizon), 
                                    columns=[varname])
            for iso in data_mad.index.values:
                data.loc[data.iso == iso, varname] = data_mad.loc[iso, varname]

        if type == "ham":
            varname = var + '_ham'
            data[varname] = np.nan
            data_ham = pd.DataFrame(data_grouped.apply(hamilton_filter, var, 2, 4),
                                    columns=[varname])
            for iso in data_ham.index.values:
                data.loc[data.iso == iso, varname] = data_ham.loc[iso, varname]
        names_out.append(varname)
    return data, names_out


def make_relative_change(data_input, variables, index='gdp', horizon=5):
    """ Computes the change of a variable relative the the change of of another variable
     :param pd.dDtaFrame data_input: Dataset. The tranformed variable will be appended to that data
     :param list of str variables: Name of the variables in data_input that will be transformed
     :param str index: Name of the variables to which the change is relative to (default is GDP)
    """
    names_out = []
    data = data_input.copy()
    data_grouped = data.groupby('iso')
    if isinstance(variables, str):
        variables = [variables]
    for var in variables:
        varname = var + '_idiff' + str(horizon)
        data[varname] = np.nan
        data_idiff = pd.DataFrame(data_grouped.apply(index_ratio_change, var,
                                                     index, horizon), columns=[varname])
        for iso in data_idiff.index.values:
            data.loc[data.iso == iso, varname] = data_idiff.loc[iso, varname]
        names_out.append(varname)
    return data, names_out


def mov_ave_diff(group, col, L=10):
    """ Computes the gap between a moving average (of length L) and the
    observations on a grouped data set """
    values = group[col].values
    N = len(values)
    out = np.zeros(N) * np.nan
    if N >= L:
        for i in range(N - L + 1):
            out[i + L - 1] = values[i + L - 1] - np.mean(values[i-1:i + L-1])
    return out

def index_ratio_change(group, ind1, ind2, l=5):
    """relative change of ind1 to ind2 over period l for group values."""

    val1 = group[ind1].values
    val2 = group[ind2].values
    N = len(val1)
    out = np.zeros(N) * np.nan

    if N >= l:
        for i in range(N - l):
            out[i + l] = (val1[i + l]  / val1[i]) / (val2[i + l] / val2[i]) - 1
    return out


def hamilton_filter(group, col, h=2, p=4, output="cycle"):  
    """ computes Hamilton filter
    : param int h: look-head period
    : param int p: number of lagged variables
    """

    x = group[col].values
    # note: Hamilton used 100 times x's logrithm in his employment data,
    # however, this is commented out because our data has negative values
    # x = 100*np.log(x)
    # Get the trend/predicted series
    trend = hamilton_filter_glm(x, h, p)
    if trend is not None:  # if dataframe passed is not full of nans
        # Get the cycle which is simple the original series substracted by the trend
        cycle = x - trend
        # Get the random walk series which is simply the difference between
        # the original series and the h look back
        df_x = pd.DataFrame(x)
        df_x_h = df_x.shift(h)
        random = df_x - df_x_h
    else:
        trend = x
        cycle = x
        random = x
    # Return required series in result, if all is selected then all results
    # are returned in a data frame
    if (output == "x"):
        return x
    elif (output == "trend"):
        return trend
    elif (output == "cycle"):
        return np.asarray(cycle)
    elif (output == "random"):
        return random
    elif (output == "all"):
        df = pd.DataFrame()
        df['x'] = x
        df['trend'] = trend
        df['cycle'] = cycle
        df['random'] = random
        df.plot()
        # pyplot.show()
        return df
    else:
        print ('\nInvalid output type')



def hamilton_filter_glm(x, h=2, p=4):
    """ Runs the linear model for the specification of the hamilton filter """
    # Create dataframe for time series to be smoothed, the independent variable y
    df = pd.DataFrame(x)
    df.columns = ['yt8']
    # Create matrix of dependent variables X which are the shifts of 8 period back
    # for 4 consecutive lags on current time t
    for lag in range(h, (h + p)):
        df['xt_' + str(lag - h + 1)] = df.yt8.shift(lag)
    # Getting the dependent varaibles X's index names
    X_columns = []
    for i in range(1, p + 1):
        new_s = 'xt_' + str(i)
        X_columns.append(new_s)
    # y and X variables for regression
    y = df['yt8']
    X = df[X_columns]

    xt_0 = pd.DataFrame(np.ones((df.shape[0], 1)))
    xt_0.columns = ['xt_0']
    X = xt_0.join(X)
    # Build the OLS regression model and drop the NaN
    try:
        if (sum(np.isnan(y)) != y.size):
            model = sm.OLS(y, X, missing='drop').fit()
            # Optional: print out the statistics
            model.summary()
            predictions = model.predict(X)
            return predictions
        else:
            return y
    except ValueError:
        pass

def all_same(items):
    return all(x==items[0] for x in items)

def sigmoid(x):
  return 1/(1 + np.exp(-x))

def sigmoidinv(x):
  return -np.log(1.0/x -1)

def normalize(data):
  return data.apply(normalizeV)

def normalizeV(x):
  x = x.astype(dtype="float32")
  return (x- np.nanmin(x))/(np.nanmax(x) - np.nanmin(x))

def performance_results(Y_in, Y_pred_in, threshold = 0.5):
    """ Computes performance metrics
    : param np.array Y_in: true values (0 or 1) of the response variable
    : param np.array Y_pred_in: predicted values of the response variable (between 0 an 1)
    : param float threshold: if Y_pred_in >= threshold, the predcited class is positive, otherwise negative
    """
    Y_pred_in = np.array(Y_pred_in, dtype=float)
    # types of Y and Y_pred are pd.Seres
    ix_miss = np.isnan(Y_pred_in)

    Y = Y_in[~ix_miss].copy()
    Y_pred = Y_pred_in[~ix_miss].copy()

    n = Y.size
    Y = Y * 1 # typecast boolean variables
    Y_pred = Y_pred * 1  # typecast boolean variables

    Y_bool = np.array(Y,dtype = "bool")
    Y_pred_bool = Y_pred >= threshold
    # True positive (tp), ture negative (tn), false positive (fp), false negative (fn)
    tp = np.logical_and(Y_bool, Y_pred_bool).sum()
    tn = np.logical_and(~Y_bool, ~Y_pred_bool).sum()
    fp = np.logical_and(~Y_bool, Y_pred_bool).sum()
    fn = np.logical_and(Y_bool, ~Y_pred_bool).sum()

    out = dict()
    if any(pd.isna(Y_pred)):
        return dict([
            ("accuracy", float("nan")),
            ("balanced", float("nan")),
            ("tp_rate", float("nan")),
            ("fp_rate", float("nan")),
            ("auc", float("nan")),
            ("tp", float("nan")),
            ("tn", float("nan")),
            ("fp", float("nan")),
            ("fn", float("nan")),
        ])
    else:
        out['tp'] = tp
        out['tn'] = tn
        out['fp'] = fp
        out['fn'] = fn
        out['accuracy'] = float(tp + tn) / float(n)
        if ((tp + fn == 0)):
            out['tp_rate'] = float("nan")
        else:
            out['tp_rate'] = float(tp) / float(tp + fn)

        if (tn + fp == 0):
            out['fp_rate'] = float("nan")
        else:
            out['fp_rate'] = 1 - float(tn) / float(tn + fp)

        out['balanced'] = (out["tp_rate"] + (1 - out["fp_rate"])) / 2.0

        if ((tp + fn > 0) & (tn + fp > 0)):
            out['auc'] = roc_auc_score(Y_bool, Y_pred)
        else:
            out['auc'] = float('nan')
        return out


def remove_file(file_path):
    """ removes file from hard drive """
    if os.path.exists(file_path):
        os.remove(file_path)