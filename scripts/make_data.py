"""
SUPPLEMENTARY CODE FOR BOE SWP 848: 
Credit Growth, the Yield Curve and Financial Crisis Prediction: Evidence from a Machine Learning Approach 

This script loads the Jorda-Schularick-Taylor Macrohistry database and transforms the variables.
"""
from utils import *
import pandas as pd
import numpy as np
import re

def create_data(config):
    """ Create the data set from the raw data from "http://www.macrohistory.net/data/"
    according to the specifications in the Config object"""
    
    # the data can be downlaoded directly:
    # df_jst = pd.read_excel('http://www.macrohistory.net/JST/JSTdatasetR3.xlsx', sheet_name="Data") 
    df_jst = pd.read_excel('data/JSTdatasetR3.xlsx', sheet_name="Data") 
    df = df_jst.copy()
    # rename variables
    df.rename(columns={
        "crisisJST": "crisis",
        'stir': 'srate',
        'ltrate': 'lrate',
        'iy': 'inv_gdp',
        'debtgdp': 'pdebt_gdp',
        'money': 'bmon',
        'narrowm': 'nmon',
        'tloans': 'tloan',
        'tbus': 'bloan',
        'thh': 'hloan',
        'tmort': 'mort',
        'stocks': 'stock',
        'hpnom': 'hp',
        'rconpc': 'cons'
    }, inplace=True)

    horizon = config.data_horizon
    predictors = config.data_predictors
    # we do not compute growth rates for the interest rates and the slope for the yield curve.
    no_change = ["drate",  "global_drate", "lrate", "srate"]

    # For the other predictors we compute growth rate (percentage change or ratio change)
    # and add the horizon (e.g. 2 year change) to the variable name
    predictors = [p + str(horizon) for p in predictors if p not in no_change] +\
        list(set(predictors).intersection(set(no_change)))

    df, exclude_ix = exclude_periods(df, config)  # exclude periods that are not normal economic conditions (e.g. WW2)

    df.loc[:, 'drate'] = df['lrate'] - df['srate']  # rate differential

    df.loc[:, 'pdebt'] = df['pdebt_gdp'] * df['gdp'] # compute public debt from public debt/gdp ratio
    df.loc[:, 'inv'] = df['inv_gdp'] * df['gdp']  # compute investment from investment/gdp ratio

    # Calculaute debt to service ratios
    df.loc[:, 'tdbtserv'] = df['tloan'] * df['lrate'] / 100.0
    
    pre_gdp_ratios = ['bmon', 'nmon', 'tloan', 'bloan',
                      'hloan', 'mort', 'ca', 'cpi', 'tdbtserv',
                      'inv', 'pdebt'] # vector of variables that will be transformed by GDP ratio
    df, gdp_ratios = make_ratio(df, pre_gdp_ratios, denominator='gdp')

    # here we compute the transformations and att the variables to the dataset df 

    # ratio change by GDP (rdiff)
    df, _ = make_shift(df, ["lrate", "srate", "drate"] + gdp_ratios,
                       type="absolute", horizon=horizon)
    # percentage change (pdiff)
    df, _ = make_shift(df, ['stock', 'cpi', 'hp', 'cons', 'gdp'] + pre_gdp_ratios,
                       type="percentage",
                       horizon=horizon)  # do not use absolute change
    # hamilton filter (ham)
    df, _ = make_level_change(df, ["cons"] + gdp_ratios, type="ham")


    # --- Computing global variables --- #

    # global credit growth (global_loan)
    for year in df["year"].unique():
        ix = df["year"] == year
        for country in df["iso"].unique():
            # computing the average across all countries but the selected one
            perc_pos = df.loc[ix.values & (df.iso != country).values,
                              "tloan_gdp_rdiff" + str(horizon)].mean()

            if not np.isnan(perc_pos):
                df.loc[ix.values & (df.iso == country).values,
                       "global_loan" + str(horizon)] = perc_pos

    # global slope of the yield curve
    for year in df["year"].unique():
        ix = df["year"] == year
        for country in df["iso"].unique():
            # computing the average across all countries but the selected one
            perc_pos = df.loc[ix.values & (df.iso != country).values, "drate"].mean()

            if not np.isnan(perc_pos):
                df.loc[ix.values & (df.iso == country).values, "global_drate"] = perc_pos

    # check whether we have created all features that will be used in the experiment
    if len(set(predictors).difference(set(df.columns.values))) > 0:
        raise ValueError('Features ' + ', '.join(set(predictors).difference(set(df.columns.values))) + "\n" +
                         "could not be found in the data!")




    # --- creating the 'landing zone' on the crisis outcome --- #
    years = df.year.values
    isos = df['iso'].values

    crisis_in = df_jst.crisisJST.values == 1
    crisis = crisis_in * 0
    for i, (yr, cr) in enumerate(zip(years, crisis_in)):
        if cr:
            for l in np.arange(1, 1 + config.data_years_pre_crisis): # flagging years before crisis as positive 
                if yr > (np.min(years) + l - 1):
                    crisis[i - l] = 1
            if config.data_include_crisis_year: 
                crisis[i] = 1  # crisis year

    # --- treatment of actual crisis and post crisis observations --- #
    i_keep = np.ones(len(df), dtype=int)
    for i, (yr, cr, iso) in enumerate(zip(years, crisis_in, df.iso)):
        if cr:
            if not config.data_include_crisis_year: 
                i_keep[i] = 0

            for j in range(1, 1 + config.data_post_crisis):
                k = i + j
                if (iso == df.iso[k]) & (k < len(df)):
                    i_keep[k] = 0

    # --- Give all observations of the same crisis the same ID --- #
    # This ID is used for cross-validation to make sure that the same crisis
    # is not in the training and test set
    # This function generalizes to any length of crises

    crisis_id = np.zeros(len(df))
    count = int(1)
    for i in np.arange(2, len(df)):
        if crisis[i] == 1:
            if not ((crisis[i - 1] == 1) & (isos[i] == isos[i - 1])):
                count += 1
            crisis_id[i] = count
    # All other observations get unique identifier
    crisis_id[crisis_id == 0] = np.random.choice(sum(crisis_id == 0),
                                                 size=sum(crisis_id == 0),
                                                  replace=False) + 2 + int(max(crisis_id))

    # create the data set
    features = df.loc[:, predictors]
    data = features
    data['crisis'] = crisis.astype(int)
    data['crisis_id'] = crisis_id.astype(int)
    data['year'] = years.astype(int)
    data['iso'] = isos # name of countries

    exclude_ix = exclude_ix | (i_keep == 0)
    data = data.loc[~exclude_ix, :]

    data = data.dropna()  # remove missing values
    data = data.reset_index(drop=True)  # update index
    return data
