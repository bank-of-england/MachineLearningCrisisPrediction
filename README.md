# Code for the BOE SWP XXX

This repository includes the code used in the [Bank of England Staff Working Paper XXX](http::/) "__Credit Growth, the Yield Curve and Financial Crisis Prediction: Evidence from a Machine Learning Approach__" by Kristina Bluwstein, Marcus Buckmann, Andreas Joseph, Miao Kang, Sujit Kapadia, and Özgür Şimşek. 

In the paper, we develop early warning models for financial crisis prediction using machine learning techniques applied to macrofinancial data for 17 countries over 1870-2016. Machine learning models typically outperform logistic regression in out-of-sample prediction and forecasting experiments. We identify economic drivers of our machine learning models using a novel framework based on Shapley values, uncovering nonlinear relationships between the predictors and the risk of crisis.  Across all models, the most important predictors are credit growth and the slope of the yield curve, both domestically and globally. A flat or inverted yield curve is of most concern when coupled with high credit growth.


The dataset we use is the [Jordà-Schularick-Taylor Macrohistory Database](http://www.macrohistory.net/data/). We accessed the data using [this link](http://www.macrohistory.net/JST/JSTdatasetR3.xlsx).  

This repository does _not_ include all results of the experiments. Rather, it contains a small subset of the results to illustrate the empirical methodology and the implementation. 

Should you have any questions or spot a bug in the code, please email to marcus.buckmann@bankofengland.co.uk or raise an issue within the repository.



# Prerequisites 
The code has been developed and used under Python 3.6.5, Anaconda distribution and R 3.5.1. 

The file R script _R_installer.R_ in the _setup_ folder installs all necessary R packages.
The file _python_env.yml_ in the _setup_ folder contains the Anaconda virtual environment under which the experiments were run.
 

# Running the code

## Running the out-ofsample experiments 

The paper is based on two main empirical experiments: cross-validation and forecasting. These experiments are run using the respective Python scripts in the experiments folder.
In these files, the user can specify the models to be trained, the variables to be included, and how the variables should be transformed. The results are then written in the _results_ folder. The _pickle_ files in this folder contain all the results of the individual iterations. Each iteration uses a random seed and therefore partitions the data into training and test sets differently. 

The experiments do not need to be run at once. The user can terminate the experiments after a certain number of iterations and open a new Python session at another point in time. Then new pickle files will be added to the folder and will be merged with the previous experiments. 
The _.txt_ files in the results folder are written based on the information contained in all the _pickle_ files. 

The key files in the results folder are the following:
The _data[...].txt_ contains the dataset that is used in the experiment. This is not the raw dataset, rather all transformations and exclusions of data points have been applied.
- The _all_pred[...].txt_ contains the predictions for each observations, algorithm and iteration. 
- The _shapley_append[...].txt_ show the Shapley values for each observation, predictor and iteration. For each algorithm tested, an individual file is created.
- The _shapley_mean[...].txt_ file, shows the average Shapely values for each observation and predictor, averaged across all observations. For each algorithm tested, an individual file is created.
- The _mean_fold[...].txt_ shows the mean performance achieved in the individual folds. The files _mean_iter[...].txt_ and _mean_append[...].txt_ are similar. They just average the results differently. The former measures the performance for each iteration and averages the performance measures across iterations. The latter first averages the predicted values across all observations and then computes the performance on these averaged predicted values. 
- The files _se_fold[...].txt_ and _se_iter[...].txt_ show the standard errors of the respective performance results.


## Analyising the results 
The analysis and the regressions are conducted in _R_ and are based on the _.txt_ files in the results folder.
In the analysis folder, the files _analysis_cross_validation.R_ and _analysis_forecasting.R_ produce charts and regression models for the two types of experiments.

The Excel sheet _visual_params.xlsx_ in the analysis folder specifies visual characteristics of the plots. The user can alter the name, colour, or symbol of algorithms and variables shown in the charts.



