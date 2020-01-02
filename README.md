# Code for the Bank of England Staff Working Paper 848

This repository includes the code used in the [Bank of England Staff Working Paper 848](http::/) "__Credit Growth, the Yield Curve and Financial Crisis Prediction: Evidence from a Machine Learning Approach__" by Kristina Bluwstein, Marcus Buckmann, Andreas Joseph, Miao Kang, Sujit Kapadia, and Özgür Şimşek. 

In the paper, we develop early warning models for financial crisis prediction using machine learning techniques applied to macrofinancial data for 17 countries over 1870-2016. Machine learning models typically outperform logistic regression in out-of-sample prediction and forecasting experiments. We identify economic drivers of our machine learning models using a novel framework based on [Shapley values](https://bankunderground.co.uk/2019/05/24/opening-the-machine-learning-black-box/), uncovering nonlinear relationships between the predictors and the risk of crisis.  Across all models, the most important predictors are credit growth and the slope of the yield curve, both domestically and globally. 


The dataset we use is the [Jordà-Schularick-Taylor Macrohistory Database](http://www.macrohistory.net/data/). It is published under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](http://www.macrohistory.net/data/licence-terms/). We accessed [Version 3](http://www.macrohistory.net/JST/JSTdatasetR3.xlsx) from the dataset's [website](http://www.macrohistory.net/data/). This version is contained in the _data_ folder of this repository.  

The code is not intended as a stand-alone package. It can be used to reproduce the results of the paper. Parts of it may be transfered to other applications. No warranty is given. Please consult the licence file.

This repository does _not_ include all results of the experiments. Rather, it contains a small subset of the results to illustrate the empirical methodology and the implementation. 

Should you have any questions or spot a bug in the code, please send an email to marcus.buckmann@bankofengland.co.uk or raise an issue within the repository.


# Prerequisites 
The code has been developed and used under ```Python``` 3.6.5, Anaconda distribution and ```R``` 3.5.1. 

The ```R``` script _R_installer.R_ in the _setup_ folder installs all necessary ```R``` packages.
The file _python_env.yml_ in the _setup_ folder specifies the Anaconda virtual environment in which the experiments were run.
 

# Structure of the code

## Estimating the prediction models
The paper is based on two main empirical experiments: cross-validation and forecasting. These experiments are run using the respective ```Python``` scripts in the _experiments_ folder.
In these scripts, the user can specify the models to be trained, the variables to be included, and how the variables should be transformed. The results of the experiments are written to the _results_ folder. To obtain stable perfomance estimates, we repeat the experiments many times. For this repository, we repeated the 5-fold cross-validation 10 times. Each _pickle_ file in the _results_ folder contains the results of one iteration. Each iteration uses a different random seed and therefore partitions the data into a training and test set differently. 

The experiments do not need to be run at once. The user can terminate the experiments after a certain number of iterations and run more iterations at another point in time. Then, new pickle files will be added to the folder.
The _.txt_ files in the _results_ folder are written based on the information contained in all the _pickle_ files and are updated after each iteration.

The key files in the _results_ folder are the following:
The _data[...].txt_ contains the dataset that is used in the experiment. This is not the raw dataset, rather all transformations and exclusions of data points have been applied.
- The _all_pred[...].txt_ contains the predictions for each observations, algorithm and iteration. 
- The _shapley_append[...].txt_ show the Shapley values for each observation, predictor and iteration. For each algorithm tested, an individual file is created.
- The _shapley_mean[...].txt_ file, shows the average Shapely values for each observation and predictor, averaged across all observations. For each algorithm tested, an individual file is created.
- The _mean_fold[...].txt_ shows the mean performance achieved in the individual folds. The files _mean_iter[...].txt_ and _mean_append[...].txt_ are similar. They just average the results differently. The former measures the performance for each iteration and averages the performance measures across iterations. The latter first averages the predicted values across all observations and then computes the performance on these averaged predicted values. 
- The files _se_fold[...].txt_ and _se_iter[...].txt_ show the standard errors of the respective performance results.


## Analyising the results 
The analyses of the files in the _results folder_ are conducted in ```R```. In the _analysis_ folder, the files _analysis_cross_validation.R_ and _analysis_forecasting.R_ produce charts and regression models for the two types of experiments. 

The Excel sheet _visual_params.xlsx_ in the _analysis_ folder specifies visual characteristics of the plots. The user can alter the name, colour, and symbol of algorithms and variables shown in the charts.

# Disclaimer
This package is an outcome of a research project. All errors are those of the authors. All views expressed are personal views, not those of any employer.

