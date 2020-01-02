"""
SUPPLEMENTARY CODE FOR BOE SWP 848: 
Credit Growth, the Yield Curve and Financial Crisis Prediction: Evidence from a Machine Learning Approach 
"""


from make_data import *
from train_and_test import *
import xarray as xr
import subprocess
import pickle
import math
import warnings
import yaml
from utils import *
class Procedure:

    def __init__(self, config, df_in=None, file_name=None, name_suffix="", folder=None,
                 keep_models=False, keep_data=False, save_data=True, skipExperiment=False):

        """This is the Procedure class. Here the experiments are conducted and
        the results are saved to the hard drive.

        :param str Config: Config objects specifying data processing and the experimental setup
        :param pd.Data.Frame df_in: The input data. If 'None', the data is generated in this function
        :param str file_name: The name given to the output files. If 'None', the name is given by the Config._make_name
        :param str name_suffix: Characters that are appended to the file name. This parameter is useful when
            you want to add something to the automatically generated file names by Config._make_name.
        
        :param str folder: The folder where the results are saved. If 'None'
            the results are saved in the working directory
        
        :param Boolean keep_models: Whether the actual models should be saved in pickle files.
            This can require substantial disk space and is not recommended
        :param Boolean keep_data:  Whether all training and test set partitions should
            be saved in the pickle. This can also require substantial space and is not recommended
        :param Boolean save_data:  Whether the dataset on which the algorithms are trained
            and tested should be written to the hard drive

        :param Boolean skipExperiment: Whether the experiment should be skipped
            and only the existing pickle files should be aggregated.

         """
        self.collected = False # Indicates whether the results have been collected from the hard drive.
        self.name_suffix = name_suffix
        self.keep_models = keep_models
        self.keep_data = keep_data
        self.save_data = save_data
        self.config = config

        if file_name is None:
            self.file_name = config._make_name(self.name_suffix)
        else:
            self.file_name = file_name
        if folder is None:
            self.folder = "results/" + config._make_name(self.name_suffix) + "/"
        else:
            self.folder = folder
        try:
            os.makedirs(self.folder)
        except:
            pass

        with open(self.folder + 'config.yml', 'w') as outfile:
            yaml.dump(config, outfile, default_flow_style=False)

        # Create dataset
        if not df_in is None:
            print('Data set given with ')
            self.df = df_in.copy()
        else:
            self.df = create_data(self.config)
            print('Data set created with ')

        print('    ' + str(np.sum(self.df.crisis.values == 1)) + " Crises")
        print('    ' + str(np.sum(self.df.crisis.values == 0)) + " No crises")
        print('    ' + str(self.df.shape[1] - len(["year", "crisis",
                           "crisis_id", "iso"])) + " Features")



        self.metrics = ["auc", "accuracy", "balanced", "tp_rate", "fp_rate",
                        "tp", "tn", "fp", "fn"] # performance metrics
        self.results = list()
        if self.save_data:
            write_file(self.df, "data_" + self.file_name, # save the dataset
                      path=self.folder, shorten=False)
        self.X = self.df.copy()
        self.Y = self.X['crisis'].values.astype(int)
        self.X = self.X.drop(columns=['crisis', 'crisis_id', 'year', 'iso'])
        self.feature_names  = self.X.columns.values
        self.X = np.array(self.X)

        # run the experiment
        if not skipExperiment:
            self._do_experiment()
            self._save_pickle(keep_models=self.keep_models, keep_data=self.keep_data)
        # write the results to the hard drive
        self._write_results()
        # write the Shapley values to the hard drive
        if self.config.exp_do_shapley:
             for m in self.config.exp_algos:
                 # we cannot do Shapley decomposition for those models trained in R:
                if not m in ["r_elnet", "r_cart", "r_c50"]:
                    self._write_shapley(model_name=m)

    def _do_experiment(self):
        """Conduct the experiment"""
        self.results.append(train_and_test(self.df, self.config))
        self.config.exp_nrep = len(self.results)


    def _collect_results(self):
        """Read the results from the pickles of the individual iterations saved on the hard drive
        and adds results to self.results"""
        if not self.collected:
            self.results = []
            self.collected = True
            if os.path.exists(self.folder):
                file_list = os.listdir(self.folder)
                pickle_names = [s for s in file_list if "pickle_" in s]
                pickle_names = [s for s in pickle_names if self.file_name in s]

                for p in pickle_names:
                    o_old = pickle.load(open(self.folder + p, 'rb'))
                    self._add_results(o_old)
                    
        if not all_same([i["predictions"].shape for i in self.results]):
            raise ValueError("You try to merge results of different experiments. This is not possible.")
        
        if not all_same([set(i["predictions"].columns) for i in self.results]): # results of different sizes are merged
                    raise ValueError("You try to merge results of different models. This is not possible.")
                
    

    def _add_results(self, old_object):
        # the current results are added to a new object!
        if self.file_name != old_object.file_name:
            print("Experimental configurations do not match")
            return None

        self.results = self.results + old_object.results

    def _write_results(self, ix=None):
        """Write the results to the hard drive. This function collects the results from
        previous iterations (saved in the pickle files) of this experiment (by calling _collect_results)
        and processes them  to produce the csv files.
        
        :param boolean list ix: Used to subset the observations such that the results
            are saved and performance metrics are computed only for these observations.
            If it is 'None' all observations are used. """


        if ix is None: # ix is used
            ix_select = [True] * self.df.shape[0] # select all observations
        else:
            ix_names = self.df[["year", "iso"]].apply(lambda x: str(x[0]) + "_" + str(x[1]), axis=1).values
            ix_select = [x in ix for x in ix_names.tolist()]

        self._collect_results()

        out_pred = list()
        for i in range(len(self.results)):
            dout = self.results[i]["predictions"].copy()
            dout["crisis"] = self.df.crisis
            dout["year"] = self.df.year
            dout["iso"] = self.df.iso
            dout["iter"] = i

            # identify fold
            folds = [np.where(np.isin(np.arange(len(self.df.crisis)), 
                                      self.results[i]["ix_test"][j]), j + 1, 0) for j in
                     np.arange(len(self.results[i]["ix_test"]))]

            folds = np.vstack(folds).sum(0)

            dout["fold"] = folds
            out_pred.append(dout)
                
        pred_all = pd.concat(out_pred)
        write_file(pred_all , "all_pred" + "_" + self.file_name,
                  path=self.folder, shorten=False)

        # Three types of processing the results of the repeated cross-validation experiment:

        # ---- 1 --- #
        # Then performance metrics are computed for each iteration across all objects. Then the performance
        # metrics are averaged across iterations.
        output_metric_across = xr.DataArray(np.zeros((len(self.results), len(self.config.exp_algos),
                                                           len(self.metrics))) * float("nan"),
                                                 [('iterations', np.arange(len(self.results))),
                                                  ("algorithms", self.config.exp_algos),
                                                  ("metrics", self.metrics)])
        for r in np.arange(len(self.results)):
            res_across_folds = self.results[r]['predictions'].apply(lambda x:
                                                    np.array([performance_results(self.df.crisis.values[ix_select],
                                                    x[ix_select])[z] for z in self.metrics])).T
            res_across_folds.columns = self.metrics
            output_metric_across.loc[r, :, :] = res_across_folds

        # mean
        performance_across_mean = output_metric_across.mean(axis=0).to_pandas()

        # standard error
        performance_across_se = output_metric_across.std(axis=0).to_pandas() \
        / math.sqrt(float(len(self.results)))

        # add the iteration index to the output
        iters = [len(self.results)] * performance_across_se.shape[0]
        iters = pd.DataFrame({"iter": iters}, index=performance_across_se.index.values)

        # --- 2 --- #
        # The mean predicted value of each object is computed across all replications of the cross-validation.
        # Then the performance metrics are calculated based on the mean predicted values.
        pred_all_mean = [np.array(out_pred[x][self.config.exp_algos]) for x in range(len(out_pred))]
        pred_all_mean = pd.DataFrame(np.stack(pred_all_mean).mean(0))
        pred_all_mean.columns = self.config.exp_algos

        output_metric_append = pred_all_mean.apply(lambda x:
                            np.array([performance_results(self.df.crisis.values[ix_select], x[ix_select])[z]
                                                  for z in self.metrics])).T
        output_metric_append.columns = self.metrics

        # --- 3 --- #
        # The performance metrics are computed for each fold in each replication and are then averaged across
        # folds and replications.

        # This approach is only sensible for the cross-validation experiment. If we split training and test set by year,
        # this approach is equivalent to the first type of processing the results
        if self.config.exp_year_split is None:
            output_metric_fold = xr.DataArray(np.zeros((len(self.results), self.config.exp_nfolds,
                                                        len(self.config.exp_algos), len(self.metrics))) * float("nan"),
                                                     [('iterations', np.arange(len(self.results))),
                                                      ('folds', np.arange(self.config.exp_nfolds)),
                                                      ("algorithms", self.config.exp_algos),
                                                      ("metrics", self.metrics)])
            for r in np.arange(len(self.results)):
                for f in np.arange(self.config.exp_nfolds):
                    ix_fold = self.results[r]['ix_test'][f] # get objects of the folds
                    ix_fold = set(ix_fold).intersection(set(np.where(np.array(ix_select))[0])) # only investigate those that are on our ix_select
                    ix_fold = np.array(list(ix_fold))
                    res_in_fold = self.results[r]['predictions'].iloc[ix_fold, :].apply(lambda x:
                                    np.array([performance_results(self.df.crisis.values[ix_fold],
                                                                  x)[z] for z in self.metrics])).T
                    res_in_fold.columns = self.metrics
                    output_metric_fold.loc[r, f, :, :] = res_in_fold


            # First average across folds in single iteration, then average (and estimate SE) over the replications.
            output_metric_fold_mean_folds = output_metric_fold.mean(axis=1)
            output_metric_fold_mean = output_metric_fold_mean_folds.mean(axis=0).to_pandas()
            output_metric_fold_se = output_metric_fold_mean_folds.std(axis=0).to_pandas()/\
                math.sqrt(float(len(self.results)))

        # ix controls the subset selection of year - country pairs
        if ix is None:

            write_file(pd.concat([performance_across_mean, iters], axis=1),
                      "mean_iter_" + self.file_name, path=self.folder)
            write_file(pd.concat([performance_across_se, iters], axis=1),
                      "se_iter_" + self.file_name, path=self.folder)
            write_file(pd.concat([output_metric_append, iters], axis=1),
                      "mean_append_" + self.file_name,
                      path=self.folder)
            if self.config.exp_year_split is None:
                write_file(pd.concat([output_metric_fold_mean, iters], axis=1),
                          "mean_fold_" + self.file_name,
                          path=self.folder)
                write_file(pd.concat([output_metric_fold_se, iters], axis=1),
                          "se_fold_" + self.file_name,
                          path=self.folder)
        else:
            write_file(pd.concat([performance_across_mean, iters], axis=1),
                      "mean_iter_ix_" + self.file_name,
                      path=self.folder)
            write_file(pd.concat([performance_across_se, iters], axis=1),
                      "se_iter_ix_" + self.file_name,
                      path=self.folder)
            write_file(pd.concat([output_metric_append, iters], axis=1),
                      "mean_append_ix_" + self.file_name,
                      path=self.folder)

        print("After " + str(len(self.results)) + " iterations:")
        print(performance_across_mean.round(3))

        for algo in self.config.exp_algos:
            self._write_hyper_param(algo)

    def _write_hyper_param(self, algo):
        """ Writes hyperparameters to the hard drive."""
        
        try:

            params = self.results[0]["parameters"][algo][0].keys()
            out = {}
            for p in params:
                out[p] = [[z[p] for z in x['parameters'][algo]] for x in self.results]
                listed = [x['parameters'][algo] for x in self.results]
                listed = [item for sublist in listed for item in sublist]
                out[p] = [x[p] for x in listed]
                write_file(pd.DataFrame(out), "hyper_" + algo + "_" + self.file_name,
                  path=self.folder, shorten=False)
        except:
            pass



    def _write_shapley(self, model_name=None, **kwargs):
        """ Collects the results of the Shapley experiments from the pickle files
        and writes them into csv files.
        :param str model_name: Name of the model for which the Shapley values should be collected
        """
        self._collect_results()
        do_shap = self.config.exp_do_shapley
        if not do_shap:
            print("No Shapley values found!")
            return None
        shap_values = [np.array(x["shapley"].loc[model_name, :, :]) for x in self.results]
        nrep = len(shap_values)

        shap_values_mean = np.nanmean(np.dstack(shap_values), axis=2) # mean Shapley values across all observations
        shap_values_append = np.concatenate(shap_values, axis=0) # Shapley values appended for all replications

        pred_matrix = [self.results[x]["predictions"][model_name] for x in np.arange(len(self.results))]
        # mean predicted value across all replications:
        pred_mean = pd.Series(pd.concat((pred_matrix), axis=1).mean(axis=1), name="pred").values  
        # predicted values appended for all replications:
        pred_append = np.concatenate(pred_matrix, axis=0).astype(float)  

        # Prepare a data set with mean Shapley values
        shap_out_mean = pd.DataFrame(shap_values_mean, columns=self.feature_names)
        shap_out_mean["pred"] = pred_mean
        shap_out_mean["year"] = self.df["year"]
        shap_out_mean["iso"] = self.df["iso"]
        shap_out_mean["crisis"] = self.Y
        write_file(shap_out_mean, file_name="shapley_mean_" + model_name \
                  + "_" + self.file_name, path=self.folder, shorten=False)

        # Prepare a data set with the Shapley values of all replications
        shap_out_append = pd.DataFrame(shap_values_append, columns=self.feature_names)
        shap_out_append["pred"] = pred_append
        shap_out_append["year"] = np.tile(self.df["year"], nrep)
        shap_out_append["iso"] = np.tile(self.df["iso"], nrep)
        shap_out_append["crisis"] = np.tile(self.Y, nrep)

        write_file(shap_out_append, file_name="shapley_append_" + model_name \
                  + "_" + self.file_name, path=self.folder,
                  shorten=False)

    
    def _save_pickle(self, file_name=None, keep_models=False, keep_data=False):

        """ Saves the results of the iteration of the experiment into a pickle file
        :param Boolean keep_models: Whether the actual models should be saved in pickle files.
            This can require substantial disk space and is not recommended.
        :param Boolean keep_data:  Whether all training and test set partitions should 
            be saved in the pickle. This can also require substantial space and is not recommended.
        """

        if file_name is None:
            file_name = "pickle_" + self.file_name + "_" + str(np.random.randint(100000000))

        if not keep_models:
            for i in np.arange(len(self.results)):
                self.results[i]["models"] = None
        if not keep_data:
            for i in np.arange(len(self.results)):
                self.results[i]["data"] = None

        pickle.dump(self, open(self.folder + file_name +".p", "wb"))
