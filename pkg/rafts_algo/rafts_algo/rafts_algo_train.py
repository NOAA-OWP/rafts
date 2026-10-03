# rafts_algo_train.py
from __future__ import annotations # enables the | operator for function typehints back to python 3.7
from sklearn.model_selection import train_test_split, GridSearchCV,learning_curve
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor, AdaBoostRegressor
import xgboost as xgb
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.inspection import permutation_importance
from collections.abc import Iterable
from pathlib import Path
import pandas as pd
import numpy as np
import geopandas as gpd
import joblib
import logging
import matplotlib.pyplot as plt
import random
import scipy.stats as st
import forestci as fci
from sklearn.utils import resample
from mapie.regression import MapieRegressor
import rafts_algo.utils as utils
import rafts_algo.plots as plots
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, pairwise_distances
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.base import BaseEstimator, ClusterMixin
import gower
import gc
import traceback

# Set up basic logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Small-region safety floors: below these, GridSearchCV/MAPIE's cv or a clustering
# algorithm's requested k would raise (or silently produce a degenerate result) rather than
# train at all. The policy-level guard against this is rafts_algo.regions' min_train_gages/
# fallback (applied before training even starts); these floors are the mechanical fallback
# for whatever sample count training actually proceeds with.
_MIN_CV_FOLDS = 2
_MIN_SAMPLES_PER_CLUSTER = 2

class AlgoTrainEval:
    def __init__(self, df: pd.DataFrame, attrs: Iterable[str], algo_config: dict,
                 uncertainty: dict,
                 dir_out_alg_ds: str | Path, dataset_id: str,
                 metr: str, task_type: str = 'regression',
                 test_size: float = 0.3,rs: int = 32,
                 test_ids = None,test_id_col:str = 'featureID',
                 verbose: bool = False,
                 confidence_levels: list[int] = [95],
                 uncn_bnd_algo: bool = False, min_lim: float = None, max_lim: float = None,
                 save_all_clusters: bool = False,n_jobs: int = -1
                 ):
        """The algorithm training and evaluation class.

        :param df: The combined response variable and predictor variables DataFrame.
        :type df: pd.DataFrame
        :param attrs: The column names of attributes.
        :type attrs: Iterable[str]
        :param algo_config: The algorithm configuration as read from the algo_config yaml where each key is the algorithm that will be run. Presently allowable keys include:
            - `rf`:  :class:`sklearn.ensemble.RandomForestRegressor` algorithm. Strongly recommended to include.
            - `mlp`:  :class:`sklearn.neural_network.MLPRegressor` multilayer perceptron algorithm.
            Each algorithm key contains sub-dict keys for the parameters that may be passed to the corresponding :mod:`sklearn` algorithm. 
            If no parameters keys are passed, the :mod:`sklearn` algorithm's default arguments are used.
        :type algo_config: dict
        :param uncertainty: The uncertainty as read from the uncertainty yaml where each key is the uncertainty that will be run. Presently allowable keys include:
            - `fci_flag`:  :dict:Forestci uncertainty. Only if 'rf' algorithm is selected.
            - `bagging`:  :dict:Configuration dictionary for Bagging-based confidence interval uncertainty estimation. Works for both `rf` and `mlp` algorithms.
            - `mapie`:  :dict:Configuration dictionary for MAPIE prediction interval estimation. Works for both `rf` and `mlp` algorithms.
            Each method contains sub-dict keys for the parameters that may be passed to the corresponding method.
        :type algo_config: dict
        :param dir_out_alg_ds: Directory where algorithm's output stored.
        :type dir_out_alg_ds: str | os.PathLike
        :param dataset_id: Unique identifier/descriptor of the dataset of interest, and will be used in file writing.
        :type dataset_id: str
        :param metr: Column name in `df`. The metric or hydrologic signature identifier of interest, defaults to None.
        :type metr: str, optional
        :param test_size: Parameter for :func:`sklearn.model_selection.train_test_split`, represents proportion of dataset to include in the test split, defaults to 0.3.
        :type test_size: float, optional
        :param rs: The random seed, defaults to 32.
        :type rs: int, optional
        :param test_ids: The explicit comids of interest for testing. Defaults to None. If None, use the test_size instead for the train/test split 
        :type test_ids: Iterable or None
        :param test_id_col: The column name for comid, defaults to 'comid'
        :type test_id_col: str
        :param verbose: Should print, defaults to False.
        :type verbose: bool, optional
        :param: confidence_levels: confidence levels for ci calculation, defaults to 95
        :type confidence_levels: int, optional
        :param mapie_alpha: alpha for MAPIE, defaults to 0.05.
        :type mapie_alpha: float, optional
        :param uncn_bnd_algo: Flag to apply min/max bounds to predictions, defaults to False.
        :type uncn_bnd_algo: bool, optional
        :param min_lim: The minimum bound for the metric, defaults to None.
        :type min_lim: float, optional
        :param max_lim: The maximum bound for the metric, defaults to None.
        :type max_lim: float, optional
        :param save_all_clusters: When performing unsupervised clustering, should all cluster numbers be considered? Default False
        :type save_all_clusters: bool, optional
        :param n_jobs: Total number of jobs when performing GridSearchCV. Optional default -1 
        :type n_jobs: int
        """
        # class args
        self.df = df # NOTE: df MUST NEVER CHANGE!! df represents the original data, and is used as an index reference in the algo-train script (e.g. rafts_proc_algo_viz.py)
        self.attrs = attrs
        self.algo_config = algo_config
        self.uncertainty = uncertainty
        self.dir_out_alg_ds = dir_out_alg_ds
        self.metric = metr
        self.task_type = task_type
        self.test_size = test_size
        self.test_ids = test_ids # No guarantee these remain in the appropriate order
        self.test_id_col = test_id_col
        self.rs = rs
        self.dataset_id = dataset_id
        self.verbose = verbose
        self.confidence_levels = confidence_levels
        self.uncn_bnd_algo = uncn_bnd_algo
        self.min_lim = min_lim
        self.max_lim = max_lim
        self.save_all_clusters = save_all_clusters
        
        # train/test split
        self.X_train = pd.DataFrame()
        self.X_test = pd.DataFrame()
        self.y_train = pd.Series()
        self.y_test = pd.Series()
        
        # grid search
        self.algo_config_grid = dict()
        self.grid_search_algs = list()
        self.n_jobs = n_jobs

        # train/pred/eval metadata
        self.algs_dict = {}
        self.preds_dict = {}
        self.eval_dict = {}

        # The evaluation summary result
        self.eval_df = pd.DataFrame()

    def split_data(self):
        """Split dataframe into training and testing predictors (X) and response (y)
          variables using :func:`sklearn.model_selection.train_test_split`
        """
        # Changelog:
        #     2024-12-02 Add in the explicitly provided comid option

        if self.task_type == 'clustering':
            # Unsupervised: No 'metric' required!
            self.df_non_na = self.df.dropna(subset=self.attrs)
            X = self.df_non_na[self.attrs]
            if self.verbose:
                logging.info(f"      Performing clustering split as {round(1-self.test_size,2)}/{self.test_size}")
            self.X_train, self.X_test = train_test_split(X, test_size=self.test_size, random_state=self.rs)
            self.y_train, self.y_test = None, None
        else:
            # Check for NA values first
            self.df_non_na = self.df[self.attrs + [self.metric]].dropna()
            if self.df_non_na.shape[0] < self.df.shape[0]:
                logging.warning(f"\
                    \n   !!!!!!!!!!!!!!!!!!!\
                    \n   NA VALUES FOUND IN INPUT DATASET!! \
                    \n   DROPPING {self.df.shape[0] - self.df_non_na.shape[0]} ROWS OF DATA. \
                    \n   !!!!!!!!!!!!!!!!!!!")
                    
            if self.test_ids is not None:
                # The Truth is in the indices: e.g. `self.df` shares the same indicise as `self.test_ids`` 
                # Use the manually provided comids for testing, then the remaining data for training
                logging.info("Using the custom test comids, and letting all remaining comids be used for training.")
                df_sub_test = self.df.loc[self.test_ids.index]#self.df[self.df[self.test_id_col].isin(self.test_ids)].dropna(subset=self.attrs + [self.metric])
                df_sub_train = self.df.loc[~self.df.index.isin(df_sub_test.index)]#self.df[~self.df[self.test_id_col].isin(self.test_ids)].dropna(subset=self.attrs + [self.metric])
                # Assign class objects
                self.y_test = df_sub_test[self.metric]
                self.y_train = df_sub_train[self.metric]
                self.X_test = df_sub_test[self.attrs]
                self.X_train = df_sub_train[self.attrs]
            else: # The standard train_test_split (Caution when processing multiple datasets, if total dims differ, then basin splits may differ)
                if self.verbose:
                    logging.info(f"      Performing train/test split as {round(1-self.test_size,2)}/{self.test_size}")
                X = self.df_non_na[self.attrs]
                y = self.df_non_na[self.metric]
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X,y, test_size=self.test_size, random_state=self.rs)

    def all_X_all_y(self):
        """ Combine the train/test splits into a single dataframe/array. 
            This method may be called after calling AlgoTrainEval.split_data() 
            to concatenate the training and testing datasets into single DataFrames
            for features (X) and response variable (y). 

        :return: A tuple containing concatenated df for features (X) and response variable (y). 
        :rtype: tuple(pandas.DataFrame, pandas.Series) 
        """
        # Combine the train/test splits into a single dataframe/array
        # This may be called after calling AlgoTrainEval.split_data()
        X = pd.concat([self.X_train,  self.X_test])
        # Check if y values exist before concatenating
        if self.y_test is not None and self.y_train is not None:
            y = pd.concat([self.y_test, self.y_train])
        else: # e.g. for unsupervised clustering
            y = None
        return X, y

    def convert_to_list(self,d:dict) ->dict:
        """Runcheck: In situations where self.algo_config_grid is used, all objects must be iterables 

        :param d: A dict containing sub-dicts with key-value pairs
        :type d: dict
        :return: The dict where any non-iterable values have been converted into a list
        :rtype: dict
        """
        for key, value in d.items():
            if isinstance(value, dict):
                self.convert_to_list(value)
            elif not isinstance(value, (list, tuple)):
                d[key] = [value]
        return(d)

    def list_to_dict(self, config_ls):
        """Convert to dict if a config object is inconveniently
          formatted as a list of multiple dicts

        :param config_ls: possibly a list of objects
        :type config_ls: list
        :return: dict of objects
        :rtype: dict
        """
        # 
        if isinstance(config_ls,list):
            config_dict = {}
            for d in config_ls:
                config_dict.update(d)
        else:
            config_dict = config_ls
        return config_dict
    
    def select_algs_grid_search(self):
        """Determines which algorithms' params involve hyperparameter tuning
            based on if multiple parameters designated for consideration
        """
        ls_move_to_srch_cfig = list()
        for k, alg_ls in self.algo_config.items():
            sub_dict = {k: alg_dict[k] for alg_dict in alg_ls for k in alg_dict.keys()}
            totl_opts_per_param = list()
            for kk, v in sub_dict.items():
                if isinstance(v,Iterable) and len(v)>1:
                    totl_opts_per_param.append(len(v))
                else:
                    totl_opts_per_param.append(1)
            if any([x > 1 for x in totl_opts_per_param]):
                if self.verbose:
                    logging.info(f"Performing grid search CV for {k}")
                self.grid_search_algs.append(k)
                ls_move_to_srch_cfig.append(k)

        # Move hyperparams from basic algo params into grid search params
        for k in ls_move_to_srch_cfig:
            self.algo_config_grid[k] = self.algo_config.pop(k)
        
        # Convert lists inside algo_config_grid['algo_name_here'] to a dict:
        dict_acg = {}
        for key, val in self.algo_config_grid.items():
            dict_acg[key] = self.list_to_dict(val)
        self.algo_config_grid = dict_acg

        # Convert lists inside algo_config['algo_name_here'] to a dict:    
        dict_ac = {}
        for key, val in self.algo_config.items():
            dict_ac[key] = self.list_to_dict(val)
        self.algo_config = dict_ac

        if self.algo_config_grid: # If there are non iterable values, convert them to lists to aid the algo training
            # e.g. {'activation':'relu'} becomes {'activation':['relu']}
            self.algo_config_grid  = self.convert_to_list(self.algo_config_grid)

    def _capped_cv(self, requested_cv: int, n_samples: int) -> int:
        """Cap a cross-validation fold count to what the current training set can support.

        :param requested_cv: The configured (or default) number of CV folds.
        :type requested_cv: int
        :param n_samples: The number of samples available to fit on (``len(self.X_train)``).
        :type n_samples: int
        :return: `requested_cv`, capped to `n_samples` and floored at `_MIN_CV_FOLDS`.
        :rtype: int
        """
        capped = max(_MIN_CV_FOLDS, min(requested_cv, n_samples))
        if capped != requested_cv:
            logging.warning(f"      Capping cv={requested_cv} to {capped} "
                             f"(only {n_samples} training samples available).")
        return capped

    def _capped_n_clusters(self, requested_k: int, n_samples: int) -> int:
        """Cap a cluster count to what the current training set can support.

        :param requested_k: The configured (or default) number of clusters.
        :type requested_k: int
        :param n_samples: The number of samples available to fit on (``len(self.X_train)``).
        :type n_samples: int
        :return: `requested_k`, capped to ``max(1, n_samples // _MIN_SAMPLES_PER_CLUSTER)``.
        :rtype: int
        """
        max_k = max(1, n_samples // _MIN_SAMPLES_PER_CLUSTER)
        capped = min(requested_k, max_k)
        if capped != requested_k:
            logging.warning(f"      Capping n_clusters={requested_k} to {capped} "
                             f"(only {n_samples} training samples available).")
        return capped

    def calculate_forestci_uncertainty(self, forest, X_train, X_test):
        """
        Calculate uncertainty using forestci for a Random Forest model.
    
        :param forest: Trained Random Forest model.
        :type forest: RandomForestRegressor
        :param X_train: Training data.
        :type X_train: ndarray
        :param X_test: Test data.
        :type X_test: ndarray
        :return: Confidence intervals for each prediction.
        :rtype: ndarray
        """
        # Compute standard deviation of prediction errors

        confidence_levels = self.confidence_levels
        ci_std = np.sqrt(fci.random_forest_error(
            forest=forest,
            X_train_shape=X_train.shape,
            X_test=np.array(X_test),
            inbag=None, 
            calibrate=True, 
            memory_constrained=False, 
            memory_limit=None, 
            y_output=0  
        ))        
    
        # Compute confidence intervals for each level
        ci_dict = {}
        for alpha in confidence_levels:
            z_value = st.norm.ppf(1 - (1 - alpha/100) / 2)  # Get z-score for two-tailed CI
            ci_lower = -z_value * ci_std
            ci_upper = z_value * ci_std
            ci_dict[f'ci_{int(alpha)}'] = {
                "lower_bound": ci_lower,
                "upper_bound": ci_upper
            }
    
        return ci_dict
    def calculate_bagging_ci(self, algo_str,best_algo):
        """
        Generalized function to calculate Bagging confidence intervals for any model.
        """
        algo_cfg = self.algo_config.get(algo_str, self.algo_config_grid.get(algo_str))
        if algo_cfg is None:
            logging.error(f"Algorithm {algo_str} not found in configurations.")
            raise KeyError(f"Algorithm {algo_str} not found in configurations.")

        n_algos = next(d['n_algos'] for d in self.uncertainty.get('bagging', []))
        predictions = []
        
        # Safely extract the base ML model if it is wrapped inside a Pipeline
        if isinstance(best_algo, Pipeline):
            # The base estimator is universally the final step in a standard scikit-learn pipeline
            algo_step = best_algo.steps[-1][1]
        else:
            algo_step = best_algo  # Direct model
    
        base_algo = algo_step  # Now we have the extracted model
        
        random.seed(self.rs)
        random_states = [random.randint(1, 10000) for _ in range(n_algos)]
    
        for rand_state in random_states:
            # Resample data with a fixed random state for reproducibility
            X_train_resampled, y_train_resampled = resample(
                self.X_train, self.y_train, random_state=rand_state
            )
    
            # Create a new model with the same parameters but a different random_state
            algo_tmp = type(base_algo)(**{**base_algo.get_params(), "random_state": rand_state})
    
            algo_tmp.fit(X_train_resampled, y_train_resampled)
            predictions.append(algo_tmp.predict(self.X_test))        

        predictions = np.array(predictions)
        mean_pred = predictions.mean(axis=0)
        std_pred = predictions.std(axis=0)
        confidence_levels = self.confidence_levels 
        confidence_intervals = {}

        for cl in confidence_levels:
            lower_bound, upper_bound = np.percentile(predictions, [(100 - cl) / 2, 100 - (100 - cl) / 2], axis=0)
            confidence_intervals[f"confidence_level_{cl}"] = {
                "lower_bound": lower_bound,
                "upper_bound": upper_bound
            }  

        if 'Uncertainty' not in self.algs_dict[algo_str]:
            self.algs_dict[algo_str]['Uncertainty'] = {}
    
        self.algs_dict[algo_str]['Uncertainty']['bagging_mean_pred'] = mean_pred
        self.algs_dict[algo_str]['Uncertainty']['bagging_std_pred'] = std_pred
        self.algs_dict[algo_str]['Uncertainty']['bagging_confidence_intervals'] = confidence_intervals

    
    def calculate_mapie(self):
        """Generalized function to calculate prediction uncertainty using MAPIE."""
        mapie_method = next((d['method'] for d in self.uncertainty.get('mapie', []) if 'method' in d), None)
        mapie_cv = next((d['cv'] for d in self.uncertainty.get('mapie', []) if 'cv' in d), None)
        if mapie_cv is not None:
            mapie_cv = self._capped_cv(mapie_cv, len(self.X_train))
        mapie_agg_function = next((d['agg_function'] for d in self.uncertainty.get('mapie', []) if 'agg_function' in d), None)
        try:
            for algo_str, algo_data in self.algs_dict.items():
                algo = algo_data['algo']
                mapie = MapieRegressor(
                    algo,
                    method=mapie_method,
                    cv=mapie_cv,
                    agg_function=mapie_agg_function
                )
                mapie.fit(self.X_train, self.y_train)
                self.algs_dict[algo_str]['mapie'] = mapie
        except ValueError as e:
            logging.error(f"Invalid MAPIE method '{mapie_method}'. Please choose either 'plus' or 'minmax'.")
            raise ValueError(f"Invalid MAPIE method '{mapie_method}'. Please choose either 'plus' or 'minmax'.") from e
            
    def train_algos(self):
        """Train algorithms based on what has been defined in the algo config file

        .. note::
            Algorithm options include the following:
                - `rf` for :class:`sklearn.ensemble.RandomForestRegressor`
                - `mlp` for :class:`sklearn.neural_network.MLPRegressor`
        """
        # Train algorithms based on config
        if 'rf' in self.algo_config:  # RANDOM FOREST
            if self.verbose:
                logging.info(f"      Performing Random Forest Training")
            
            rf = RandomForestRegressor(n_estimators=self.algo_config['rf'].get('n_estimators',300),
                                       max_depth = self.algo_config['rf'].get('max_depth', None),
                                       min_samples_split=self.algo_config['rf'].get('min_samples_split',2),
                                       min_samples_leaf=self.algo_config['rf'].get('min_samples_leaf',1),
                                       oob_score=True,
                                       random_state=self.rs,
                                       )
            pipe_rf = make_pipeline(rf)                       
            pipe_rf.fit(self.X_train, self.y_train)
            
            # --- Compare predictions with confidence intervals ---
            self.algs_dict['rf'] = {'algo': rf,
                                    'pipeline': pipe_rf,
                                    'type': 'random forest regressor',
                                    'metric': self.metric,
                                    'Uncertainty': {}
                }
        if 'mlp' in self.algo_config:  # MULTI-LAYER PERCEPTRON
            if self.verbose:
                logging.info(f"      Performing Multilayer Perceptron Training")
            mlpcfg = self.algo_config['mlp']
            mlp = MLPRegressor(random_state=self.rs,
                               hidden_layer_sizes=mlpcfg.get('hidden_layer_sizes', (100,)),
                               activation=mlpcfg.get('activation', 'relu'),
                               solver=mlpcfg.get('solver', 'lbfgs'),
                               alpha=mlpcfg.get('alpha', 0.001),
                               batch_size=mlpcfg.get('batch_size', 'auto'),
                               learning_rate=mlpcfg.get('learning_rate', 'constant'),
                               power_t=mlpcfg.get('power_t', 0.5),
                               max_iter=mlpcfg.get('max_iter', 200))
            pipe_mlp = make_pipeline(StandardScaler(),mlp)
            pipe_mlp.fit(self.X_train, self.y_train)

            self.algs_dict['mlp'] = {'algo': mlp,
                                     'pipeline': pipe_mlp,
                                     'type': 'multi-layer perceptron regressor',
                                     'metric': self.metric,
                                     'Uncertainty': {}
                                     }
        if 'hgbr' in self.algo_config:  # HIST GRADIENT BOOSTING
            if self.verbose: logging.info(f"      Performing HistGradientBoostingRegressor Training")
            # Unpack defaults first, allowing config to override safely
            hgbr = HistGradientBoostingRegressor(**{'random_state': self.rs, **self.algo_config['hgbr']})
            pipe_hgbr = make_pipeline(StandardScaler(), hgbr)
            pipe_hgbr.fit(self.X_train, self.y_train)
            self.algs_dict['hgbr'] = {'algo': hgbr, 'pipeline': pipe_hgbr, 'type': 'hist gradient boosting regressor', 'metric': self.metric, 'Uncertainty': {}}

        if 'gbr' in self.algo_config:  # GRADIENT BOOSTING
            if self.verbose: logging.info(f"      Performing GradientBoostingRegressor Training")
            gbr = GradientBoostingRegressor(**{'random_state': self.rs, **self.algo_config['gbr']})
            pipe_gbr = make_pipeline(StandardScaler(), gbr)
            pipe_gbr.fit(self.X_train, self.y_train)
            self.algs_dict['gbr'] = {'algo': gbr, 'pipeline': pipe_gbr, 'type': 'gradient boosting regressor', 'metric': self.metric, 'Uncertainty': {}}

        if 'adaboost' in self.algo_config:  # ADABOOST
            if self.verbose: logging.info(f"      Performing AdaBoostRegressor Training")
            ada = AdaBoostRegressor(**{'random_state': self.rs, **self.algo_config['adaboost']})
            pipe_ada = make_pipeline(StandardScaler(), ada)
            pipe_ada.fit(self.X_train, self.y_train)
            self.algs_dict['adaboost'] = {'algo': ada, 'pipeline': pipe_ada, 'type': 'adaboost regressor', 'metric': self.metric, 'Uncertainty': {}}

        if 'xgb' in self.algo_config:  # XGBOOST
            if self.verbose: logging.info(f"      Performing XGBRegressor Training")
            # Safely merge default arguments with user-provided config arguments
            xgb_args = {'random_state': self.rs, 'n_jobs': self.n_jobs}
            xgb_args.update(self.algo_config['xgb'])
            xgb_model = xgb.XGBRegressor(**xgb_args)
            pipe_xgb = make_pipeline(StandardScaler(), xgb_model)
            pipe_xgb.fit(self.X_train, self.y_train)
            self.algs_dict['xgb'] = {'algo': xgb_model, 'pipeline': pipe_xgb, 'type': 'xgboost regressor', 'metric': self.metric, 'Uncertainty': {}}
        # ---------------- UNSUPERVISED ALGORITHMS ------------- #
        if 'kmeans' in self.algo_config:  # K-MEANS CLUSTERING
            if self.verbose: logging.info(f"      Performing KMeans Clustering")
            
            n_clust = self.algo_config['kmeans'].get('n_clusters', 5)
            if isinstance(n_clust, list) and len(n_clust) == 1:
                n_clust = int(n_clust[0])
            n_clust = self._capped_n_clusters(n_clust, len(self.X_train))

            kmeans = KMeans(n_clusters=n_clust,
                            random_state=self.rs)
            pipe_kmeans = make_pipeline(StandardScaler(), kmeans)
            
            # FIT ON X_TRAIN ONLY
            pipe_kmeans.fit(self.X_train) 
            
            self.algs_dict[f'kmeans_k{n_clust}'] = {'algo': kmeans, 'pipeline': pipe_kmeans, 
                                        'type': 'clustering', 'metric': self.metric, 
                                        'Uncertainty': {}}
            
        if 'gower_agglomerative' in self.algo_config: 
            if self.verbose: logging.info("      Performing Gower Agglomerative Clustering")
            from sklearn.cluster import AgglomerativeClustering
            
            n_clust = self.algo_config['gower_agglomerative'].get('n_clusters', 5)
            if isinstance(n_clust, list) and len(n_clust) == 1:
                n_clust = int(n_clust[0])
            n_clust = self._capped_n_clusters(n_clust, len(self.X_train))

            # 1. Use Scikit-Learn's native Agglomerative Clustering
            # 'average' linkage is highly stable for mixed continuous/categorical hydrologic data
            base_algo = AgglomerativeClustering(n_clusters=n_clust, metric='precomputed', linkage='average')
            
            # 2. Pass it into our Universal Wrapper to calculate Gower's and enable .predict()
            universal_clusterer = UniversalDistanceClusterer(estimator=base_algo, metric='gower')
            
            # 3. Create the pipeline and fit
            pipe_gower = make_pipeline(universal_clusterer)
            pipe_gower.fit(self.X_train) 
            
            self.algs_dict[f'gower_agglomerative_k{n_clust}'] = {
                'algo': universal_clusterer, 'pipeline': pipe_gower, 
                'type': 'clustering', 'metric': self.metric, 'Uncertainty': {}
            }    
        # if 'gower_kmedoids' in self.algo_config: 
        #     from sklearn_extra.cluster import KMedoids
        #     n_clust = int(self.algo_config['gower_kmedoids'].get('n_clusters', [5])[0])

        #     # Pass KMedoids into our Universal Wrapper
        #     base_algo = KMedoids(n_clusters=n_clust, metric='precomputed', random_state=self.rs)
        #     universal_clusterer = UniversalDistanceClusterer(estimator=base_algo, metric='gower')
            
        #     # Now it acts exactly like a normal scikit-learn pipeline!
        #     pipe_gower = make_pipeline(universal_clusterer)
        #     pipe_gower.fit(self.X_train) 
            
        #     self.algs_dict['gower_kmedoids'] = {'algo': universal_clusterer, 'pipeline': pipe_gower, 
        #                                 'type': 'clustering', 'metric': self.metric, 'Uncertainty': {}}

    def train_algos_grid_search(self):
        """Train algorithms using GridSearchCV based on the algo config file.
        
        .. note::
            Algorithm options include the following:
                - `rf` for :class:`sklearn.ensemble.RandomForestRegressor`
                - `mlp` for :class:`sklearn.neural_network.MLPRegressor`
        """
        # --- SUPERVISED REGRESSION ALGORITHMS ---
        if 'rf' in self.algo_config_grid:  # RANDOM FOREST
            if self.verbose:
                logging.info(f"      Performing Random Forest Training with Grid Search")
            rf = RandomForestRegressor(oob_score=True, random_state=self.rs)
            # TODO move into main Param dict
            param_grid_rf = {
                'randomforestregressor__n_estimators': self.algo_config_grid['rf'].get('n_estimators', [100, 200, 300]),
                'randomforestregressor__max_depth': self.algo_config_grid['rf'].get('max_depth', [None,10, 20, 30]), 
                'randomforestregressor__min_samples_leaf': self.algo_config_grid['rf'].get('min_samples_leaf', [1, 2, 4]),
                'randomforestregressor__min_samples_split': self.algo_config_grid['rf'].get('min_samples_split', [2, 5, 10])
            }
            pipe_rf = make_pipeline(rf)
            grid_rf = GridSearchCV(pipe_rf, param_grid_rf, cv=self._capped_cv(5, len(self.X_train)), scoring='neg_mean_absolute_error', n_jobs=self.n_jobs)
            
            grid_rf.fit(self.X_train, self.y_train)

            self.algs_dict['rf'] = {'algo': grid_rf.best_estimator_.named_steps['randomforestregressor'],
                                    'pipeline': grid_rf.best_estimator_,
                                    'gridsearchcv': grid_rf,
                                    'type': 'random forest regressor',
                                    'metric': self.metric,
                                    'Uncertainty': {}
                                    }
        
        if 'mlp' in self.algo_config_grid:  # MULTI-LAYER PERCEPTRON
            if self.verbose:
                logging.info(f"      Performing Multilayer Perceptron Training with Grid Search")
            mlpcfg = self.algo_config_grid['mlp']
            mlp = MLPRegressor(random_state=self.rs)
            param_grid_mlp = {
                'mlpregressor__hidden_layer_sizes': mlpcfg.get('hidden_layer_sizes', [(100,), (50, 50)]),
                'mlpregressor__activation': mlpcfg.get('activation', ['relu', 'tanh']),
                'mlpregressor__solver': mlpcfg.get('solver', ['lbfgs', 'adam']),
                'mlpregressor__alpha': mlpcfg.get('alpha', [0.001, 0.01]),
                'mlpregressor__learning_rate': mlpcfg.get('learning_rate', ['constant', 'adaptive']),
                'mlpregressor__max_iter': mlpcfg.get('max_iter', [200, 300])
            }
            pipe_mlp = make_pipeline(StandardScaler(), mlp)
            grid_mlp = GridSearchCV(pipe_mlp, param_grid_mlp, cv=self._capped_cv(5, len(self.X_train)), scoring='neg_mean_absolute_error', n_jobs=self.n_jobs)
            grid_mlp.fit(self.X_train, self.y_train)
            self.algs_dict['mlp'] = {'algo': grid_mlp.best_estimator_,
                                    'pipeline': grid_mlp,
                                    'type': 'multi-layer perceptron regressor',
                                    'metric': self.metric,
                                     'Uncertainty': {}
                                     }
            
        if 'hgbr' in self.algo_config_grid:  # HIST GRADIENT BOOSTING
            if self.verbose: logging.info(f"      Performing HistGradientBoostingRegressor Training with Grid Search")
            hgbr = HistGradientBoostingRegressor(random_state=self.rs)
            param_grid_hgbr = {f'histgradientboostingregressor__{k}': v for k, v in self.algo_config_grid['hgbr'].items()}
            pipe_hgbr = make_pipeline(StandardScaler(), hgbr)
            grid_hgbr = GridSearchCV(pipe_hgbr, param_grid_hgbr, cv=self._capped_cv(5, len(self.X_train)), scoring='neg_mean_absolute_error', n_jobs=self.n_jobs)
            grid_hgbr.fit(self.X_train, self.y_train)
            self.algs_dict['hgbr'] = {'algo': grid_hgbr.best_estimator_.named_steps['histgradientboostingregressor'], 'pipeline': grid_hgbr, 'gridsearchcv': grid_hgbr, 'type': 'hist gradient boosting regressor', 'metric': self.metric, 'Uncertainty': {}}

        if 'gbr' in self.algo_config_grid:  # GRADIENT BOOSTING
            if self.verbose: logging.info(f"      Performing GradientBoostingRegressor Training with Grid Search")
            gbr = GradientBoostingRegressor(random_state=self.rs)
            param_grid_gbr = {f'gradientboostingregressor__{k}': v for k, v in self.algo_config_grid['gbr'].items()}
            pipe_gbr = make_pipeline(StandardScaler(), gbr)
            grid_gbr = GridSearchCV(pipe_gbr, param_grid_gbr, cv=self._capped_cv(5, len(self.X_train)), scoring='neg_mean_absolute_error', n_jobs=self.n_jobs)
            grid_gbr.fit(self.X_train, self.y_train)
            self.algs_dict['gbr'] = {'algo': grid_gbr.best_estimator_.named_steps['gradientboostingregressor'], 'pipeline': grid_gbr, 'gridsearchcv': grid_gbr, 'type': 'gradient boosting regressor', 'metric': self.metric, 'Uncertainty': {}}

        if 'adaboost' in self.algo_config_grid:  # ADABOOST
            if self.verbose: logging.info(f"      Performing AdaBoostRegressor Training with Grid Search")
            ada = AdaBoostRegressor(random_state=self.rs)
            param_grid_ada = {f'adaboostregressor__{k}': v for k, v in self.algo_config_grid['adaboost'].items()}
            pipe_ada = make_pipeline(StandardScaler(), ada)
            grid_ada = GridSearchCV(pipe_ada, param_grid_ada, cv=self._capped_cv(5, len(self.X_train)), scoring='neg_mean_absolute_error', n_jobs=self.n_jobs)
            grid_ada.fit(self.X_train, self.y_train)
            self.algs_dict['adaboost'] = {'algo': grid_ada.best_estimator_.named_steps['adaboostregressor'], 'pipeline': grid_ada, 'gridsearchcv': grid_ada, 'type': 'adaboost regressor', 'metric': self.metric, 'Uncertainty': {}}

        if 'xgb' in self.algo_config_grid:  # XGBOOST
            if self.verbose: logging.info(f"      Performing XGBRegressor Training with Grid Search")
            xgb_model = xgb.XGBRegressor(random_state=self.rs, n_jobs=self.n_jobs)
            param_grid_xgb = {f'xgbregressor__{k}': v for k, v in self.algo_config_grid['xgb'].items()}
            pipe_xgb = make_pipeline(StandardScaler(), xgb_model)
            grid_xgb = GridSearchCV(pipe_xgb, param_grid_xgb, cv=self._capped_cv(5, len(self.X_train)), scoring='neg_mean_absolute_error', n_jobs=self.n_jobs)
            grid_xgb.fit(self.X_train, self.y_train)
            self.algs_dict['xgb'] = {'algo': grid_xgb.best_estimator_.named_steps['xgbregressor'], 'pipeline': grid_xgb, 'gridsearchcv': grid_xgb, 'type': 'xgboost regressor', 'metric': self.metric, 'Uncertainty': {}}
        # --- CLUSTERING ALGORITHMS ---
        cluster_algs_to_run = [alg for alg in ['kmeans', 'gower_agglomerative'] if alg in self.algo_config_grid]

        for alg_name in cluster_algs_to_run:
            if self.verbose:
                logging.info(f"      Performing {alg_name} Clustering with Grid Search")
            
            cluster_sizes = self.algo_config_grid[alg_name].get('n_clusters', [3, 5, 8])
            max_k = max(1, len(self.X_train) // _MIN_SAMPLES_PER_CLUSTER)
            filtered_sizes = [k for k in cluster_sizes if k <= max_k] or [max_k]
            if filtered_sizes != list(cluster_sizes):
                logging.warning(f"      Capping {alg_name} candidate cluster sizes {list(cluster_sizes)} "
                                 f"to {filtered_sizes} (only {len(self.X_train)} training samples available).")
            cluster_sizes = filtered_sizes

            best_score = -2.0
            best_model = None
            best_pipe = None
            
            for k in cluster_sizes:
                # 1. Initialize the correct algorithm
                if alg_name == 'kmeans':
                    base_algo = KMeans(n_clusters=k, random_state=self.rs)
                    pipe = make_pipeline(StandardScaler(), base_algo)
                elif alg_name == 'gower_agglomerative':
                    from sklearn.cluster import AgglomerativeClustering
                    agglom = AgglomerativeClustering(n_clusters=k, metric='precomputed', linkage='average')
                    universal_clusterer = UniversalDistanceClusterer(estimator=agglom, metric='gower')
                    pipe = make_pipeline(universal_clusterer)
                
                # 2. Fit the pipeline
                pipe.fit(self.X_train)
                labels = pipe.predict(self.X_train)
                
                # 3. Calculate Score
                if len(np.unique(labels)) > 1:
                    score = silhouette_score(self.X_train, labels)
                else:
                    score = -1.0 # Heavily penalize if it collapses
                
                # 4. Save logic based on the user's flag
                if self.save_all_clusters:
                    # Save EVERY iteration as a distinct algorithm! (e.g., 'kmeans_k5')
                    iter_name = f"{alg_name}_k{k}"
                    self.algs_dict[iter_name] = {
                        'algo': base_algo if alg_name == 'kmeans' else universal_clusterer,
                        'pipeline': pipe,
                        'type': 'clustering',
                        'metric': self.metric,
                        'Uncertainty': {}
                    }
                    if self.verbose:
                        logging.info(f"      Saved {iter_name} (Silhouette: {score:.3f})")
                else:
                    # Only track the best performing iteration
                    if score > best_score:
                        best_score = score
                        best_model = base_algo if alg_name == 'kmeans' else universal_clusterer
                        best_pipe = pipe
            
            # 5. If not saving all, write the absolute best model to the dictionary
            if not self.save_all_clusters:
                if hasattr(best_model, 'n_clusters'):
                    opt_k = best_model.n_clusters
                else:
                    opt_k = best_model.estimator.n_clusters
                if self.verbose:
                    logging.info(f"      Optimal {alg_name} clusters chosen: {opt_k} (Silhouette: {best_score:.3f})")
                winning_name = f"{alg_name}_k{opt_k}"
                self.algs_dict[winning_name] = {
                    'algo': best_model,
                    'pipeline': best_pipe,
                    'type': 'clustering',
                    'metric': self.metric,
                    'Uncertainty': {}
                }
        # --------------------------------
            


    def predict_algos(self) -> dict:
        """ Make predictions with trained algorithms   

        :return: Evaluation results, with the following keys:
            - `y_pred`: The predicted values vector
            - `type`: The type of algorithm used for prediction (e.g. `"random forest"`)
            - `metric`: The formulation evaluation metric or hydrologic signature represented by `y_pred`
        :rtype: dict
        """
          
        for k, v in self.algs_dict.items():
            algo = v['algo']
            pipe = v['pipeline']
            type_algo = v['type']
            if self.verbose:
                logging.info(f"      Generating predictions for {type_algo} algorithm.")   
            
            # Get the feature IDs corresponding to the test set rows from the original dataframe
            feature_ids = self.df.loc[self.X_test.index, self.test_id_col]

            y_pred = pipe.predict(self.X_test)
            # --- Unconditionally warn if any predictions fall out of the physical range. ---
            if self.task_type != 'clustering':
                utils._warn_if_out_of_bounds(
                    predictions=y_pred,
                    feature_ids=feature_ids,
                    min_lim=self.min_lim,
                    max_lim=self.max_lim,
                    resp_var=self.metric,
                    correction_is_active=self.uncn_bnd_algo,
                    prediction_type="values"
                )
                # Conditionally apply the correction based on flags.
                if self.uncn_bnd_algo:
                    y_pred = utils.clip_predictions(y_pred, self.min_lim, self.max_lim)

                if 'mapie' in v:
                    mapie_alpha = next((d['alpha'] for d in self.uncertainty.get('mapie', []) if 'alpha' in d), None)
                    y_test_pred, y_test_pis = v['mapie'].predict(self.X_test, alpha=mapie_alpha)
                    
                    # Apply same warn-then-clip logic for prediction intervals.
                    utils._warn_if_out_of_bounds(
                        predictions=y_test_pis,
                        feature_ids=feature_ids,
                        min_lim=self.min_lim,
                        max_lim=self.max_lim,
                        resp_var=self.metric,
                        correction_is_active=self.uncn_bnd_algo,
                        prediction_type="intervals"
                    )
                    if self.uncn_bnd_algo:
                        y_test_pis = utils.clip_pis(y_test_pis, self.min_lim, self.max_lim)
                        
                    # Rename rows
                    row_labels = ['lower_limit', 'upper_limit']
                    
                    # Rename columns based on mapie_alpha values
                    col_labels = [f'alpha_{alpha:.2f}' for alpha in mapie_alpha]  
                    
                    # Convert to DataFrame
                    y_pis_list = [pd.DataFrame(y_test_pis[i], index=row_labels, columns=col_labels) for i in range(y_test_pis.shape[0])]
                    
                    self.preds_dict[k] = {'y_pred': y_pred,
                                        'y_pis': y_pis_list,
                                        'type': v['type'],
                                        'metric': v['metric']}
                else:
                    self.preds_dict[k] = {'y_pred': y_pred,
                                    'type': v['type'],
                                    'metric': v['metric']}
            else:
                    self.preds_dict[k] = {'y_pred': y_pred,
                                    'type': v['type'],
                                    'metric': v['metric']}
                                
        return self.preds_dict

    def evaluate_algos(self) -> dict:
        """ Evaluate the predictions

        :return: Evaluation of algorithm performance from the test dataset. The dict keys include the following:
            - `type`: the type of algorithm, e.g. "random forest"
            - `metric`: the formulation metric, or hydrologic signature being predicted
            - `mse`: algorithm's mean squared error from test data
            - `r2`: algorithm's r-squared from test data
            - 'MinResid': minimum residual. min(sim - obs)
            - 'MaxResid': minimum residual. max(sim - obs)
            - 'AvgResid': average residual. mean(sim - obs)
            - 'SDResid': standard deviation of residuals. std(sim - obs)
            - 'RMSE': square root of algorithm's mean squared error. sqrt(mean((sim - obs)^2))
            - 'NRMSE':  normalized RMSE, RMSE divided by obs range. (max(obs)-min(obs)); NaN if range==0
        :rtype: dict
        """
       
        if self.verbose:
            logging.info(f"      Evaluating predictions.")   

        for k, v in self.preds_dict.items():
            y_pred = v['y_pred']
            if self.task_type == 'clustering':
                # Evaluate cluster density only if at least 2 clusters exist in test set
                n_clusters_predicted = len(np.unique(y_pred))
                if n_clusters_predicted > 1:
                    sil_score = silhouette_score(self.X_test, y_pred)
                    db_score = davies_bouldin_score(self.X_test, y_pred)
                else:
                    sil_score = np.nan
                    db_score = np.nan
                    logging.warning(f"Only 1 cluster predicted in the test set. Clustering metrics cannot be calculated.")
                # Evaluate cluster density and separation
                self.eval_dict[k] = {
                    'type': v['type'],
                    'metric': v['metric'],
                    'silhouette_score': sil_score,
                    'davies_bouldin_score': db_score
                }
            else:
                resid = y_pred - self.y_test
                rmse = float(np.sqrt(np.mean(resid**2)))
                obs_range = float(np.max(self.y_test) - np.min(self.y_test))
                rmse_obs = float(rmse / obs_range) if obs_range > 0.0 else np.nan
                # Calculate Spearman's correlation coefficient
                spearman_corr, _ = st.spearmanr(self.y_test, y_pred)

                self.eval_dict[k] = {'type': v['type'],
                                'metric': v['metric'],
                                'mse': mean_squared_error(self.y_test, y_pred),
                                'r2': r2_score(self.y_test, y_pred),
                                'spearman': float(spearman_corr),
                                'MinResid': float(np.min(resid)),
                                'MaxResid': float(np.max(resid)),
                                'AvgResid': float(np.mean(resid)),
                                'SDResid': float(np.std(resid)),
                                'RMSE': rmse,
                                'NRMSE': rmse_obs}

        return self.eval_dict

    def save_algos(self):
        """ Write pipeline to file & record save path in `algs_dict['file_pipe']`

        """
        
        for algo in self.algs_dict.keys():
            if self.verbose:
                logging.info(f"      Saving {algo} pipeline for {self.metric} to file")

            path_algo = utils.std_algo_path(self.dir_out_alg_ds, algo, self.metric, self.dataset_id)
            
            # write trained algorithm
            joblib.dump(self.algs_dict[algo]['pipeline'], path_algo)
            
            # Save pipeline and metadata in a dictionary
            pipeline_data = {
                'pipeline': self.algs_dict[algo]['pipeline'],  # The trained model pipeline
                'X_train_shape': self.X_train.shape,  # Store the shape of X_train
                'Uncertainty': self.algs_dict[algo]['Uncertainty']
            }

            # If mapie_alpha exists and is not empty, save mapie
            if any('alpha' in d and d['alpha'] for d in self.uncertainty.get('mapie', [])):  #getattr(self, 'mapie_alpha', None):
                pipeline_data['mapie'] = self.algs_dict[algo].get('mapie', None)

            joblib.dump(pipeline_data, path_algo)  # Save pipeline + X_train shape
            
            self.algs_dict[algo]['file_pipe'] = str(path_algo.name)
   
    def org_metadata_alg(self):
        """Must be called after running AlgoTrainEval.save_algos(). Records saved location of trained algorithm

        """

        self.eval_df = pd.DataFrame(self.eval_dict).transpose().rename_axis(index='algorithm')

        self.eval_df['dataset'] = self.dataset_id

        # Assign the locations where algorithms were saved
        self.eval_df['file_pipe'] = [self.algs_dict[alg]['file_pipe'] for alg in self.algs_dict.keys()] 
        self.eval_df['algo'] = self.eval_df.index
        self.eval_df = self.eval_df.reset_index()

    def train_eval(self):
        """ The overarching train, test, evaluation wrapper that also saves algorithms and evaluation results

        """

        # Run the train/test split
        self.split_data()

        # Check whether supplied params designed for grid search:
        self.select_algs_grid_search()

        # Train algorithms; returns self.algs_dict 
        if self.grid_search_algs: # Perform hyperparameterization grid search for these algos
            self.train_algos_grid_search()

        if self.algo_config: # Just run a single simulation for these algos
            self.train_algos()

        # Calculate forestci uncertainty if enabled
        # Determine the best Random Forest model
        if 'rf' in self.algs_dict:  # Ensure 'rf' is in selected algorithms
            if self.grid_search_algs and 'gridsearchcv' in self.algs_dict['rf']:
                best_rf_algo = self.algs_dict['rf']['gridsearchcv'].best_estimator_.named_steps['randomforestregressor']
            else:
                best_rf_algo = self.algs_dict['rf']['algo']
                
            # Check if forestci is enabled in self.uncertainty
            forestci_enabled = any(
                d.get('fci_flag', False) for d in self.uncertainty.get('forestci', [])
            )
            # Compute forestci uncertainty with the best RF model
            if forestci_enabled:
                self.algs_dict['rf']['Uncertainty']['forestci'] = self.calculate_forestci_uncertainty(
                    best_rf_algo, np.array(self.X_train), np.array(self.X_test)
                )

        # Calculate Bagging uncertainty if enabled
        if any('n_algos' in d and d['n_algos'] for d in self.uncertainty.get('bagging', [])):
            for algo_dict in [self.algo_config, self.algo_config_grid]:  # Iterate over both configurations
                for algo_str in algo_dict.keys():  # algo_str is the correct algorithm name
                    # Select the best model if Grid Search was performed
                    best_algo = (
                        self.algs_dict[algo_str]['gridsearchcv'].best_estimator_
                        if self.grid_search_algs and 'gridsearchcv' in self.algs_dict[algo_str]
                        else self.algs_dict[algo_str]['algo']
                    )
    
                    # Compute Bagging CI (pass correct algorithm name + best model)
                    self.calculate_bagging_ci(algo_str, best_algo)
                
        # --- Calculate prediction intervals using MAPIE if enabled ---
        if any('alpha' in d and d['alpha'] for d in self.uncertainty.get('mapie', [])): #getattr(self, 'mapie_alpha', None):
            self.calculate_mapie()

        # Make predictions  (aka validation) 
        self.predict_algos()

        # Evaluate predictions; returns self.eval_dict
        self.evaluate_algos()

        # Write algorithms to file; returns self.algs_dict_paths
        self.save_algos()

        # Generate metadata dataframe
        self.org_metadata_alg() # Must be called after save_algos()


class UniversalDistanceClusterer(BaseEstimator, ClusterMixin):
    """
    Universal wrapper to handle ANY distance metric and ANY clustering algorithm,
    while guaranteeing a .predict() method exists for out-of-sample data.
    """
    def __init__(self, estimator, metric='gower'):
        self.estimator = estimator
        self.metric = metric
        # We use KNN(k=1) as a universal fallback for out-of-sample cluster assignment
        self.knn_fallback = KNeighborsClassifier(n_neighbors=1, metric='precomputed')

    def _get_distance(self, X1, X2=None):
        if self.metric == 'gower':
            return gower.gower_matrix(np.asarray(X1), np.asarray(X2) if X2 is not None else None)
        else:
            return pairwise_distances(X1, X2, metric=self.metric)

    def fit(self, X, y=None):
        if hasattr(X, "columns"): # Record feature names for sklearn compatibility
            self.feature_names_in_ = np.array(X.columns, dtype=object)
        self.X_train_ = np.asarray(X)
        dist_matrix = self._get_distance(self.X_train_)
        
        # Fit the underlying clustering algorithm using the precomputed distances
        self.labels_ = self.estimator.fit_predict(dist_matrix)
        
        # Train the KNN fallback so we can assign new basins to these clusters later
        self.knn_fallback.fit(dist_matrix, self.labels_)
        return self

    def predict(self, X):
        # Calculate distance from new ungaged basins to the training donor basins
        dist_to_train = self._get_distance(X, self.X_train_)
        
        # Use the KNN fallback to assign the new basin to the cluster of its nearest donor
        return self.knn_fallback.predict(dist_to_train)

def _extr_rf_algo(train_eval:AlgoTrainEval)->RandomForestRegressor:
    """Extract random forest from the algs_dict created by AlgoTrainEval class

    :param train_eval: The instantiated & processed AlgoTrainEval object
    :type train_eval: AlgoTrainEval
    :return: The trained random forest algorithm
    :rtype: RandomForestRegressor
    """
    if 'rf' in train_eval.algs_dict.keys():
        rfr = train_eval.algs_dict['rf']['algo']
    else:
        logging.info("Trained random forest object 'rf' non-existent in the provided AlgoTrainEval class object." \
              "Check to make sure the algo processing config file creates a random forest. Then make sure the ")
        rfr = None
    return rfr

class AlgoEvalPlotLC:
    def __init__(self,X,y):
        # The entire dataset of predictors/response    
        self.X = X
        self.y = y

        # Initialize Learning curve objects
        self.train_sizes_lc = np.empty(1)
        self.train_scores_lc = np.empty(1)
        self.valid_scores_lc = np.empty(1)


    def gen_learning_curve(self,model, cv = 5,n_jobs=-1,
                            train_sizes =np.linspace(0.1, 1.0, 10),
                            scoring = 'neg_mean_squared_error'
                            ):
        
        # Generate learning curve data
        self.train_sizes_lc, self.train_scores_lc, self.valid_scores_lc = learning_curve(
            model, self.X, self.y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, 
            scoring=scoring
        )

        # Calculate mean and standard deviation
        self.train_mean_lc = np.mean(-self.train_scores_lc, axis=1)  # Negate to get positive MSE
        self.train_std_lc = np.std(-self.train_scores_lc, axis=1)
        self.valid_mean_lc = np.mean(-self.valid_scores_lc, axis=1)
        self.valid_std_lc = np.std(-self.valid_scores_lc, axis=1)

    def plot_learning_curve(self,ylabel_scoring:str = "Mean Squared Error (MSE)",
                            title:str='Learning Curve',
                            training_uncn:bool = False):
        # GENERATE LEARNING CURVE FIGURE 
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_sizes_lc, self.train_mean_lc, 'o-', label='Training error')
        plt.plot(self.train_sizes_lc, self.valid_mean_lc, 'o-', label='Cross-validation error')
        if training_uncn:
            plt.fill_between(self.train_sizes_lc, self.train_mean_lc - self.train_std_lc, self.train_mean_lc + self.train_std_lc, alpha=0.1, color="r", label='Training uncertainty')
        plt.fill_between(self.train_sizes_lc, self.valid_mean_lc - self.valid_std_lc, self.valid_mean_lc + self.valid_std_lc, alpha=0.1, color="g", label='Cross-validation uncertainty')
        plt.xlabel('Training Size', fontsize = 18)
        plt.ylabel(ylabel_scoring, fontsize = 18)
        plt.title(title)
        plt.legend(loc='best',fontsize=15)
        plt.grid(True)

        # Adjust tick parameters for larger font size 
        plt.tick_params(axis='both', which='major', labelsize=15)
        plt.tick_params(axis='both', which='minor', labelsize=15)

        fig = plt.gcf()
        return fig
    
    def extr_modl_algo_train(self, train_eval:AlgoTrainEval):
        modls = list(train_eval.algs_dict.keys())

        for k, v in train_eval.algs_dict.items():
            v['algo']

def plot_learning_curve_save_wrap(algo_plot:AlgoEvalPlotLC, train_eval:AlgoTrainEval, 
                            dir_out_viz_base:str|Path,
                            ds:str,
                            cv:int = 5,n_jobs:int=-1,
                            train_sizes = np.linspace(0.1, 1.0, 10),
                            scoring:str = 'neg_mean_squared_error',
                            ylabel_scoring:str = "Mean Squared Error (MSE)",
                            training_uncn:bool = False
                            ):
    """Wrapper to generate & write learning curve plots forsklearn ML algorithms

    :param algo_plot: The initialized AlgoEvalPlotLC object with the full predictor matrix and response variable values
    :type algo_plot: AlgoEvalPlotLC
    :param train_eval: The initialized AlgoTrainEval class object
    :type train_eval: AlgoTrainEval
    :param dir_out_viz_base: The base directory for saving plots
    :type dir_out_viz_base: str | os.PathLike
    :param ds: The unique dataset name
    :type ds: str
    :param cv: The number of folds in a K-fold cross validation, defaults to 5
    :type cv: int, optional
    :param n_jobs: The number of parallel jobs, defaults to -1 for using all available cores
    :type n_jobs: int, optional
    :param train_sizes: Relative or absolute numbers of training examples that will be used 
      to generate the learning curve, defaults to np.linspace(0.1, 1.0, 10)
    :type train_sizes: array-like, optional
    :param scoring: A str or a scorrer collable object/function, defaults to 'neg_mean_squared_error'
    :type scoring: str, optional
    :param ylabel_scoring: Learning curve plot's y-axis label representing scoring metric, defaults to "Mean Squared Error (MSE)"
    :type ylabel_scoring: str, optional
    :param training_uncn: Should training uncertainty be represented as a shaded object?, defaults to False
    :type training_uncn: bool, optional

    """
    algs_dict = train_eval.algs_dict
    eval_dict = train_eval.eval_dict

    # Looping over e/ algo inside algs_dict from AlgoTrainEval.train_eval
    for algo_str, val in algs_dict.items():
        best_algo = val['pipeline']
        metr = eval_dict[algo_str]['metric']
        full_algo_str = eval_dict[algo_str]['type'].title()

        # Generate custom plot title
        cstm_title = f'{full_algo_str} Learning Curve: {metr} - {ds}'
        algo_str_file = f'{algo_str}' # Custom filepath string (e.g. 'rf', 'mlp')
        
        # Generate learning curve data
        algo_plot.gen_learning_curve(model=best_algo, cv=cv,n_jobs=n_jobs,
                            train_sizes =train_sizes,scoring=scoring)
        # Create learning curve figure
        fig_lc = algo_plot.plot_learning_curve(ylabel_scoring=ylabel_scoring,
                            title=cstm_title,training_uncn=training_uncn)
        # Standardize filepath to learning curve
        path_plot_lc = plots.std_lc_plot_path(dir_out_viz_base, ds, metr, algo_str = algo_str_file)
    
        fig_lc.savefig(path_plot_lc)

        plt.clf()
        plt.close()

#%% rafts_proc_algo script catch-all function (paralellized) -----------


def _process_single_metric(args_dict):
    """Worker function to train and evaluate a single metric safely."""
    metr = args_dict['metr']
    
    # Custom logger formatting for the worker to identify which metric is logging
    log_prefix = f"[{metr}]"
    logging.info(f"{log_prefix} Starting processing...")

    try:
        # 1. Instantiate and run AlgoTrainEval
        train_eval = AlgoTrainEval(
            df=args_dict['df_pred_resp'], attrs=args_dict['attrs_sel'], 
            algo_config=args_dict['algo_config'], uncertainty=args_dict['uncertainty_cfg'], 
            dir_out_alg_ds=args_dict['dir_out_alg_ds'], dataset_id=args_dict['ds'],
            metr=metr, task_type = args_dict['task_type'],test_size=args_dict['test_size'], rs=args_dict['seed'], 
            test_ids = args_dict.get('test_ids'),
            test_id_col=args_dict['col_locid'], verbose=args_dict['verbose'], 
            confidence_levels=args_dict['confidence_levels'],
            uncn_bnd_algo=args_dict['uncn_bnd_algo'], min_lim=args_dict['min_lim'], 
            max_lim=args_dict['max_lim'], save_all_clusters=args_dict['save_all_clusters'],
            n_jobs=args_dict['n_jobs']
        )
        train_eval.train_eval()

        # 2. Save evaluation metrics
        path_eval_metr = utils.std_eval_metrs_path(args_dict['dir_out_viz_base'], args_dict['ds'], metr)
        train_eval.eval_df.to_csv(path_eval_metr)

        # 3. Handle Plotting (RF Importance, Learning Curves, Maps, etc.)
        if args_dict['make_plots']:
            
            logging.info(f"{log_prefix} Generating plots...")
            
            # Extract data safely outside so learning curves can use them regardless of the model
            df_X, y_all = train_eval.all_X_all_y()
            out_viz_dir = Path(args_dict['dir_out_viz_base']) / args_dict['ds']
            out_viz_dir.mkdir(parents=True, exist_ok=True)

            # --- DYNAMIC FEATURE IMPORTANCE FOR ALL COMPATIBLE MODELS ---
            for algo_str, algo_info in train_eval.algs_dict.items():
                model = algo_info['algo']
                pipeline = algo_info['pipeline'] 
                # Retrieve .feature_importances_ natively if supported (rf, xgb, gbr, adaboost)
                imp_train = getattr(model, "feature_importances_", None)

                if imp_train is None:
                    # Calculate permutation importance (requires X and y data)
                    # NOTE the following permutation importances are based on training data to be 
                    #. consistent with getattr(model,'feature_importances_'). Using validation/testing
                    #. data instead would be more appropriate
                    try:
                        result = permutation_importance(
                            estimator=pipeline, 
                            X=train_eval.X_train,
                            y=train_eval.y_train, 
                            n_repeats=5, 
                            random_state=args_dict['seed'],
                            n_jobs=args_dict['n_jobs']
                        )
                        
                        # Extract the mean importances to match the 1D array format of .feature_importances_
                        imp_train = result.importances_mean
                    except: 
                        logging.info(f"The {algo_str} does not estimate feature importance")
                
                if imp_train is not None:
                    # Save importances to CSV
                    fi_df = pd.DataFrame({"feature": df_X.columns, "importance": imp_train}).sort_values("importance", ascending=False)
                    fi_csv = out_viz_dir / f"{algo_str}_feature_importance_train_{args_dict['ds']}_{metr}.csv"
                    fi_df.to_csv(fi_csv, index=False)
                    logging.info(f"{log_prefix} Wrote {algo_str} feature importances to {fi_csv}")

                    # Plot importances using the generalized wrapper
                    plots.save_feat_imp_fig_wrap(
                        model=model, 
                        attrs=df_X.columns, 
                        dir_out_viz_base=args_dict['dir_out_viz_base'], 
                        ds=args_dict['ds'], 
                        metr=metr, 
                        algo_str=f"{algo_str}_Training"
                    )
                    logging.info(f"{log_prefix} Wrote {algo_str} train feature importances to {out_viz_dir}")
                                    
            
            try: # Now create the feature importance plot on test data:
                result_test = permutation_importance(
                    estimator=pipeline, 
                    X=train_eval.X_test,  
                    y=train_eval.y_test,  
                    n_repeats=5, 
                    random_state=args_dict['seed'],
                    n_jobs=args_dict['n_jobs']
                )
                imp_test = result_test.importances_mean
                # --- Save and Plot Testing Importances ---
                fi_test_df = pd.DataFrame({"feature": df_X.columns, "importance": imp_test}).sort_values("importance", ascending=False)
                fi_test_csv = out_viz_dir / f"{algo_str}_feature_importance_test_{args_dict['ds']}_{metr}.csv"
                fi_test_df.to_csv(fi_test_csv, index=False)

                
                fig_test = plots.plot_feature_importance(
                    feat_imprt=imp_test, 
                    attrs=df_X.columns, 
                    title=f"{algo_str.upper()} Testing Feature Importance: {args_dict['ds']}"
                )
                path_fig_test = out_viz_dir / f"{algo_str}_feature_importance_test_{args_dict['ds']}_{metr}.png"
                fig_test.savefig(path_fig_test, bbox_inches='tight')
                plt.close(fig_test)

                logging.info(f"{log_prefix} Wrote {algo_str} test feature importances to {out_viz_dir}")
            except Exception as e: 
                    logging.info(f"The {algo_str} does not estimate feature importance. Error: {e}")
            
            # Create learning curves for each algorithm
            if args_dict['task_type'] != 'clustering':
                algo_plot_lc = AlgoEvalPlotLC(df_X, y_all)
                plot_learning_curve_save_wrap(algo_plot_lc, train_eval, 
                                dir_out_viz_base=args_dict['dir_out_viz_base'],
                                ds=args_dict['ds'],
                                cv=5, n_jobs=1,
                                train_sizes=np.linspace(0.1, 1.0, 10),
                                scoring='neg_mean_squared_error',
                                ylabel_scoring="Mean Squared Error (MSE)",
                                training_uncn=False
                                )
                
            # Calculate global min and max for consistent uncertainty scaling
            min_err, max_err = float('inf'), float('-inf')
            for algo_str in train_eval.algs_dict.keys():
                if train_eval.preds_dict[algo_str].get('y_pis', None) is not None:
                    y_pred = train_eval.preds_dict[algo_str]['y_pred']
                    y_pis = train_eval.preds_dict[algo_str]['y_pis']
                    for alpha_val in next(d['alpha'] for d in args_dict['uncertainty_cfg'].get('mapie', [])):
                        lower_err = y_pred - np.array([y_pis[i].loc['lower_limit', f'alpha_{alpha_val:.2f}'] for i in range(len(y_pred))])
                        upper_err = np.array([y_pis[i].loc['upper_limit', f'alpha_{alpha_val:.2f}'] for i in range(len(y_pred))]) - y_pred
                        total_err = lower_err + upper_err
                        min_err, max_err = min(min_err, total_err.min()), max(max_err, total_err.max())
        # ----- Extract y_pred for each algorithm and build output DataFrames -----
        dict_test_gdf = dict()
        col_locid = args_dict['col_locid']
        ds = args_dict['ds']
        dir_out_viz_base = args_dict['dir_out_viz_base']
        
        for algo_str in train_eval.algs_dict.keys():
            y_pred = train_eval.preds_dict[algo_str].get('y_pred')
            if args_dict['task_type'] == 'clustering':
                y_obs = None
            else:
                y_obs = train_eval.y_test.values
            
            r2_val = train_eval.eval_dict[algo_str].get('r2', None)
            spearman_val = train_eval.eval_dict[algo_str].get('spearman',None)

            if args_dict['make_plots'] and args_dict['task_type'] != 'clustering':
                # Regression of testing holdout's prediction vs observation
                if train_eval.preds_dict[algo_str].get('y_pis', None) is not None:
                    y_pis = train_eval.preds_dict[algo_str].get('y_pis')
                    for alpha_val in next(d['alpha'] for d in args_dict['uncertainty_cfg'].get('mapie', [])):
                        plots.plot_pred_vs_obs_wrap_mapie(
                            y_pred, y_obs, dir_out_viz_base, ds, metr, algo_str=algo_str,
                            y_pis=y_pis, alpha_val=alpha_val, split_type=f"testing{args_dict['test_size']}",
                            r2_val=r2_val,spearman_val=spearman_val
                        )
                else:
                    plots.plot_pred_vs_obs_wrap(
                        y_pred, y_obs, dir_out_viz_base, ds, metr, 
                        algo_str=algo_str, split_type=f"testing{args_dict['test_size']}",
                        r2_val=r2_val,spearman_val=spearman_val
                    )
                        
            # PREPARE THE GDF TO ALIGN PREDICTION VALUES BY COMIDS/COORDS
            comids_test = train_eval.df[col_locid].loc[train_eval.X_test.index].values
            test_gdf = args_dict['gdf_comid'][args_dict['gdf_comid'][col_locid].isin(comids_test)].copy()
            
            if args_dict['task_type'] == 'clustering':
                df_test = pd.DataFrame({col_locid: comids_test, 'observed': np.nan})
            else:
                df_test = train_eval.df.loc[train_eval.y_test.index][[col_locid, metr]].rename(columns={metr:'observed'})
            df_test['prediction'] = y_pred

            test_gdf = test_gdf.merge(df_test, left_on=col_locid, right_on=col_locid, how='left')
            test_gdf.loc[:, 'dataset'] = ds
            test_gdf.loc[:, 'metric'] = metr
            test_gdf.loc[:, 'algo'] = algo_str
            test_gdf.drop_duplicates(subset=[col_locid, 'observed', 'prediction'], inplace=True)

            dict_test_gdf[algo_str] = test_gdf

            if args_dict['make_plots']:
                plots.plot_map_pred_wrap(
                    test_gdf, dir_out_viz_base, ds, metr, algo_str,
                    split_type='test', colname_data='prediction',
                    epsg_reproj = 4326, task_type=args_dict['task_type']
                )
                
                # Test Prediction Uncertainty Plotting 
                if train_eval.preds_dict[algo_str].get('y_pis', None) is not None:
                    y_pis = train_eval.preds_dict[algo_str].get('y_pis')
                    for alpha_val in next(d['alpha'] for d in args_dict['uncertainty_cfg'].get('mapie', [])):
                        test_gdf[f'mapie_lower_{alpha_val:.2f}'] = [y_pis[i].loc['lower_limit', f'alpha_{alpha_val:.2f}'] for i in range(len(y_pis))]
                        test_gdf[f'mapie_upper_{alpha_val:.2f}'] = [y_pis[i].loc['upper_limit', f'alpha_{alpha_val:.2f}'] for i in range(len(y_pis))]
                        
                        plots.plot_map_pred_wrap_uncn(
                            test_gdf=test_gdf, 
                            dir_out_viz_base=dir_out_viz_base, 
                            ds=ds, 
                            metr=metr, 
                            algo_str=algo_str,
                            alpha_val=alpha_val, 
                            uncn_col=None,
                            split_type='test_mapie', 
                            colname_data='prediction',
                            epsg_reproj=4326
                        )                   
                                
        # Generate analysis path out and SAVE the critical CSV
        path_pred_obs = utils.std_test_pred_obs_path(args_dict['dir_out_anlys_base'], ds, metr)
        #df_pred_obs_ds_metr = pd.concat(dict_test_gdf.values())
        df_pred_obs_ds_metr = pd.concat(dict_test_gdf)
        df_pred_obs_ds_metr.to_csv(path_pred_obs)
        logging.info(f"{log_prefix} Wrote prediction-observation dataset to {path_pred_obs}")
        # ... [Your existing learning curve and map plotting logic goes here] ...

        logging.info(f"{log_prefix} Successfully completed.")
        return metr, train_eval.eval_df

    except Exception as e:
        logging.error(f"{log_prefix} FAILED with error: {e}")
        logging.error(traceback.format_exc()) #
        return metr, None

    finally:
        # FORCE memory cleanup on the worker before it closes
        plt.close('all')
        if 'train_eval' in locals():
            del train_eval
        gc.collect()

def assign_donors_to_receivers(
    df_donors: pd.DataFrame,
    df_receivers: pd.DataFrame,
    attrs: list,
    metric: str = 'euclidean',
    cluster_col: str = 'prediction',
    id_col: str = 'featureID',
    max_spa_dist: float = None,
    spatial_cols: tuple = ('lon', 'lat'),
    spatial_crs: str = 'EPSG:4326',
) -> pd.DataFrame:
    """Pairs each receiver basin with the most similar donor basin within its assigned
    cluster. Tracks and reports any receiver basins that fail to receive a donor assignment.

    :param df_donors: Donor basins, with `attrs`, `cluster_col`, `id_col`, and (only when
        `max_spa_dist` is set) `spatial_cols`.
    :type df_donors: pd.DataFrame
    :param df_receivers: Receiver basins, with the same required columns as `df_donors`.
    :type df_receivers: pd.DataFrame
    :param attrs: Attribute columns defining the attribute-space distance.
    :type attrs: list
    :param metric: 'gower' or any :mod:`sklearn` pairwise-distance metric, defaults to
        'euclidean'.
    :type metric: str, optional
    :param cluster_col: Column holding each row's assigned cluster, defaults to 'prediction'.
    :type cluster_col: str, optional
    :param id_col: Column identifying each row, defaults to 'featureID'.
    :type id_col: str, optional
    :param max_spa_dist: Optional maximum spatial distance (km) a receiver's assigned donor
        may be from it. None (default) preserves the original attribute-only
        nearest-neighbor behavior exactly -- the spatial ranking/fallback logic below is
        skipped entirely, so every existing caller is unaffected. When set, each receiver
        still prefers its attribute-nearest donor, but only among candidates within this
        spatial distance; if none qualify, falls back to the plain attribute-nearest donor
        regardless of distance (a receiver is never left unassigned just because every
        in-cluster candidate was spatially far), and that fallback is tallied separately.
        Defaults to None.
    :type max_spa_dist: float, optional
    :param spatial_cols: Column names holding (x, y)-style coordinates, used only when
        `max_spa_dist` is set. Defaults to ('lon', 'lat').
    :type spatial_cols: tuple, optional
    :param spatial_crs: The CRS `spatial_cols` are expressed in; reprojected to EPSG:5070 for
        a true-distance (km) comparison against `max_spa_dist`. Used only when `max_spa_dist`
        is set. Defaults to 'EPSG:4326'.
    :type spatial_crs: str, optional
    :return: One row per paired receiver: receiver_id, donor_id, cluster_id,
        distance_to_donor (attribute distance). When `max_spa_dist` is set, also
        spatial_dist_to_donor_km and used_spatial_fallback.
    :rtype: pd.DataFrame
    """
    pairing_results = []
    spatial_fallback_receivers = []

    if max_spa_dist is not None:
        # Reproject once, up front, rather than per-cluster -- the subsetting below
        # (donors_in_clust/receivers_in_clust) carries these columns through naturally.
        df_donors = df_donors.copy()
        df_receivers = df_receivers.copy()
        donor_pts = gpd.GeoSeries(gpd.points_from_xy(df_donors[spatial_cols[0]], df_donors[spatial_cols[1]]),
                                   crs=spatial_crs).to_crs('EPSG:5070')
        receiver_pts = gpd.GeoSeries(gpd.points_from_xy(df_receivers[spatial_cols[0]], df_receivers[spatial_cols[1]]),
                                      crs=spatial_crs).to_crs('EPSG:5070')
        df_donors['_spa_x'], df_donors['_spa_y'] = donor_pts.x.values, donor_pts.y.values
        df_receivers['_spa_x'], df_receivers['_spa_y'] = receiver_pts.x.values, receiver_pts.y.values

    # Track initial receivers to verify completeness at the end
    initial_receivers = set(df_receivers[id_col])

    # Identify receivers that missed predictions entirely (NaN values)
    unassigned_missing_cluster = df_receivers[df_receivers[cluster_col].isna()][id_col].tolist()
    if unassigned_missing_cluster:
        logging.warning(f"{len(unassigned_missing_cluster)} receivers have NaN cluster predictions and will be skipped.")

    unique_clusters = df_receivers[cluster_col].dropna().unique()
    unassigned_empty_donor_cluster = []

    for cluster_id in unique_clusters:
        donors_in_clust = df_donors[df_donors[cluster_col] == cluster_id].reset_index(drop=True)
        receivers_in_clust = df_receivers[df_receivers[cluster_col] == cluster_id].reset_index(drop=True)

        if donors_in_clust.empty:
            logging.warning(f"No donors found for Cluster {cluster_id}. {len(receivers_in_clust)} receivers unassigned.")
            # Record the specific receivers that are dropped here
            unassigned_empty_donor_cluster.extend(receivers_in_clust[id_col].tolist())
            continue

        X_donor = donors_in_clust[attrs]
        X_recv = receivers_in_clust[attrs]

        if max_spa_dist is None:
            # Calculate Nearest Neighbor
            if metric == 'gower':
                dist_matrix = gower.gower_matrix(np.asarray(X_recv), np.asarray(X_donor))
                closest_donor_indices = np.argmin(dist_matrix, axis=1)
                distances = np.min(dist_matrix, axis=1)
            else:
                nn = NearestNeighbors(n_neighbors=1, metric=metric)
                nn.fit(X_donor)
                distances, closest_donor_indices = nn.kneighbors(X_recv)
                distances = distances.flatten()
                closest_donor_indices = closest_donor_indices.flatten()

            # Record the pairings
            clust_pairings = pd.DataFrame({
                'receiver_id': receivers_in_clust[id_col],
                'donor_id': donors_in_clust.loc[closest_donor_indices, id_col].values,
                'cluster_id': cluster_id,
                'distance_to_donor': distances
            })
        else:
            # Full attribute-distance matrix (receivers x donors) -- each receiver ranks
            # every in-cluster donor by attribute distance, rather than only ever seeing the
            # single nearest one, so it can walk down that ranking until a spatially-eligible
            # candidate is found.
            if metric == 'gower':
                dist_matrix = gower.gower_matrix(np.asarray(X_recv), np.asarray(X_donor))
            else:
                dist_matrix = pairwise_distances(np.asarray(X_recv), np.asarray(X_donor), metric=metric)

            donor_x, donor_y = donors_in_clust['_spa_x'].values, donors_in_clust['_spa_y'].values
            recv_x, recv_y = receivers_in_clust['_spa_x'].values, receivers_in_clust['_spa_y'].values

            donor_ids_sel, attr_dists_sel, spa_dists_sel, used_fallback = [], [], [], []
            for i in range(len(receivers_in_clust)):
                order = np.argsort(dist_matrix[i])
                spa_dists_row_km = np.sqrt((donor_x - recv_x[i]) ** 2 + (donor_y - recv_y[i]) ** 2) / 1000.0
                within_range = [j for j in order if spa_dists_row_km[j] <= max_spa_dist]
                chosen = within_range[0] if within_range else order[0]
                used_fallback.append(not within_range)
                donor_ids_sel.append(donors_in_clust.loc[chosen, id_col])
                attr_dists_sel.append(dist_matrix[i, chosen])
                spa_dists_sel.append(spa_dists_row_km[chosen])

            clust_pairings = pd.DataFrame({
                'receiver_id': receivers_in_clust[id_col].values,
                'donor_id': donor_ids_sel,
                'cluster_id': cluster_id,
                'distance_to_donor': attr_dists_sel,
                'spatial_dist_to_donor_km': spa_dists_sel,
                'used_spatial_fallback': used_fallback,
            })
            spatial_fallback_receivers.extend(
                rid for rid, fb in zip(receivers_in_clust[id_col].values, used_fallback) if fb)

        pairing_results.append(clust_pairings)

    if pairing_results:
        df_final_pairings = pd.concat(pairing_results, ignore_index=True)
    else:
        df_final_pairings = pd.DataFrame()

    # --- Final Reconciliation and Reporting ---
    paired_receivers = set(df_final_pairings['receiver_id']) if not df_final_pairings.empty else set()
    missed_receivers = initial_receivers - paired_receivers

    if missed_receivers:
        logging.warning(f"A total of {len(missed_receivers)} receivers were NOT assigned a donor.")
        if unassigned_empty_donor_cluster:
            logging.warning(f"Missed due to empty donor clusters: {unassigned_empty_donor_cluster}")
        if unassigned_missing_cluster:
            logging.warning(f"Missed due to missing (NaN) cluster predictions: {unassigned_missing_cluster}")
    else:
        logging.info("All receivers were successfully assigned a donor.")

    if max_spa_dist is not None and spatial_fallback_receivers:
        logging.warning(f"{len(spatial_fallback_receivers)} receiver(s) had no donor within "
                         f"max_spa_dist={max_spa_dist}km; paired to the attribute-nearest donor "
                         f"regardless of distance instead.")

    return df_final_pairings


