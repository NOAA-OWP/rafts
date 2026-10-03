# utils.py
from __future__ import annotations # enables the | operator for function typehints back to python 3.7
import inspect
import numpy as np
import pandas as pd
import xarray as xr
import pynhd as nhd
import dask.dataframe as dd
import os
from collections.abc import Iterable
from typing import List, Union, Optional, Dict, Any, Tuple
from pathlib import Path
import itertools
import yaml
import logging
import re
from shapely.geometry import Point
import geopandas as gpd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import ast
from sklearn.model_selection import train_test_split
import joblib
import sys 
import sqlite3
import json

# Set up basic logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import schemas here to avoid circular dependency, they are separate modules
try:
    import rafts_algo.schemas.schemas as schemas
    from rafts_algo.schemas.pydantic_schemas import ModelMetadata
except ImportError:
    logging.warning("Cannot import schemas: rafts_algo/schemas/schemas.py and pydantic_schemas.py are required for validation utilities.")

# %% ALGO CONFIG FILE PARSER
class AlgoConfigParser:
    ## Initialize a new instance of the class AlgoConfigParser
    def __init__(self, path_algo_config: str | os.PathLike):
        self.path_algo_config = path_algo_config
        self.algo_cfg_unc_dict = dict()

    ## Define a function to read algo configuration parameters from the YAML config file
    def _read_algo_config(self ) -> dict:
        """A function that extracts parameters for the algorith and uncertainty from yaml configuration file.
        
        :param path_algo_config: Full path (folder and name of file) of algorithm config file
        :type path_algo_config: str | os.PathLike
        Returns a dictionary with two dictionaries containing parameter values.
            :param algorithms: Selected algorithm(s), refer to AlgoTrainEval.train_algos to see what options are present (e.g. rf, mlp)
            :type algorithms: string
                :param rf: Random Forest algorithm
                :type rf: string, OPTIONAL, STRONGLY RECOMMENDED
                    :Additional rf parameters can be included, refer to sklearn.ensemble.RandomForestRegressor for arguments to pass here - otherwise defaults will be used
                :param mlp: Multi-layer Perceptron Regressor Algorithm
                :type mlp: string, OPTIONAL
                    :Additional mlp parameters can be included, refer to sklearn.neural_network.MLPRegressor for arguments to pass here - otherwise defaults will be used
            :param test_size: The proportion of dataset to be used for testing, defaults to 0.3. All values should be between 0 and 1.
            :type test_size:  float
            :param seed: The starting point for the random number generator, defaults to 32
            :type seed: int
            :param read_type: 'filename' # Optional. Recommend 'filename' for faster data loading. Should all parquet files be lazy-loaded, assign 'all'. Otherwise just files with comids_resp in the file name? assign 'filename'.
            :param read_type: OPTIONAL. All parquet files loaded, assign 'all'. Just files with comids_resp in the filename, assign 'filename'. Defaults to 'all'
            :type read_type: str
            :param metrics: OPTIONAL. The metrics (hydraulic signature identifier) of interest for processing.
            Defaults to 'all', all metrics in the input dataset will be processed
            :type metrics: ??sublist structure??
            :param make_plots: If True plots are created & saved to file. Defaults to False
            :type make_plots: bool, OPTIONAL
            :param same_test_ids: Are all datasets being compared required to have the same test ID? If False, algos will be trained true to the test_size, 
            but the train_test split may not be the same across each dataset (particularly total basins differ). Defaults to True
            :type same_test_ids: bool, OPTIONAL
            :param verbose: Should the train/test/eval provide printouts on progress? Defaults to True.
            :type verbose: bool, OPTIONAL
            :param uncertainty_cfg: Uncertainty analyses to be run. Defaults to "{}"
            :type uncertainty_cfg: str, OPTIONAL
                :param confidence_levels: Confidence levels for confidence interval calculations.  
                :type confidence_levels: list, OPTIONAL, REQUIRED if a value was read for n_algos
                :param forestci: Applies forestci to estimate confidence intervals for the model training based on the variance of predictions from 
                trees in the random forest model. For more details, see: https://github.com/scikit-learn-contrib/forest-confidence-interval
                :type forestci: str, OPTIONAL
                    :param fci_flag: If True use Forestci model to calculate confidence interval for rf model
                    :type fci_flag: bool
                :param bagging: Enables bootstrap aggregating (bagging) to calculate confidence intervals during the model training 
                by training multiple models on resampled data. More broadly applicable than forestci.
                :type bagging: str, OPTIONAL
                    :param n_algos: Number of bootstrap runs for Bagging confidence interval calculation (integer). 
                    Bagging ci calculation is disabled if n_algos is empty
                    :type n_algos: int, OPTIONAL
                :param mapie: Applies the MAPIE (Model Agnostic Prediction Interval Estimator) framework to estimate **prediction intervals**, 
                which provide bounds around individual predicted values. Supports many model types
                See documentation for details: https://mapie.readthedocs.io/en/stable/index.html
                :type mapie: str, OPTIONAL
                    :param alpha: Alpha parameter (0 < α < 1) in an array format to calculate MAPIE prediction intervals
                    :type alpha: list, OPTIONAL, MAPIE prediction interval estimation will be enabled if not empty
                    :param method: MAPIE method: 'plus' (CV+) or 'minmax' (CV-minmax). For more information and other methods, refer to:
                    https://mapie.readthedocs.io/en/stable/theoretical_description_regression.html
                    :type method: str, OPTIONAL, but REQUIRED if MAPIE_alpha provided
                    :param cv: Specifies the number of cross-validation folds
                    :type cv: str, OPTIONAL, but REQUIRED if MAPIE_alpha provided
                    :param agg_function: Option selection, either 'mean' or 'median'
                    :type agg_function: str, OPTIONAL, but REQUIRED if MAPIE_alpha provided
        :return: Dictionary of algorithm input parameters, required and optional
        """
        # Changelog/contributions
        #  2025-07-29 - Converted rafts_proc_algo_viz.py config file read section to function, Justin Clark
        #. 2025-10-11 add AlgoTrainEval import & default seed, GL
        if not Path(self.path_algo_config).exists():
            logging.error("Ensure that algo config file is defined")
            raise ValueError(f"Ensure that algo config file is defined" )
        
        with open(self.path_algo_config, 'r') as file:
            algo_cfg = yaml.safe_load(file)

        # Algorithm selection and parameters
        algo_config = algo_cfg.get('algorithms')
        if algo_config is None:
            logging.error("Missing required 'algorithms' section in YAML config.")
            raise KeyError("Missing required 'algorithms' section in YAML config.")

        # Ensure the string literal is converted to a tuple for `hidden_layer_sizes`
        if algo_config.get('mlp',None):
            if algo_config['mlp'][0].get('hidden_layer_sizes',None): # purpose: evaluate string literal to a tuple
                algo_config['mlp'][0]['hidden_layer_sizes'] = ast.literal_eval(algo_config['mlp'][0]['hidden_layer_sizes'])
       
        # In some situations, we only care about parsing config files, not running algorithms
        #. The following helps avoid import the algo module
        if not algo_cfg.get('seed') or not algo_cfg.get('test_size'):
            # Need to import here to avoid circular dependency
            try:
                from rafts_algo.rafts_algo_train import AlgoTrainEval
                # Use the signature of AlgoTrainEval for default values
                sig = inspect.signature(AlgoTrainEval.__init__)  
                seed = sig.parameters['rs'].default
                test_size = sig.parameters['test_size'].default
            except:
                seed = 32
                test_size = 0.3
        else:
            seed = 32
            test_size = 0.3
        # Generate dictionary "algo_cfg_dict" with primary training parameters
        algo_cfg_dict = {'algo_config' : algo_config,
                         'task_type' : algo_cfg.get('task_type', 'regression'),
                         'save_all_clusters': algo_cfg.get('save_all_clusters',False),
                            'test_size': algo_cfg.get('test_size', test_size), # Must be between 0 and 1
                            'seed': algo_cfg.get('seed',seed ),
                            'read_type': algo_cfg.get('read_type', 'filename'), #DEFAULT to 'filename'
                            'metrics': algo_cfg.get('metrics', None),
                            'make_plots': algo_cfg.get('make_plots', False),
                            'same_test_ids': algo_cfg.get('same_test_ids', True),
                            'verbose': algo_cfg.get('verbose', True),
                            'name_attr_config': algo_cfg.get('name_attr_config', Path(self.path_algo_config).name.replace('algo', 'attr')),
                            'name_attr_csv': algo_cfg.get('name_attr_csv',None),
                            'colname_attr_csv': algo_cfg.get('colname_attr_csv',None),
                            'n_jobs': algo_cfg.get('n_jobs',None)
                            }
        
        algo_cfg_dict['path_attr_config'] = build_cfig_path(self.path_algo_config, algo_cfg_dict['name_attr_config'])

        if not algo_cfg_dict['path_attr_config'].exists():
            logging.error(f"Ensure that 'name_attr_config' as defined inside {self.path_algo_config.name} \
                              \n is also in the same directory as the algo config file {self.path_algo_config.parent}")
            raise ValueError(f"Ensure that 'name_attr_config' as defined inside {self.path_algo_config.name} \
                              \n is also in the same directory as the algo config file {self.path_algo_config.parent}" )

        # Type checks for common required parameters
        if not isinstance(algo_cfg_dict['test_size'], (float, int)):
            logging.error(f"'test_size' must be a float. Got: {type(algo_cfg_dict['test_size'])}")
            raise TypeError(f"'test_size' must be a float. Got: {type(algo_cfg_dict['test_size'])}")
        if not (0 < algo_cfg_dict['test_size'] < 1):
            logging.error(f"'test_size' must be between 0 and 1. Got: {algo_cfg_dict['test_size']}")
            raise ValueError(f"'test_size' must be between 0 and 1. Got: {algo_cfg_dict['test_size']}")
        if not isinstance(algo_cfg_dict['seed'], int):
            logging.error(f"'seed' must be an integer. Got: {type(algo_cfg_dict['seed'])}")
            raise TypeError(f"'seed' must be an integer. Got: {type(algo_cfg_dict['seed'])}")
        if not isinstance(algo_cfg_dict['read_type'], str):
            logging.error(f"'read_type' must be a string. Got: {type(algo_cfg_dict['read_type'])}")
            raise TypeError(f"'read_type' must be a string. Got: {type(algo_cfg_dict['read_type'])}")
        if algo_cfg_dict['read_type'] not in ['all', 'filename']:
            logging.error(f"'read_type' must be either 'all' or 'filename'. Got: {algo_cfg_dict['read_type']}")
            raise ValueError(f"'read_type' must be either 'all' or 'filename'. Got: {algo_cfg_dict['read_type']}")
        if not isinstance(algo_cfg_dict['make_plots'], bool):
            logging.error(f"'make_plots' must be a boolean. Got: {type(algo_cfg_dict['make_plots'])}")
            raise TypeError(f"'make_plots' must be a boolean. Got: {type(algo_cfg_dict['make_plots'])}")
        if not isinstance(algo_cfg_dict['same_test_ids'], bool):
            logging.error(f"'same_test_ids' must be a boolean. Got: {type(algo_cfg_dict['same_test_ids'])}")
            raise TypeError(f"'same_test_ids' must be a boolean. Got: {type(algo_cfg_dict['same_test_ids'])}")
        if not isinstance(algo_cfg_dict['verbose'], bool):
            logging.error(f"'verbose' must be a boolean. Got: {type(algo_cfg_dict['verbose'])}")
            raise TypeError(f"'verbose' must be a boolean. Got: {type(algo_cfg_dict['verbose'])}")
        
        # Generate dictionary "algo_unc_dict" with uncertainty parameters
        algo_unc_dict = {'uncertainty_cfg': algo_cfg.get("uncertainty", {})}

        if not isinstance(algo_unc_dict["uncertainty_cfg"], dict):
            logging.error("The 'uncertainty' block must be a dictionary")
            raise TypeError("The 'uncertainty' block must be a dictionary")

        # Generate dictionary with combined training and uncertainty parameters
        self.algo_cfg_unc_dict = {
            'algo_cfg_dict': algo_cfg_dict,
            'algo_unc_dict': algo_unc_dict}

        # Get uncertainty configuration
        uncertainty_cfg = self.algo_cfg_unc_dict["algo_unc_dict"]["uncertainty_cfg"]

        # Error check for confidence levels data type, and use function default if not provided
        if not algo_unc_dict == {}:
            try:
                from rafts_algo.rafts_algo_train import AlgoTrainEval
                # Use the signature of AlgoTrainEval for default values
                sig = inspect.signature(AlgoTrainEval.__init__)
                confidence_levels = uncertainty_cfg.get("confidence_levels", sig.parameters['confidence_levels'].default)
            except:
                confidence_levels = [95]
        else:
            confidence_levels = [95]

        # Update the algo_cfg & class object just-in-case
        self.algo_cfg_unc_dict["algo_unc_dict"]["uncertainty_cfg"]['confidence_levels'] = confidence_levels
        if not isinstance(confidence_levels, list):
            logging.error(f"'confidence_levels' must be a list of numeric values (e.g., [90, 95]). Got: {type(confidence_levels).__name__}")
            raise TypeError(f"'confidence_levels' must be a list of numeric values (e.g., [90, 95]). Got: {type(confidence_levels).__name__}")
        for level in confidence_levels:
                if not isinstance(level, (int, float)) or not (0 < level <= 100):
                    logging.error(f"Each 'confidence_levels' entry must be a number between 0 and 100. Got: {level}")
                    raise ValueError(f"Each 'confidence_levels' entry must be a number between 0 and 100. Got: {level}")

        # Validate MAPIE parameters if provided
        mapie_cfg = uncertainty_cfg.get("mapie")
        if mapie_cfg:
            # Flatten the list of dicts into one dictionary
            mapie_params = {k: v for d in mapie_cfg for k, v in d.items()}

            # Check A. If 'alpha' is provided, check that it is a list of floats between 0 and 1
            alpha = mapie_params.get("alpha")
            if alpha is not None:
                if not isinstance(alpha, list) or not all(isinstance(a, float) and 0 < a < 1 for a in alpha):
                    logging.error(f"'alpha' in MAPIE must be a list of floats between 0 and 1. Got: {alpha}")
                    raise ValueError(f"'alpha' in MAPIE must be a list of floats between 0 and 1. Got: {alpha}")

                # Check B. If alpha is provided, 'method' must be 'plus' or 'minmax'
                method = mapie_params.get("method")
                if method not in ["plus", "minmax"]:
                    logging.error(f"If 'alpha' is provided, 'method' must be 'plus' or 'minmax'. Got: {method}")
                    raise ValueError(f"If 'alpha' is provided, 'method' must be 'plus' or 'minmax'. Got: {method}")

                # Check C. If alpha is provided, 'cv' must also be defined and an int
                cv = mapie_params.get("cv")
                if not isinstance(cv, int):
                    logging.error(f"If 'alpha' is provided, 'cv' must also be defined and must be an integer. Got: {cv}")
                    raise TypeError(f"If 'alpha' is provided, 'cv' must also be defined and must be an integer. Got: {cv}")

                # Check D. If alpha is provided, 'agg_function' must be 'mean' or 'median'
                agg_function = mapie_params.get("agg_function")
                if agg_function not in ["mean", "median"]:
                    logging.error(f"If 'alpha' is provided, 'agg_function' must be 'mean' or 'median'. Got: {agg_function}")
                    raise ValueError(f"If 'alpha' is provided, 'agg_function' must be 'mean' or 'median'. Got: {agg_function}")

# %% BASIN ATTRIBUTES (PREDICTORS) & RESPONSE VARIABLES (e.g. METRICS)
class AttrConfigAndVars:
    def __init__(self, path_attr_config: str | os.PathLike):
        self.path_attr_config = path_attr_config
        self.attrs_cfg_dict = dict()
        self.attr_config = dict()

    def _read_attr_config(self ) -> dict:
        """Extract the desired basin attribute variable names from yaml file

        :raises warnings.warn: Assumes all attributes desired if not specified
        :return: dictionary of required configuration items: 
            - `attrs_sel`: attributes `list[str]`
            - `dir_db_attrs`: directory where attribute .parquet files live `list[str]`
            - `dir_std_base`: directory of standardized data generated by :mod:`rafts_prep`, `list[str]`
            - `dir_base`: base directory for file output, `list[str]`
            - `datasets`: dataset names, `list[str]`
        :rtype: dict
        """

        if not Path(self.path_attr_config).exists():
            logging.error("Attribute config path does not exist")
            raise ValueError(f"Attribute config path does not exist")

        # Attribute data location:
        with open(self.path_attr_config, 'r') as file:
            self.attr_config = yaml.safe_load(file)

        # identify attribute data of interest from attr_config
        attrs_all = [v for x in self.attr_config['attr_select'] for k,v in x.items() if '_vars' in k]
        attrs_all_filtered = [attr for attr in attrs_all if attr is not None]
        attrs_sel = [x for x in list(itertools.chain(*attrs_all_filtered)) if x is not None]

        if len(attrs_sel) == 0: # If no attributes generated, assume all attributes are of interest
            attrs_sel = 'all'
            logging.warning("No attributes discerned from 'attr_select'. Assuming all attributes desired.")
        
        home_dir = _define_home_dir(self.attr_config)

        dir_base = Path(list([x for x in self.attr_config['file_io'] if 'dir_base' in x][0].values())[0].format(home_dir=home_dir))
    
        # Location of attributes (predictor data):
        try:
            raw_path = list([x for x in self.attr_config['file_io'] if 'dir_db_attrs' in x][0].values())[0]
            # 1. Escape {ds} by turning it into {{ds}}
            # 2. Format dir_base and home_dir
            # 3. Python automatically turns {{ds}} back into {ds}
            formatted_path = raw_path.replace('{ds}', '{{ds}}').format(dir_base=dir_base, home_dir=home_dir)
            dir_db_attrs = Path(formatted_path)
            
        except Exception as e:
            # Catch specific exceptions so we don't fail silently anymore!
            logging.error(f"Failed to parse 'dir_db_attrs'. Error: {e}")
            raise ValueError(f"Could not parse 'dir_db_attrs' from config: {e}")
        # parent location of response variable data:
        dir_std_base = Path(list([x for x in self.attr_config['file_io'] if 'dir_std_base' in x][0].values())[0].format(dir_base=dir_base, home_dir=home_dir))

        # Determine the datasets
        file_io = self.attr_config.get('file_io', [])
        # 1. Safely extract the prep config name from the list of dicts
        name_prep_config = next((item['name_prep_config'] for item in file_io if 'name_prep_config' in item), None)
        datasets = []
        # 2. If a prep config is linked, read it to extract the dataset_name
        if name_prep_config:
            path_prep_config = build_cfig_path(self.path_attr_config,name_prep_config)
            if path_prep_config.exists():
                with open(path_prep_config, 'r') as f:
                    prep_cfg = yaml.safe_load(f)
                prep_form_meta = prep_cfg.get('formulation_metadata', [])
                dataset_name = next((item['dataset_name'] for item in prep_form_meta if 'dataset_name' in item), None)
                if dataset_name:
                    datasets = [dataset_name] # Wrap in a list for downstream loops
            else:
                logging.warning(f"Prep config {path_prep_config.name} not found.")
                
        # 3. DURABLE FALLBACK: Legacy behavior if no prep config is found or dataset_name is missing
        if not datasets:
            try: 
                datasets = list([x for x in self.attr_config.get('formulation_metadata', []) if 'datasets' in x][0].values())[0]
            except Exception as e:
                logging.error("Could not resolve dataset names from either the prep config or attribute config.")
                raise ValueError("Could not resolve dataset names from either the prep config or attribute config.")

        # Compile output
        self.attrs_cfg_dict = {'attrs_sel' : attrs_sel,
                            'dir_db_attrs': dir_db_attrs,
                            'dir_std_base': dir_std_base,
                            'dir_base': dir_base,
                            'home_dir': home_dir,
                            'datasets': datasets}

# %% PREDICTION CONFIGURATION
class PredConfigParser:
    def __init__(self, path_pred_config: str):
        self.path_pred_config = path_pred_config
        self.pred_cfg_dict = None

    def _read_pred_config(self) -> dict:
        """
        Read and parse the prediction configuration YAML file.

        Sets:
            self.pred_cfg_dict (dict): Parsed and formatted config items, including:
                - Required: `name_attr_config`, `name_algo_config`, 
                            `path_meta`, `ds_type`, `pred_file_comid_colname`,
                - Parsed from attribute config: `datasets`, `dir_base`, `dir_std_base`, `home_dir`
                - Optional: `algo_response_vars`, `algo_type`, `MAPIE_alpha`, `donor_map_states`
        Returns:
            dict: Same dictionary stored in self.pred_cfg_dict
        Raises:
            ValueError: If any REQUIRED config fields are missing.
        """

        if not Path(self.path_pred_config).exists():
            logging.error(f"Prediction config file not found: {self.path_pred_config}")
            raise FileNotFoundError(f"Prediction config file not found: {self.path_pred_config}")

        # Load prediction config YAML
        with open(self.path_pred_config, 'r') as file:
            pred_cfg = yaml.safe_load(file)

        # --- Required top-level prediction keys ---
        required_pred_keys = [
            "name_attr_config", "name_algo_config",
            "ds_type", "path_meta", "pred_file_comid_colname"
        ]

        missing_keys = [k for k in required_pred_keys if k not in pred_cfg or pred_cfg[k] is None]
        if missing_keys:
            logging.error(f"Missing required keys in prediction config file: {missing_keys}\nConfig path: {self.path_pred_config}\n")
            raise ValueError(
                f"Missing required keys in prediction config file: {missing_keys}\n"
                f"Config path: {self.path_pred_config}\n"
            )

        # Extract required fields
        name_attr_config     = pred_cfg["name_attr_config"]
        name_algo_config     = pred_cfg["name_algo_config"]
        write_type           = pred_cfg.get("write_type",'parquet')
        ds_type              = pred_cfg["ds_type"]
        path_meta            = pred_cfg["path_meta"]
        pred_file_comid_colname = pred_cfg["pred_file_comid_colname"]

        # Resolve full path to attribute config
        path_attr_config = build_cfig_path(self.path_pred_config, name_attr_config)
        # TODO Integrate AttrConfigParser here once available

        # --- Load attribute config using AttrConfigAndVars ---
        attr_cfg = AttrConfigAndVars(path_attr_config)
        attr_cfg._read_attr_config()

        home_dir     = attr_cfg.attrs_cfg_dict.get('home_dir') 
        dir_base     = attr_cfg.attrs_cfg_dict.get('dir_base')
        dir_std_base = attr_cfg.attrs_cfg_dict.get('dir_std_base')
        datasets     = attr_cfg.attrs_cfg_dict.get('datasets')

        if dir_base and not Path(dir_base).is_absolute():
            dir_base = str((Path(path_attr_config).parent / dir_base).resolve())
        if dir_std_base and not Path(dir_std_base).is_absolute():
            dir_std_base = str((Path(path_attr_config).parent / dir_std_base).resolve())

        # Check if dir_base exists
        if not Path(dir_base).exists():
            logging.error(f"Resolved dir_base path does not exist: {dir_base}")
            raise FileNotFoundError(f"Resolved dir_base path does not exist: {dir_base}")

        # Check if dir_std_base exists
        if not Path(dir_std_base).exists():
            logging.error(f"Resolved dir_base path does not exist: {dir_std_base}")
            raise FileNotFoundError(f"Resolved dir_base path does not exist: {dir_std_base}")

        # Optional prediction config values
        name_tfrm_config     = pred_cfg.get("name_tfrm_config", None)
        path_tfrm_script     = pred_cfg.get("path_tfrm_script", None)
        conda_env            = pred_cfg.get("conda_env", None)
        algo_response_vars = pred_cfg.get("algo_response_vars", [])
        algo_type = pred_cfg.get("algo_type", [])
        mapie_alpha = pred_cfg.get("MAPIE_alpha", None)
        uncn_bnd_pred = pred_cfg.get("uncn_bnd_pred", False)
        path_gpkg_pred = pred_cfg.get('path_gpkg_pred',None)
        pred_gpkg_lyr = pred_cfg.get('pred_gpkg_lyr', None)
        pred_gpkg_id_col = pred_cfg.get('pred_gpkg_id_col',None)
        path_crosswalk_ids = pred_cfg.get('path_crosswalk_ids',None)
        crosswalk_target_col = pred_cfg.get('crosswalk_target_col',None) # Oftentimes 'divide_id' for hydrofabric applications
        overwrite_sql = pred_cfg.get('overwrite_sql', False)
        # Prediction-time featureSource: independent of the training-time
        # value in the attribute/prep config. Predictions may legitimately be
        # made at a different spatial scale than training was performed on
        # (e.g. trained on USGS gage-basin-aggregated attributes, predicting
        # at HUC10/HUC14/divide scale), so rafts_pred_algo.py must not assume
        # the two are the same. None here means "not explicitly set" --
        # callers fall back to the training-time value only if this is unset.
        featureSource = pred_cfg.get('featureSource', None)
        path_hf_finl_gpkg = pred_cfg.get('path_hf_finl_gpkg', None)
        layr_hf_finl_gpkg = pred_cfg.get('layr_hf_finl_gpkg',None)
        algo_select = pred_cfg.get('algo_select', None)
        donor_map_states = pred_cfg.get('donor_map_states', None)

        # Compile dictionary
        self.pred_cfg_dict = {
            'algo_response_vars': algo_response_vars,
            'algo_type': algo_type,
            'datasets': datasets,
            'path_meta': path_meta,
            'write_type': write_type,
            'ds_type': ds_type,
            'dir_std_base': Path(dir_std_base),
            'dir_base': Path(dir_base),
            'home_dir': home_dir,
            'name_attr_config': name_attr_config,
            'name_algo_config': name_algo_config,
            'name_tfrm_config': name_tfrm_config,
            'path_tfrm_script': path_tfrm_script,
            'conda_env': conda_env,
            'pred_file_comid_colname': pred_file_comid_colname,
            'mapie_alpha': mapie_alpha,
            'uncn_bnd_pred': uncn_bnd_pred,
            'path_pred_config': self.path_pred_config,
            'path_gpkg_pred':path_gpkg_pred,
            'pred_gpkg_lyr':pred_gpkg_lyr,
            'pred_gpkg_id_col':pred_gpkg_id_col,
            'path_crosswalk_ids':path_crosswalk_ids,
            'crosswalk_target_col':crosswalk_target_col, # target hydrofabric identifier
            'path_hf_finl_gpkg': path_hf_finl_gpkg,
            'layr_hf_finl_gpkg':layr_hf_finl_gpkg,
            'overwrite_sql': overwrite_sql,
            'algo_select': algo_select,
            'featureSource': featureSource,
            'donor_map_states': donor_map_states,
        }

def _make_home_dir(home_dir_read:str|os.PathLike=[])-> os.PathLike:
    """Make the home directory based on what is passed or use default

    :param home_dir_read: A desired home directory, defaults to None which means the system default will be used
    :type home_dir_read: str | os.PathLike, optional
    :return: home directory path
    :rtype: os.PathLike
    """
    home_dir = str(Path.home()) # Initialize home_dir to a default value (system home)
    if len(home_dir_read) == 0:
        # home_dir is already set to Path.home()
        pass
    elif home_dir_read[0] is None:
        # home_dir is already set to Path.home()
        pass
    elif "~" in str(home_dir_read):
        if isinstance(home_dir_read, list) and home_dir_read and isinstance(home_dir_read[0], (str, Path)):
             # Assuming we mean to expand the passed value if it contains '~'
             home_dir = home_dir_read[0]
        home_dir = Path(home_dir).expanduser()
    elif not Path(home_dir_read[0]).exists():
        logging.warning(f"The user-defined home directory path {home_dir_read[0]} " \
            f"inside attribute config file does not exist. Using system default {Path.home()}")
        # home_dir remains set to the system default
        pass
    else:
        home_dir = home_dir_read[0]
    home_dir = Path(home_dir).expanduser()
    return home_dir

def _define_home_dir(attr_config:dict) -> os.PathLike:
    """Define the home directory of this system after parsing the attr config

    :param attr_config: The attribute config file object generated using rafts_algo.utils.AttrConfigAndVars
    :type attr_config: dict
    :return: The filepath to the home directory
    :rtype: os.PathLike
    """
    # Determine if home_dir. Either defined in attribute config file or assumed to be system default.
    home_dir_read = [v for x in attr_config['file_io'] for k, v in x.items() if 'home_dir' in k ]
    home_dir = _make_home_dir(home_dir_read)
    return home_dir        

def _check_attr_rm_dupes(attr_df:pd.DataFrame, 
                   uniq_cols:list = ['featureID','featureSource','data_source','attribute','value'],
                   sort_col:str = 'dl_timestamp',
                   ascending=True)-> pd.DataFrame:
    """Check if duplicate attributes exist in the dataset. If so, remove them.

    :param attr_df: The standard dataframe of attributes, location identifierws and their values
    :type attr_df: pd.DataFrame
    :param uniq_cols: The columns in attr_df to be tested for duplication, defaults to ['featureID','featureSource','data_source','attribute','value']
    :type uniq_cols: list, optional
    :param sort_col: The column name of the timestamps. Default 'dl_timestamp'
    :type sort_col: str, optional
    :param ascending: The argument to pass into sort_values on the `sort_col`. If ascending = False, the most recent timestamp will be kept, and the oldest with True. Default True.
    :type ascending: bool, optional
    :return: The dataframe with removed attributes
    :rtype: pd.DataFrame

    note:: When ascending = False, the most recent timestamp will be kept, and the oldest with True.
    """

    if attr_df[['featureID','attribute']].duplicated().any():
        logging.info("Duplicate attribute data exist. Attempting to remove using rafts_algo.utils._check_attr_rm_dupes().")
        attr_df = attr_df.sort_values(sort_col, ascending = ascending)
        attr_df = attr_df.drop_duplicates(subset=uniq_cols, keep='first')
    return attr_df

def rafts_read_attr_comid(dir_db_attrs:str | os.PathLike, comids_resp:list | Iterable = None, attrs_sel: str | Iterable = 'all',
                       _s3 = None,storage_options=None,read_type:str=['all','filename'][0],
                       reindex:bool=False)-> pd.DataFrame:
    """Read attribute data acquired using proc.attr.hydfab R package & subset to desired attributes

    :param dir_db_attrs: directory where attribute .parquet files live
    :type dir_db_attrs: str | os.PathLike
    :param comids_resp: Unique location ID (e.g.USGS COMID, hf_uid) values of interest 
    :type comids_resp: list | Iterable
    :param attrs_sel: desired attributes to select from the attributes .parquet files, defaults to 'all'
    :type attrs_sel: str | Iterable, optional
    :param _s3: future feature, defaults to None
    :type _s3: future feature, optional
    :param storage_options: future feature, defaults to None
    :type storage_options: future feature, optional
    :param read_type: should all parquet files be lazy-loaded, assign 'all'
     otherwise just files with comids_resp in the file name? assign 'filename'. Defaults to 'all',
     which should be fastest when querying multiple locations. For single locations, the 'filename'
     approach is fastest.
    :type read_type: str
    :param reindex: Should attribute dataframe be reindexed? Default False
    :type reindex: bool
    :return: dict of the following keys:
        - `attrs_sel`
        - `dir_db_attrs`
        - `dir_std_base`
        - `dir_base`
        - `datasets`
    :rtype: pd.DataFrame
    """
    # Changelog/contributions
    #  2025-04-01 Add logic to remove empty parquet files
    #  2025-05-19 refactor: remove _NA_ parquet files, udpate 'all' to dd.read_parquet, GL
    #  2025-06-10 fix: rm accidental elif in entry point for read_type; add partitioning schema, GL
    #  2025-06-13 feat: add timestamp datetime coercion, GL
    #. 2025-09-30 fix: 'filename' option uses only dd instead of pd inside dd
    #. 2026-05-03 refactor: allow no comids_resp, meaning all comids returned with data, GL
    if _s3:
        storage_options={"anon",True} # for public
        # TODO  Setup the s3fs filesystem that will be used, with xarray to open the parquet files
        #_s3 = s3fs.S3FileSystem(anon=True)

    # Parquet files sized 0 bytes cause fatal errors when trying to read. Remove them.
    files_empty = [file for file in Path(dir_db_attrs).rglob("*.parquet") if file.stat().st_size == 0]
    # The location identifer of _NA_ should not exist. Remove it.
    file_NA = [file for file in Path(dir_db_attrs).rglob("*.parquet") if '_NA_' in str(file)]
    files_rm = files_empty + file_NA
    if len(files_rm) > 0:
        for file in files_rm:
            os.remove(file)

    # Define a pyarrow schema of the attribute data
    partitioning = ds.partitioning(
        pa.schema([pa.field("featureID", pa.string()), pa.field('featureSource', pa.string()),
                pa.field("data_source", pa.string()), pa.field('dl_timestamp', pa.string()),
                pa.field("attribute", pa.string()), pa.field('value', pa.float64())]),
                flavor="hive")

    if comids_resp is None:
        # No specific location identifiers specified, so grab all.
        logging.info("comids_resp is None. Reading all available locations in subdirectory.")
        attr_ddf_subloc = dd.read_parquet(dir_db_attrs, storage_options=storage_options,
                                          engine='pyarrow', partitioning=partitioning)
    else:
        # ------------------- Subset based on comids of interest ------------------
        if read_type == 'all': # Considering all parquet files inside directory
            # Read attribute data acquired using proc.attr.hydfab R package
            comids_resp_str = [str(s) for s in comids_resp]
            all_attr_ddf = dd.read_parquet(dir_db_attrs, storage_options = storage_options,
                                            engine='pyarrow',partitioning=partitioning)
            attr_ddf_subloc = all_attr_ddf[all_attr_ddf['featureID'].isin(comids_resp_str)]
        elif read_type == 'filename': # Read based on comid being located in the parquet filename
            substrings = [f'_{sub}_' for sub in comids_resp]
            pattern = re.compile('|'.join(map(re.escape,substrings)))
            all_files = [file for file in Path(dir_db_attrs).iterdir() if file.is_file()]
            matching_files = [file for file in all_files if pattern.search(str(file))]

            if not matching_files: # NOTE: This was recommended by Gemini3.1Pro and may need further testing.
                logging.warning("No files matched the 'filename' pattern. Falling back to reading 'all' partitions.")
                comids_resp_str = [str(s) for s in comids_resp]
                all_attr_ddf = dd.read_parquet(dir_db_attrs, storage_options = storage_options,
                                                engine='pyarrow', partitioning=partitioning)
                attr_ddf_subloc = all_attr_ddf[all_attr_ddf['featureID'].isin(comids_resp_str)]
            else: # The historic approach
                # Read in all matching filenames and proceed
                attr_ddf_subloc = dd.read_parquet(matching_files,
                                                engine='pyarrow',
                                                partitioning=partitioning)

        else:
            # Initialize attr_ddf_sub
            attr_ddf_sub = None
            logging.error(f"Unrecognized read_type provided in rafts_read_attr_comid: {read_type}")
            raise ValueError(f"Unrecognized read_type provided in rafts_read_attr_comid: {read_type}")
        
        if attr_ddf_subloc.shape[0].compute() == 0:
            logging.warning(f'None of the provided featureIDs exist in {dir_db_attrs}: \
                        \n {", ".join(attrs_sel)} ')
    
    # ------------------- Subset based on attributes of interest ------------------
    if attrs_sel == 'all':
        attrs_sel_list = attr_ddf_subloc['attribute'].unique().compute().tolist()
    else:
        atrs_sel_list = list(attrs_sel)

    attr_ddf_sub = attr_ddf_subloc[attr_ddf_subloc['attribute'].isin(attrs_sel)]
    attr_df_sub = attr_ddf_sub.compute() # This takes a while querying many files.

    if attr_df_sub.shape[0] == 0:
        logging.warning(f'The provided attributes do not exist with the retrieved featureIDs : \
                        \n {",".join(atrs_sel_list)}')
        
    # ------------------- INTERSECTION: Keep only locations with ALL attributes -------------------
    if comids_resp is None and len(attr_df_sub) > 0:
        required_attr_count = len(atrs_sel_list)
        
        # Count unique attributes per featureID
        attr_counts = attr_df_sub.groupby('featureID')['attribute'].nunique()
        
        # Filter for featureIDs that have exactly the required number of attributes
        valid_feature_ids = attr_counts[attr_counts == required_attr_count].index
        
        dropped_count = len(attr_counts) - len(valid_feature_ids)
        if dropped_count > 0:
            logging.info(f"Dropped {dropped_count} locations that did not contain all {required_attr_count} requested attributes.")
            
        attr_df_sub = attr_df_sub[attr_df_sub['featureID'].isin(valid_feature_ids)].copy()    
    # ------------------- Remove any duplicates & run checks -------------------
    attr_df_sub = _check_attr_rm_dupes(attr_df=attr_df_sub)

    # Run check that all variables are present across all basins
    dict_rslt = _check_attributes_exist(attr_df_sub,atrs_sel_list)
    attr_df_sub, attrs_sel_ser = dict_rslt['df_attr'], dict_rslt['attrs_sel']
    

    if not pd.api.types.is_float_dtype(attr_df_sub['value']):
        logging.warning("Forcing all attribute values to be float")
        attr_df_sub['value'] = np.float64(attr_df_sub['value'])

    if attr_df_sub['value'].isna().any():
        logging.warning('The attribute dataset contains unexpected NA values, \
                      which may be problematic for some algo training/testing. \
                      \nConsider reprocessing the attribute grabber (proc.attr.hydfab R package)')
    

    if reindex:
        attr_df_sub = attr_df_sub.reindex()

    # Coerce the timestamp column to datetime
    if 'dl_timestamp' in attr_df_sub.columns:
        attr_df_sub['dl_timestamp'] = pd.to_datetime(attr_df_sub['dl_timestamp'], errors='coerce')

    # Drop the duplicates pertaining to featureID,featureSource, attribute, and value:

    return attr_df_sub

def _check_attributes_exist(df_attr: pd.DataFrame, attrs_sel:pd.Series | Iterable) -> Dict[str, pd.DataFrame | pd.Series]:
    """ Checks if any COMIDs have different numbers of attributes. It's expected that they all have the same attributes.

    :param df_attr: The attribute data, as generated in :func:`rafts_read_attr_comid()`
    :type df_attr: pd.DataFrame
    :param attrs_sel: the names of the attributes
    :type attrs_sel: pd.Series | Iterable
    :return: the same objects df_attr, and attrs_sel, but attrs_sel is ensured to be a pd.Series
    :rtype: Dict[pd.DataFrame, pd.Series]
    :seealso: :func:`rafts_read_attr_comid()`

    """
    if not isinstance(attrs_sel,pd.Series):
        # Convert to a series for convenience of pd.Series.isin()
        attrs_sel = pd.Series(pd.Series(attrs_sel).unique())
    else:
        attrs_sel = pd.Series(attrs_sel.unique())

    # Run check that all attributes are present for all basins
    if df_attr.groupby('featureID')['attribute'].count().nunique() != 1:
        # multiple combos of comid/attrs exist. Find them and warn about it.
        counts = df_attr.groupby('featureID')['attribute'].count()
        vec_missing = counts != len(attrs_sel)

        if isinstance(vec_missing, pd.Series):
            bad_comids = counts.index[vec_missing].tolist()
        else: # Fallback in case vec_missing is a single boolean
            bad_comids = counts.index.tolist() if vec_missing else []
        # vec_missing = df_attr.groupby('featureID')['attribute'].count() != len(attrs_sel)
        # bad_comids = vec_missing.index.values[vec_missing]
        msg_tot_loc = f"    TOTAL unique locations with missing attributes: {len(bad_comids)} of {df_attr['featureID'].nunique()} total unique locations"
        logging.warning(msg_tot_loc)
        df_attr_sub_missing = df_attr[df_attr['featureID'].isin(bad_comids)]
    
        if isinstance(attrs_sel,list):
            missing_attrs = [attr for attr in attrs_sel if attr not in set(df_attr_sub_missing['attribute'])]
            missing_attrs = pd.DataFrame({'attribute':missing_attrs})
        else:
            missing_attrs = attrs_sel[~attrs_sel.isin(df_attr_sub_missing['attribute'])]
        msg_tot_miss = f"    TOTAL MISSING ATTRS: {len(missing_attrs)} of {len(attrs_sel)}"
        logging.warning(msg_tot_miss)
        str_missing = '\n    '.join(missing_attrs.values)

        warn_msg_missing_attrs = "\
        \n Not all featureID groupings (i.e. COMID groups) contain the same number of catchment attributes. \
        \n This could be problematic for model training and/or prediction. \
        \n Consider running attribute grabber with proc.attr.hydfab."
        warn_msg2 = "\nMissing attributes include: \n    " + str_missing
        if df_attr['featureID'].nunique() > 0:
            warn_msg_3 = "\n COMIDs with missing attributes include: \n" + ', '.join(bad_comids)
        else: # An empty dataframe was passed to this function, so we can't say what location IDs are missing
            warn_msg_3 = ''
        logging.warning(warn_msg_missing_attrs + warn_msg2 + warn_msg_3)
    return {'df_attr': df_attr, 'attrs_sel': attrs_sel}


def _id_attrs_sel_wrap(attr_cfig: AttrConfigAndVars,
                    path_cfig: str | os.PathLike = None,
                    name_attr_csv: str = None,
                    colname_attr_csv: str = None) -> list:
    """Get attributes of interest from a csv file with column name, or the attribute config object

    :param attr_cfig: The attribute config file object generated using rafts_algo.utils.AttrConfigAndVars
    :type attr_cfig: AttrConfigAndVars
    :param path_cfig: Optional path to a file, that also lives in the same directory as the `name_attr_csv`, defaults to None
    :type path_cfig: str | os.PathLike
    :param name_attr_csv: The name of the csv file containing the attribute listing of interest, defaults to None
    :type name_attr_csv: str, optional
    :param colname_attr_csv: The column name inside the csv file containing the attributes of interest, defaults to None
    :type colname_attr_csv: str, optional
    :return: list of all attributes of interest, likely to use for training/prediction
    :rtype: list

    """
    # Changelog / contributions
    #  2024-04-15 Force into ensure uniqueness attributes, GL
    if name_attr_csv:
        path_attr_csv = build_cfig_path(path_cfig,name_attr_csv)
        attrs_sel = pd.read_csv(path_attr_csv)[colname_attr_csv].tolist()
    else:
        attrs_sel = attr_cfig.attrs_cfg_dict.get('attrs_sel', None)
    attrs_sel = list(set(attrs_sel))
    return attrs_sel

def _find_feat_srce_id(dat_resp: Optional[xr.core.dataset.Dataset] = None,
                       col_schema: Optional[list[Dict]]=None,
                       attr_config: Optional[Dict] = None) -> List[str]:
    """ Try grabbing :mod:`rafts_prep` standardized dataset attributes &/or config file.

    :param dat_resp: The standardized dataset generated by :mod:`rafts_prep`, defaults to None
    :type dat_resp: Optional[xr.core.dataset.Dataset], optional
    :param col_schema: Column schema mapping out the featureID f-string format
      (e.g. "USGS-{gage_id}") and featureSource. One f-string formatting completed,
       serves as argument used in nhdplusTools::get_nldi_features() in proc.attr.hydfab 
       R package, for example. Extracted from the attribute config file. defaults to None
    :type col_schema: Optional[list[Dict]], optional
    :param attr_config: The attribute config file object generated using
      rafts_algo.utils.AttrConfigAndVars, defaults to None
    :type attr_config: Optional[Dict], optional
    :raises ValueError: featureSource could not be identified from the provided
    :raises ValueError: _description_
    :return: The featureSource and f-string formatted featureID, e.g. ['nwissite','USGS-{gage_id}']
    :rtype: List[str]

    note:: Standardized dataset attributes preferred in cases of processing multiple
      datasets & attributes differ by dataset). Otherwise, fallback on config file.
        At least one argument must be provided.

    # 2025-06-13 refactor: simplify to allow passing just the col_schema section of the attribute config file, GL
    """

    featureSource = None
    try: # dataset attributes first
        featureSource = dat_resp.attrs.get('featureSource', None)
    except (KeyError, StopIteration,AttributeError): # config file second
        pass
    if attr_config is not None:
        col_schema = attr_config.get('col_schema', None)

    if featureSource is None:
        try: 
            featureSource = next(x['featureSource'] for x in col_schema if 'featureSource' in x)
        except:
            pass
    
    if not featureSource:
        logging.error('The featureSource could not be found. Ensure it is present in the col_schema section of the attribute config file.')
        raise ValueError(f'The featureSource could not be found. Ensure it is present in the col_schema section of the attribute config file.')
    # Attempt to grab featureID from dataset attributes, fallback to the config file
    featureID = None
    try: # dataset attributes first
        featureID = dat_resp.attrs.get('featureID', None)
    except (KeyError, StopIteration,AttributeError): # config file second
        pass
    if featureID is None:
        try:
            featureID = next(x['featureID'] for x in col_schema if 'featureID' in x)
        except:
            pass
    if not featureID:
        logging.error('The featureID could not be found. Ensure it is present in the col_schema section of the attribute config file.')
        raise ValueError(f'The featureID could not be found. Ensure it is present in the col_schema section of the attribute config file.')
        # TODO need to map gage_id to location identifier in attribute data!

    return [featureSource, featureID]
    
def rafts_retr_nhdp_comids_geom(featureSource:str,featureID:str,gage_ids: Iterable[str] 
                             ) -> gpd.geodataframe.GeoDataFrame:    
    """Retrieve response variable's comids & point geom, querying the shortest distance in the flowline

    :param featureSource: the datasource for featureID from the R function :mod:`nhdplusTools` :func:`get_nldi_features()`, e.g. 'nwissite'
    :type featureSource: str
    :param featureID: The conversion format of `gage_ids` into a recognizable string for :mod:`nhdplusTools`, which is an f-string configured conversion of `gage_id` e.g. `'USGS-{gage_id}'`. Expected to contain the string `"{gage_id}"`
    :type featureID: str
    :param gage_ids: The location identifiers compatible with the format specified in `featureID`
    :type gage_ids: Iterable[str]
    :raises warnings.warn: In case number of retrieved comids does not match total requested gage ids
    :return: The COMIDs, gage_id, & point geometry corresponding to the provided location identifiers, `gage_ids`
    :rtype: GeoDataFrame
    """
    # Changelog:
    #     2024-12-01 refactor: return GeoDataFrame with coordinates instead of a list of just comids, GL
    #     2025-03-07 add gage_id to return geodataframe

    nldi = nhd.NLDI()
    
    gageids_miss = []
    comids_resp = []
    geom_pts = []
    feature_id = []
    for gage_id in gage_ids:
        feature_id.append(featureID.format(gage_id=gage_id))
        try:
            upstr_flowline = nldi.navigate_byid(
                fsource=featureSource,
                fid=featureID.format(gage_id=gage_id),
                navigation='upstreamMain',
                source='flowlines',
                distance=1
            ).loc[0]
            geom_pts.append(Point(upstr_flowline['geometry'].coords[0]))
            comid = upstr_flowline['nhdplus_comid']
            comids_resp.append(comid)
        except Exception as e:
            logging.info(f"Error processing gage_id {gage_id}: {e}")
            # Handle the error (e.g., log it, append None, or any other fallback mechanism)

            # TODO Attempt a different approach for retrieving comid:
            gageids_miss.append(gage_id)
            geom_pts.append(np.nan)
            comids_resp.append(np.nan)  # Appending NA for failed gage_id, or handle differently as needed

    # if len(comids_resp) != len(gage_ids) or comids_resp.count(None) > 0: # May not be an important check
    #     raise warnings.warn("The total number of retrieved comids does not match \
    #                   total number of provided gage_ids",UserWarning)

    gdf_comid = gpd.GeoDataFrame(pd.DataFrame({ 'comid': comids_resp,
                                               'gage_id': gage_ids}),
                                            geometry=geom_pts,crs=4326 
                                )

    return gdf_comid

def build_cfig_path(path_known_config:str | os.PathLike, path_or_name_cfig:str | os.PathLike) -> os.PathLike | None:
    """Build the expected configuration file path within the RAFTS framework

    :param path_known_config: path of the known configuration sent 
    :type path_known_config: str | os.PathLike
    :param path_or_name_cfig: Path or name of configuration file. If only name provided, it's assumed it resides in same directory as `path_known_config`
    :type path_or_name_cfig: str | os.PathLike
    :raises FileNotFoundError: The provided `path_known_config` does not exist
    :raises FileNotFoundError: The desired configuration file does not exist
    :return: The path to another relevant configuration file used for a different step in RAFTS processing
    :rtype: os.PathLike | None
    """
    dir_parent_cfig = Path(path_known_config).parent
    if not dir_parent_cfig.exists():
        logging.error(f"The provided 'known' configuration file does not exist: \n{path_known_config}")
        raise FileNotFoundError(f"The provided 'known' configuration file does not exist: \n{path_known_config}")
    if path_or_name_cfig: # Only perform if path_or_name_cfig not None
        path_cfig = Path(dir_parent_cfig/Path(path_or_name_cfig))
        if not path_cfig.exists():
            path_cfig = Path(path_or_name_cfig)
            if not path_cfig.exists():
                logging.error(f'The following configuration file could not be found: \n{path_or_name_cfig}')
                raise FileNotFoundError(f'The following configuration file could not be found: \n{path_or_name_cfig}')
    else:
        logging.warning("The configuration file may not have specified the path or file name.")
        path_cfig = None
    return path_cfig

def load_validated_config(path_config: str | os.PathLike, model_cls: type) -> Any:
    """Read a YAML config file and validate it against a Pydantic model in one step.

    Several flow scripts (e.g. rafts_pred_algo.py, rafts_pair_donors.py,
    rafts_regn_params_gpkg.py, rafts_map_pred_hfatl.py, rafts_proc_algo_pool.py,
    build_nexus_crosswalk.py, rafts_agg_nexus_hfatl.py) each open, parse, and
    Pydantic-validate the same config file independently; this consolidates
    that repeated three-line pattern into a single call.

    :param path_config: Path to the YAML configuration file.
    :type path_config: str | os.PathLike
    :param model_cls: The Pydantic BaseModel subclass to validate the parsed
        YAML against (e.g. rafts_algo.schemas.pydantic_schemas.PredConfig).
    :type model_cls: type
    :return: The validated Pydantic model instance.
    :rtype: Any
    """
    with open(path_config, 'r') as f:
        raw_cfg = yaml.safe_load(f)
    return model_cls(**raw_cfg)

def build_pred_locs_path(path_meta_template: str | os.PathLike,dir_std_base: str | os.PathLike,
    ds: str,ds_type: str,write_type: str='parquet') -> Path:
    """
    Build the full path to the prediction metadata location using formatting template.

    :param path_meta_template: f-string template for the path, e.g. "{dir_std_base}/{ds}/nldi_feat_{ds}_{ds_type}.parquet"
    :param dir_std_base: Base directory for standardized data
    :param ds: Dataset name
    :param ds_type: Dataset type, e.g., 'prediction'
    :param write_type: File extension/type, e.g., 'csv' or 'parquet'
    :return: Full formatted Path to prediction metadata file
    """
    path_str = path_meta_template.format(
        dir_std_base=dir_std_base,
        ds=ds,
        ds_type=ds_type,
        write_type=write_type
    )
    return Path(path_str)

def rafts_save_algo_dir_struct(dir_base: str | os.PathLike, region: Optional[str] = None,
                                scheme: Optional[str] = None) -> dict:
    """Generate a standard file saving directory structure

    :param dir_base: The base directory for saving output
    :type dir_base: str | os.PathLike
    :param region: Region identifier to nest output under, or None (default) for today's
        unregioned layout. Ignored unless `scheme` is also given.
    :type region: Optional[str]
    :param scheme: Region scheme name (e.g. 'vpu'|'huc2'|'states'|'custom'), used only to
        build the nesting path's directory name. None (default) reproduces today's layout
        exactly, regardless of `region`.
    :type scheme: Optional[str]
    :raises ValueError: If the base directory does not exist
    :return: Full paths to the `output`, `trained_algorithms`,
     `analysis` and `data_visualization` directories
    :rtype: dict
    """

    if not Path(dir_base).exists():
        logging.error(f"The provided dir_base does not exist. \
                         \n Double check the config file to make sure \
                         \n an existing directory is provided. dir_base= \
                         \n{dir_base}")
        raise ValueError(f"The provided dir_base does not exist. \
                         \n Double check the config file to make sure \
                         \n an existing directory is provided. dir_base= \
                         \n{dir_base}")

    # Define the standardized directory structure for algorithm output
    # base save directory, inside dir_base
    dir_out = Path(dir_base) / 'output'
    if scheme and region:
        dir_out = dir_out / 'regions' / scheme / region
    dir_out.mkdir(exist_ok=True, parents=True)

    # The trained algorithm directory
    dir_out_alg_base = Path(dir_out / Path('trained_algorithms'))
    dir_out_alg_base.mkdir(exist_ok=True)

    # TODO consider compatibility with std_pred_path
    dir_preds_base = Path(dir_out/Path('algorithm_predictions'))
    dir_preds_base.mkdir(exist_ok=True)

    # The analysis directory
    dir_out_anlys_base = Path(dir_out/Path("analysis"))
    dir_out_anlys_base.mkdir(exist_ok=True)

    # The data visualization directory
    dir_out_viz_base = Path(dir_out/Path("data_visualizations"))
    # TODO insert dir that Lauren creates here

    out_dirs = {'dir_out': dir_out,
                'dir_out_alg_base': dir_out_alg_base,
                'dir_out_preds_base' : dir_preds_base,
                'dir_out_anlys_base' : dir_out_anlys_base,
                'dir_out_viz_base' : dir_out_viz_base}

    return out_dirs

def _std_rafts_prep_ds_companion_gpkg_path(path_rafts_prep:str|os.PathLike)->os.PathLike:
    """Create the standardized gpkg path for coordinate data & id mapping
      corresponding to the standardized input data

    :param path_rafts_prep: Path used for the standardized dataset created using rafts_prep.proc_eval_metrics.proc_col_schema()
    :type path_rafts_prep: str | os.PathLike
    :return: Path storing the coordinates and id-mapping of each location of interest
    :rtype: os.PathLike
    """
    path_rafts_prep =Path(path_rafts_prep)
    sub_name_loc = path_rafts_prep.with_suffix('')
    new_name_loc = str(sub_name_loc.name) + '_loc'
    new_name_gpkg = str(Path(new_name_loc).with_suffix('.gpkg'))
    path_gpkg_rafts_prep = path_rafts_prep.with_name(new_name_gpkg)
    return path_gpkg_rafts_prep

def _std_rafts_prep_ds_paths(dir_std_base: str|os.PathLike,ds:str,mtch_str='*.nc') -> list:
    """The standard .nc paths for standardized dataset created using rafts_prep.proc_eval_metrics.proc_col_schema()

    :param dir_std_base:  The directory containing the standardized dataset generated from `rafts_prep`
    :type dir_std_base: str | os.PathLike
    :param ds:  a string that's unique to the dataset of interest
    :type ds: str
    :param mtch_str: the desired matching string describing datasets of interests, defaults to '*.nc'
    :type mtch_str: str, optional
    :return: list of each filepath to a dataset
    :rtype: list
    """
    ls_ds_paths = [x for x in Path(dir_std_base/Path(ds)).glob(mtch_str) if x.is_file()]
    return ls_ds_paths

def _open_response_data_rafts(dir_std_base: str | os.PathLike, ds:str, mtch_str:str='*.nc') -> xr.Dataset:
    """Read in standardized dataset generated from :mod:`rafts_prep`

    :param dir_std_base: The directory containing the standardized dataset generated from `rafts_prep`
    :type dir_std_base: str | os.PathLike
    :param ds: a string that represents the dataset of interest
    There should be a netcdf .nc or zarr .zarr file containing matches to this string
    :type ds: str
    :raises ValueError: The directory where the dataset file should live does not exist.
    :raises ValueError: Could not successfully read in the dataset `ds` as a .nc or .zarr
    :return: The hierarchical dataset as an xarray, as generated by :mod:`rafts_prep`
    :rtype: xr.Dataset
    """
    # Implement a check to ensure each dataset directory exists
    if not Path(dir_std_base).exists:
        logging.error(f'The dir_std_base directory does not exist. Double check dir_std_base: \
                         \n{dir_std_base}')
        raise ValueError(f'The dir_std_base directory does not exist. Double check dir_std_base: \
                         \n{dir_std_base}')
    
    path_nc = _std_rafts_prep_ds_paths(dir_std_base=dir_std_base,ds=ds,mtch_str=mtch_str)
    #path_nc = [x for x in Path(dir_std_base/Path(ds)).glob("*.nc") if x.is_file()]
    if len(path_nc) > 1:
        error_str = f"The following directory contains too many .nc files: {path_nc}"
        logging.error(error_str)
        raise ValueError(error_str)

    try:
        dat_resp = xr.open_dataset(path_nc[0], engine='netcdf4')
    except:
        path_zarr = [x for x in Path(dir_std_base/Path(ds)).glob("*.zarr")]
        try:
            dat_resp = xr.open_dataset(path_zarr[0],engine='zarr')
        except:
            logging.error(f"Could not identify an approach to read in dataset via {path_nc} nor {path_zarr}")
            raise ValueError(f"Could not identify an approach to read in dataset via {path_nc} nor {path_zarr}")
    return dat_resp

def std_algo_path(dir_out_alg_ds:str | os.PathLike, algo: str, metric: str, dataset_id: str) -> str:
    """Standardize the algorithm save path
    :param dir_out_alg_ds:  Directory where algorithm's output stored.
    :type dir_out_alg_ds: str | os.PathLike
    :param algo: The type of algorithm
    :type algo: str
    :param metric:  The metric or hydrologic signature identifier of interest
    :type metric: str
    :param dataset_id: Unique identifier/descriptor of the dataset of interest
    :type dataset_id: str
    :return: full save path for joblib object
    :rtype: str
    """
    Path(dir_out_alg_ds).mkdir(exist_ok=True,parents=True)
    basename_alg_ds_metr = f'algo_{algo}_{metric}__{dataset_id}'
    path_algo = Path(dir_out_alg_ds) / Path(basename_alg_ds_metr + '.joblib')
    return path_algo

def std_pred_path(dir_out: str | os.PathLike, algo: str, metric: str, dataset_id: str
                  ) -> Path:
    """Standardize the prediction results save path

    :param dir_out: The base directory for saving output
    :type dir_out: str | os.PathLike
    :param algo: The type of algorithm
    :type algo: str
    :param metric: The metric or hydrologic signature identifier of interest
    :type metric: str
    :param dataset_id: Unique identifier/descriptor of the dataset of interest
    :type dataset_id: str
    :return: full save path for parquet dataframe object of results
    :rtype: str
    """
    # TODO consider refactoring this to pass in dir_out_preds_base instead
    dir_preds_base = Path(Path(dir_out)/Path('algorithm_predictions'))
    dir_preds_ds = Path(dir_preds_base/Path(dataset_id))
    dir_preds_ds.mkdir(exist_ok=True,parents=True)
    basename_pred_alg_ds_metr = f"pred_{algo}_{metric}__{dataset_id}.parquet"
    path_pred_rslt = Path(dir_preds_ds)/Path(basename_pred_alg_ds_metr)
    return path_pred_rslt

def discover_dynamic_algos(search_dir: Path, base_algos: list, metric: str, 
                           dataset_id: str, file_prefix: str, file_extension: str) -> list:
    """
    Dynamically scans a directory to find algorithm variations (e.g., different cluster sizes).
    
    :param search_dir: The directory to search within.
    :type search_dir: Path
    :param base_algos: The base algorithm strings from the config (e.g., ['kmeans', 'rf']).
    :type base_algos: list
    :param metric: The response variable.
    :type metric: str
    :param dataset_id: The unique dataset identifier.
    :type dataset_id: str
    :param file_prefix: The prefix of the saved files (e.g., ``'algo_'`` or ``'pred_'``).
    :type file_prefix: str
    :param file_extension: The file extension (e.g., '.joblib' or '.parquet').
    :type file_extension: str
    :return: A list of the exact algorithm strings discovered (e.g., ['kmeans_k5', 'kmeans_k8']).
    :rtype: list
    """
    # Changelog/contributions
    #     2026-05-11 Autogenerated using Gemini3Pro
    dynamic_algos = []
    
    if not search_dir.exists():
        logging.warning(f"Search directory does not exist: {search_dir}")
        return dynamic_algos

    for base_algo in base_algos:
        search_pattern = f"{file_prefix}{base_algo}*_{metric}__{dataset_id}{file_extension}"
        matched_files = list(search_dir.glob(search_pattern))
        
        for match in matched_files:
            suffix = f"_{metric}__{dataset_id}{file_extension}"
            if match.name.startswith(file_prefix) and match.name.endswith(suffix):
                # Extract the exact algorithm name by stripping the known prefix and suffix
                exact_algo_name = match.name[len(file_prefix):-len(suffix)]
                if exact_algo_name not in dynamic_algos:
                    dynamic_algos.append(exact_algo_name)
                    
    return dynamic_algos

def std_eval_metrs_path(dir_out_viz_base: str|os.PathLike,
                      ds:str, metr:str
                      ) -> Path:
    """Standardize the filepath for saving model evaluation metrics table

    :param dir_out_viz_base: The base output directory
    :type dir_out_viz_base: str | os.PathLike
    :param ds: The dataset name
    :type ds: str
    :param metric: The metric or hydrologic signature identifier of interest
    :type metric: str
    :return: The model metrics filepath
    :rtype: pathlib.PosixPath
    """
    path_eval_metr = Path(f"{dir_out_viz_base}/{ds}/algo_eval_{ds}_{metr}.csv")
    path_eval_metr.parent.mkdir(parents=True,exist_ok=True)
    return path_eval_metr


def std_test_pred_obs_path(dir_out_anlys_base:str|os.PathLike,ds:str, metr:str
                      )->Path:
    """Generate the standardized path for saving the predicted & observed metric/coordinates from testing

    :param dir_out_anlys_base: Base analysis directory
    :type dir_out_anlys_base: str | os.PathLike
    :param ds: dataset name
    :type ds: str
    :param metr: metric/response variable of interest
    :type metr: str
    :return: save path to the pred_obs_{ds}_{metr}.csv file
    :rtype: pathlib.PosixPath
    """
    # Create the path for saving the predicted and observed metric/coordinates from testing
    path_pred_obs = Path(f"{dir_out_anlys_base}/{ds}/pred_obs_{ds}_{metr}.csv")
    path_pred_obs.parent.mkdir(exist_ok=True,parents=True)
    return path_pred_obs

def _read_pred_comid(path_pred_locs: str | os.PathLike, comid_pred_col:str ) -> list[str]:
    """Read the comids from a prediction file formatted as .csv

    :param path_pred_locs: The path to prediction data location, containing the comid
    :type path_pred_locs: str | os.PathLike
    :param comid_pred_col: The column name corresponding to the comid inside the prediction location dataset
    :type comid_pred_col: str
    :raises ValueError: Could not read the location data file and/or subselect the comid column
    :raises ValueError: File extension of location data file not recognized
    :return: list of comids
    :rtype: list[str]
    """
    # Changelog/contributions
    # 2025-03-30 Add in drop_duplicates(), GL
    # 2026-05-01 Add clean_hfatlas_columns for hfATLAS applications, GL
    if not Path(path_pred_locs).exists():
        logging.error(f"The path to prediction location data could not be found: \n{path_pred_locs} ")
        raise FileNotFoundError(f"The path to prediction location data could not be found: \n{path_pred_locs} ")
    if '.csv' in Path(path_pred_locs).suffix:
        try:
            comids_pred = pd.read_csv(path_pred_locs)[comid_pred_col].drop_duplicates().values            
        except:
            logging.error(f"Could not successfully read in {path_pred_locs} & select col {comid_pred_col}")
            raise ValueError(f"Could not successfully read in {path_pred_locs} & select col {comid_pred_col}")
    elif '.parquet' in Path(path_pred_locs).suffix:
        try:
            df_pred = pd.read_parquet(path_pred_locs)
            is_tuple_format = df_pred.columns.str.match(r"^\('.*', '.*'\)$")
            if is_tuple_format.all(): # Perform column name cleaning 
                df_pred = clean_hfatlas_columns(df_pred)
            comids_pred = df_pred[comid_pred_col].drop_duplicates().values
        except:
            logging.error(f"Could not successfully read in {path_pred_locs} & select col {comid_pred_col}")
            raise ValueError(f"Could not successfully read in {path_pred_locs} & select col {comid_pred_col}")
    elif Path(path_pred_locs).is_dir(): 
        try: # If a directory is provided, this assumes it only contains .parquet files
            df_pred = pd.read_parquet(path_pred_locs)
            is_tuple_format = df_pred.columns.str.match(r"^\('.*', '.*'\)$")
            if is_tuple_format.all(): # Perform column name cleaning 
                df_pred = clean_hfatlas_columns(df_pred)
            comids_pred = df_pred[comid_pred_col].drop_duplicates().values
        except:
            logging.error(f"Could not successfully read in {path_pred_locs} & select col {comid_pred_col}")
            raise ValueError(f"Could not successfully read in {path_pred_locs} & select col {comid_pred_col}")
    else:
        logging.error(f"NEED TO ADD CAPABILITY THAT HANDLES {Path(path_pred_locs).suffix} file extensions")
        raise ValueError(f"NEED TO ADD CAPABILITY THAT HANDLES {Path(path_pred_locs).suffix} file extensions")
    comids_pred = [str(x) for x in comids_pred]
    return comids_pred


def find_common_comid(dict_gdf_comids:Dict[str,gpd.GeoDataFrame], column='featureID')->list:
    """Given a collection of multiple datasets, find the shared comids

    :param dict_gdf_comids: a dictionary of multiple datasets,
      each containing a geodataframe of comids as generated by
      :func:`rafts_retr_nhdp_comids_geom`
    :type dict_gdf_comids: dict[str, geopandas.GeoDataFrame]
    :param column: The geodataframe column name for the comid, defaults to 'featureID'
    :type column: str, optional
    :seealso: :func:`split_train_test_comid_wrap`
    :seealso: :func:`rafts_retr_nhdp_comids_geom`
    :return: list of the shared comids
    :rtype: list
    """
    # Changelog/contributions
    # FY25 originally created, GL
    # 2025-05-19 change default column to 'featureID'

    common_comid = None
    for df in dict_gdf_comids.values():
        if common_comid is None:
            common_comid = set(df[column])
        else:
            common_comid &= set(df[column])

    common_comid = list(common_comid)
    return common_comid

def rafts_retr_nhdp_comids_geom_wrap(path_save_gpkg: str | os.PathLike,
                                  gage_ids: Iterable,
                                  featureSource: str = 'nwissite', 
                                  featureID: str = 'USGS-{gage_id}') -> gpd.GeoDataFrame:
    """Read or generate a geodataframe that queries NHDplus for comid and coordinate.
    
    :param path_save_gpkg: filepath where data are saved. This limits the number of hits to the NHDplus database
    :type path_save_gpkg: str | os.PathLike
    :param gage_ids: The identifiers of interest, e.g. the USGS gage id numbers
    :type gage_ids: Iterable
    :param featureSource: the datasource for featureID from the R function 
        :mod:`nhdplusTools` :func:`get_nldi_features()`,defaults to 'nwissite'
    :type featureSource: str
    :param featureID: The conversion format of `gage_ids` into a recognizable string for
        :mod:`nhdplusTools`, which is an f-string configured conversion of `gage_id`
        defaults to 'USGS-{gage_id}'. Expected to contain the string `"{gage_id}"`
    :type featureID: str
    :return: Geodataframe with the columns 'comid', 'geometry', 'gage_id'
    :rtype: gpd.GeoDataFrame
    :seealso: :func:`combine_resp_gdf_comid_wrap` A wrapper function that calls this function
    :seealso: :func:`_std_rafts_prep_ds_companion_gpkg_path` The standardized path to use for path_save_gpkg
    :seealso: :mod:`proc.attr.hydfab`:func:`rafts_retr_nhdp_comids_geom_wrap` The corresponding R function
    """
    # Changelog / Contributions:
    #     2024/2025 originally created?
    #     2026-05-27 refactor to allow skipping missing locs when not working with nwissite/comid data, Gemini3Pro
    
    path_save_gpkg = Path(path_save_gpkg)
    gage_ids_str = [str(g) for g in gage_ids]

    if path_save_gpkg.exists(): # Maybe we can skip the database connection!
        # Read the intermediate offline file
        gdf_comid_in = gpd.read_file(path_save_gpkg, layer='outlet')
        
        # Ensure safe string comparison
        if 'gage_id' in gdf_comid_in.columns:
            existing_gages = gdf_comid_in['gage_id'].astype(str).tolist()
        else:
            existing_gages = []

        cmmn_ids = np.intersect1d(existing_gages, gage_ids_str)
        
        # If all requested gages are in the file, OR if we are using an offline featureSource
        if len(cmmn_ids) == len(gage_ids_str) or featureSource not in ['nwissite', 'comid']:  # All gage_ids accounted for
            if len(cmmn_ids) != len(gage_ids_str):
                logging.warning(f"Offline GPKG is missing {len(gage_ids_str) - len(cmmn_ids)} requested gages. "
                                f"Bypassing NLDI API query because featureSource is '{featureSource}'.")
            return gdf_comid_in.copy()
            
        else:
            # Legacy NLDI Web API Logic - ONLY fetch the missing ones
            need_gage_ids = list(set(gage_ids_str) - set(existing_gages))
            logging.info(f"Fetching {len(need_gage_ids)} missing geometries from NLDI API...")
            
            gdf_new = rafts_retr_nhdp_comids_geom(featureSource=featureSource,
                                               featureID=featureID,
                                               gage_ids=need_gage_ids)
                                               
            # Combine the old good data with the new web data and save
            gdf_comid = pd.concat([gdf_comid_in, gdf_new], ignore_index=True)
            gdf_comid.to_file(path_save_gpkg, layer='outlet', driver='GPKG')
            return gdf_comid
            
    else:
        # File doesn't exist at all, query everything via Web API
        logging.info("Companion GPKG not found. Querying NLDI API for all geometries...")
        gdf_comid = rafts_retr_nhdp_comids_geom(featureSource=featureSource,
                                             featureID=featureID,
                                             gage_ids=gage_ids)
        gdf_comid.to_file(path_save_gpkg, layer='outlet', driver='GPKG')
        return gdf_comid

def _read_metadata(path_attr_config:str|os.PathLike, ds:str) -> pd.DataFrame:
    """Read the metadata file for the dataset of interest

    :param attr_config: The path to attribute config file 
    :type attr_config: str | os.PathLike
    :param ds: The unique dataset identifier
    :type ds: str
    :return: The metadata dataframe
    :rtype: pd.DataFrame
    """
    # Changelog:
    #     2025-05-19 originally created, GL

    attr_cfig = AttrConfigAndVars(path_attr_config)
    attr_cfig._read_attr_config()

    # Define directories/datasets from the attribute config file
    dir_std_base = attr_cfig.attrs_cfg_dict.get('dir_std_base')

    # Grab variables for building out the path to metadata (which contains comid-gage id mappings)
    ds_type = [x for x in attr_cfig.attr_config.get('file_io') if 'ds_type' in x][0]['ds_type']
    path_meta_fstr = [x for x in attr_cfig.attr_config.get('file_io') if 'path_meta' in x][0]['path_meta']

    vals = {'ds_type':ds_type,'write_type':'parquet', 'dir_std_base':dir_std_base,'ds':ds}
    path_meta = path_meta_fstr.format(**vals)
    if not Path(path_meta).exists():
        logging.warning(f"The dataset's metadata mapping file could not be found: \n{path_meta}")
        return None
        #raise FileNotFoundError(f"The dataset's metadata mapping file could not be found: \n{path_meta}")
    if 'parquet' in Path(path_meta).suffix:
        df_meta = pd.read_parquet(path_meta)
    elif 'csv' in Path(path_meta).suffix:
        df_meta = pd.read_csv(path_meta)

    return df_meta



def combine_resp_gdf_comid_wrap(dir_std_base:str|os.PathLike,ds:str,path_attr_config:str|os.PathLike,
                          )->dict:
    """Standardize the response variable and geodataframe/comid retrieval for a single dataset in a wrapper function

    Removes data points from consideration if no comid could be found. Makes the gdf and response data consistent.

    :param dir_std_base: The directory containing the standardized dataset generated from `rafts_prep`
    :type dir_std_base: str | os.PathLike
    :param ds:  The unique dataset identifier
    :type ds: str
    :param path_attr_config: path to the attribute configuration file
    :type attr_config: str | os.PathLike
    :return: dict of the response xarray dataset `'dat_resp'`,
      and the geodataframe with comids & coordinates `'gdf_comid'`
    :rtype: dict
    """
    # Changelog:
    #     2025-05-19 refactor: integrate path_meta for gage_id:featureID-featureSource mapping, GL   
    #     2026-05-12 refactor: update logic around NA handling for featureID col of dat_resp; allow fewer locs than provided in response vars Gemini3Pro
    #     2026-05-27 refactor: update mapper_df logic by dropping duplicate gage_id, Gemini3Pro
    dat_resp = _open_response_data_rafts(dir_std_base,ds)

    # ----- Retrieve a dataset's metadata that maps the gage_id to the featureID
    df_meta = _read_metadata(path_attr_config=path_attr_config, ds=ds)

    # %% COMID & coord retrieval and assignment to response variable's coordinate
    # TODO quickfix: read attribute config file here:
    attr_cfig = AttrConfigAndVars(path_attr_config)
    attr_cfig._read_attr_config()
        
    [featureSource,featureID] = _find_feat_srce_id(dat_resp,
                                   col_schema=attr_cfig.attr_config['col_schema']) # e.g. ['nwissite','USGS-{gage_id}']
    
    path_rafts_dat_resp =  _std_rafts_prep_ds_paths(dir_std_base=dir_std_base,ds=ds,mtch_str='*.nc')
    if len(path_rafts_dat_resp) > 1:
        error_str = f"The following directory contains too many .nc files: {path_rafts_dat_resp}"
        logging.error(error_str)
        raise ValueError(error_str)
    path_gpkg_rafts_prep = _std_rafts_prep_ds_companion_gpkg_path(path_rafts_dat_resp[0])

    # Retrieve the geodataframe of comids
    gdf_comid = rafts_retr_nhdp_comids_geom_wrap(path_save_gpkg=path_gpkg_rafts_prep,
                                  gage_ids=dat_resp['gage_id'].values,
                                featureSource=featureSource, featureID=featureID)
   
    # --- map gdf_comid and df_meta for featureID
    # remove featureID/featureSource columns in gdf_comid that pertain to the gage_id, 
    # not the unique identifiers of those same cols in the df_meta
    if 'featureID' in gdf_comid.columns:
        gdf_comid = gdf_comid.drop(columns=['featureID'])
    if 'featureSource' in gdf_comid.columns:
        gdf_comid = gdf_comid.drop(columns=['featureSource'])
    if df_meta is not None:
        # merge gdf_comid with df_meta based on gage_id column
        df_meta_map = df_meta[['featureID','featureSource','gage_id']].drop_duplicates()
    else:
        # If the file is missing, dynamically build the mapping using the known f-string
        gage_ids = dat_resp['gage_id'].values
        df_meta_map = pd.DataFrame({
            'gage_id': gage_ids,
            'featureSource': featureSource,
            # featureID here acts as the format string (e.g., 'USGS-{gage_id}')
            'featureID': [featureID.format(gage_id=g) for g in gage_ids]
        })
        
    gdf_comid = gdf_comid.merge(df_meta_map, on='gage_id', how='left')

    # --- response data identifier alignment with comids & na removal --- #
    # Subset gdf to the gage_ids that are present in the standardized response variable
    resp_gage_strs = [str(g) for g in dat_resp['gage_id'].values]
    gdf_comid['gage_id_str'] = gdf_comid['gage_id'].astype(str) # Force safe string comparison
    
    sub_gdf_comid = gdf_comid[gdf_comid['gage_id_str'].isin(resp_gage_strs)].copy()
    sub_gdf_comid = sub_gdf_comid.drop(columns=['gage_id_str'])

    if sub_gdf_comid.shape[0] != len(dat_resp['gage_id']):
        logging.warning(f"Warning: The number of gage_ids in the response variable ({len(dat_resp['gage_id'])}) does not match the number of gage_ids in the geodataframe ({sub_gdf_comid.shape[0]}).")
        valid_gage_strs = set(sub_gdf_comid['gage_id'].astype(str))
            
        # Create a boolean mask of which items to keep
        keep_mask = [str(g) in valid_gage_strs for g in dat_resp['gage_id'].values]
        
        # Apply the mask to the xarray dataset
        dat_resp = dat_resp.isel(gage_id=keep_mask)
        
        logging.info(f"Response dataset successfully reduced to {len(dat_resp['gage_id'])} locations.")
        if sub_gdf_comid.shape[0] < len(dat_resp['gage_id']):
            # This was a hard error when using the proc.attr.hydfab retrieval. This is now a warning for hfATLAS applications.
            warn_msg = f"The number of gage_ids in the geodataframe ({sub_gdf_comid.shape[0]}) is less than the number of gage_ids in the response variable ({len(dat_resp['gage_id'])})."
            logging.warning(warn_msg)
            
    if('featureID' in sub_gdf_comid.columns):
        feature_id_col = 'featureID'
    elif('comid' in sub_gdf_comid.columns):
        feature_id_col = 'comid'
    else:
        logging.error("The geodataframe does not contain a column named 'featureID' or 'comid'.")
        raise ValueError(f"The geodataframe does not contain a column named 'featureID' or 'comid'.")
    mapper_df = sub_gdf_comid.drop_duplicates(subset=['gage_id'])

    gage_to_feat_source_map = mapper_df.set_index('gage_id')['featureSource']
    mapped_feat_source = pd.Series(dat_resp['gage_id'].values).map(gage_to_feat_source_map)
    
    gage_to_comid_map = mapper_df.set_index('gage_id')[feature_id_col]
    mapped_comids = pd.Series(dat_resp['gage_id'].values).map(gage_to_comid_map)
    
    # --- Strip PyArrow right before injecting into Xarray ---
    feat_id_arr = np.array(mapped_comids.values, dtype=object)
    feat_src_arr = np.array(mapped_feat_source.values, dtype=object)
    
    dat_resp = dat_resp.assign_coords(
        featureID=("gage_id", feat_id_arr),
        featureSource=("gage_id", feat_src_arr)
    )

    # Find which locations in the GeoDataFrame have NA values for the feature ID
    idxs_na_comid = list(np.where(sub_gdf_comid[feature_id_col].isna())[0])
    
    # # Create a boolean mask of the locations to KEEP (True = Keep, False = Drop)
    gage_id_mask = ~np.isin(np.arange(len(dat_resp['gage_id'])), idxs_na_comid)
    
    if len(idxs_na_comid) > 0:
        # Define explicit integer indices based on the mask
        idx_missing = np.where(~gage_id_mask)[0]
        idx_keep = np.where(gage_id_mask)[0]
        
        # 1. Force the 'gage_id' coordinate array to standard objects/strings
        dat_resp['gage_id'] = dat_resp['gage_id'].astype('O')
        
        # 2. Force the other string coordinates to objects to be safe
        if 'featureID' in dat_resp.coords:
            dat_resp['featureID'] = dat_resp['featureID'].astype('O')
        if 'featureSource' in dat_resp.coords:
            dat_resp['featureSource'] = dat_resp['featureSource'].astype('O')

        # 3. If any data variables are PyArrow strings, convert them too
        for var in dat_resp.data_vars:
            if dat_resp[var].dtype == 'string' or str(dat_resp[var].dtype).startswith('string['):
                 dat_resp[var] = dat_resp[var].astype('O')

        # --- Force PyArrow strings into standard NumPy arrays ---
        # Manually subset the NumPy array using our integer mask to get the missing IDs
        # Now that PyArrow is stripped, Xarray's standard .isel() will work flawlessly
        gage_ids_missing = dat_resp['gage_id'].isel(gage_id=idx_missing).values
        logging.info(f"A total of {len(idxs_na_comid)} returned location IDs are NA values. \
               \nRemoving the following gage_ids from the dataset: \
              \n{gage_ids_missing}")
            
        # Remove NA vals from gage_id coord using explicit integer indexing
        dat_resp = dat_resp.isel(gage_id=idx_keep)

    sub_gdf_comid = sub_gdf_comid.drop_duplicates().dropna(subset=['featureID'],axis=0)
    if any(sub_gdf_comid[feature_id_col].duplicated()):
        logging.info("Note that some duplicated comids found in dataset based on initial location identifier, gage_id")
    sub_gdf_comid['dataset'] = ds 

    dict_resp_gdf = dict({'dat_resp':dat_resp,
                        'gdf_comid': sub_gdf_comid})
    return(dict_resp_gdf)

def split_train_test_comid_wrap(dir_std_base:str|os.PathLike, 
                datasets:list, path_attr_config:str | os.PathLike,
                id_col='featureID', test_size:float=0.3,
                random_state:int=42) -> dict:
    """Create a train/test split based on shared comids across multiple datasets
    Helpful when multiple datasets desired for intercomparison share the same comids, but 
    some datasets don't have the same size (e.g. dataset A has 489 locations whereas dataset B has 512 locations)
    If datasets all share the same comids, or only one dataset provided, then proceeds with the standard train-test split.

    :param dir_std_base:  The directory containing the standardized dataset generated from `rafts_prep`
    :type dir_std_base: str | os.PathLike
    :param datasets: The unique dataset identifiers as a list
    :type datasets: list
    :param path_attr_config: path to the attribute configuration file
    :type path_attr_config: str | os.PathLike
    :param id_col: The column name of the comid in geodataframe as returned by `rafts_retr_nhdp_comids_geom`, defaults to 'featureID'
    :type id_col: str, optional
    :param test_size: The fraction of data reserved for test data, defaults to 0.3
    :type test_size: float, optional
    :param random_state: The random state/random seed number, defaults to 42
    :type random_state: int, optional
    :seealso: :func:`train_test_split`
    :return: A dictionary containing the following objects:
        'dict_gdf_comids': dict of dataset keys, each with the geodataframe of comids
        'sub_test_ids': the comids corresponding to testing
        'sub_train_ids': the comids corresponding to training
    :rtype: dict
    """
    # Changelog/contributions
    #    2025-05-19 refactor to pass path_attr_config in lieu of attr_config, change default id_col to featureID, GL
    

    dict_gdf_comids = dict()
    for ds in datasets:
    
        # Generate the geodataframe in a standard format
        dict_resp_gdf = combine_resp_gdf_comid_wrap(dir_std_base,ds,path_attr_config=path_attr_config )
    
        dict_gdf_comids[ds] = dict_resp_gdf['gdf_comid']

    if len(datasets) > 1:
        common_locid = find_common_comid(dict_gdf_comids, column = id_col)
    else:
        common_locid = dict_gdf_comids[ds][id_col].tolist()
    
    # Create the train/test split() of comids. Note that duplicates are possible and must be removed!
    df_common_locids = pd.DataFrame({id_col:common_locid}).dropna().drop_duplicates()
    train_ids, test_ids = train_test_split(df_common_locids, test_size=test_size, random_state=random_state)

    # Compile results into a standard structure
    split_dict = {'dict_gdf_comids' : dict_gdf_comids,
                'sub_test_ids': test_ids[id_col],
                'sub_train_ids': train_ids[id_col]}
    return split_dict

def _warn_if_out_of_bounds(predictions: np.ndarray, feature_ids: pd.Series, 
                          min_lim: float, max_lim: float, resp_var: str, 
                          correction_is_active: bool, prediction_type: str):
    """
    Checks if predictions are within bounds and logs a specific warning if they are not.

    :param predictions: The 1D (values) or 3D (intervals) prediction array.
    :type predictions: np.ndarray
    :param feature_ids: The feature IDs corresponding to the predictions.
    :type feature_ids: pd.Series
    :param min_lim: The minimum allowable value for the prediction.
    :type min_lim: float
    :param max_lim: The maximum allowable value for the prediction.
    :type max_lim: float
    :param resp_var: The name of the response variable for logging purposes.
    :type resp_var: str
    :param correction_is_active: Flag indicating if a correction will be applied.
    :type correction_is_active: bool
    :param prediction_type: A string ('values' or 'intervals') for the log message.
    :type prediction_type: str
    """
    if isinstance(min_lim, str) and min_lim.lower() == 'none':
        min_lim = None
    if isinstance(max_lim, str) and max_lim.lower() == 'none':
        max_lim = None
        
    if min_lim is None and max_lim is None:
        return

    clip_min = min_lim if min_lim is not None else -np.inf
    clip_max = max_lim if max_lim is not None else np.inf

    # Find where original values are out of bounds
    out_of_bounds_mask = (predictions < clip_min) | (predictions > clip_max)
    
    if np.any(out_of_bounds_mask):
        # Logic to find affected IDs works for both 1D and 3D arrays
        if predictions.ndim == 3:
            affected_indices = np.any(out_of_bounds_mask, axis=(1, 2))
        else: # Handles 1D arrays
            affected_indices = out_of_bounds_mask
        
        affected_feature_ids = feature_ids[affected_indices].tolist()
        
        message = (
            f"'{resp_var}' prediction {prediction_type} found outside allowable range "
            f"[{clip_min}, {clip_max}] for {len(affected_feature_ids)} location(s): "
            f"{affected_feature_ids}."
        )
        if correction_is_active:
            message += " Post-hoc correction will be applied."
        else:
            message += " Correction was NOT applied because the relevant flag is disabled."
        logging.warning(message)

def clip_predictions(y_pred: np.ndarray, min_lim: float, max_lim: float) -> np.ndarray:
    """Clips a 1D prediction array to the specified min/max bounds.

    :param y_pred: The 1D prediction array.
    :type y_pred: np.ndarray
    :param min_lim: The minimum allowable value for the prediction.
    :type min_lim: float
    :param max_lim: The maximum allowable value for the prediction.
    :type max_lim: float
    :return: The clipped 1D prediction array.
    :rtype: np.ndarray
    """
    if min_lim is None and max_lim is None:
        return y_pred
    clip_min = min_lim if min_lim is not None else -np.inf
    clip_max = max_lim if max_lim is not None else np.inf
    return np.clip(y_pred, clip_min, clip_max)

def clip_pis(y_pis: np.ndarray, min_lim: float, max_lim: float) -> np.ndarray:
    """Clips prediction intervals (y_pis) to the specified min/max bounds.
    
    :param y_pis: The prediction interval array from MAPIE.
    :type y_pis: np.ndarray
    :param min_lim: The minimum allowable value for the prediction.
    :type min_lim: float
    :param max_lim: The maximum allowable value for the prediction.
    :type max_lim: float
    :return: The clipped prediction interval array.
    :rtype: np.ndarray    
    """
    if min_lim is None and max_lim is None:
        return y_pis
    clip_min = min_lim if min_lim is not None else -np.inf
    clip_max = max_lim if max_lim is not None else np.inf
    return np.clip(y_pis, clip_min, clip_max)

# %% PROC ALGO VIZ UTILITIES

def read_validated_attribute_selection(
    attr_cfig: Any, # raftsutil.AttrConfigAndVars object
    path_cfig: str | os.PathLike, # path_algo_config
    name_attr_csv: str | None,
    colname_attr_csv: str | None,
    arg_val: bool = False
) -> List[str]:
    """
    Retrieves the list of selected attributes and performs Pandera validation on the list.

    :param attr_cfig: The parsed AttrConfigAndVars object.
    :param path_cfig: Path of the referring config file (for path resolution).
    :param name_attr_csv: Name of CSV file containing attributes, if used.
    :param colname_attr_csv: Column name in the CSV file, if used.
    :param arg_val: Flag to enable Pandera schema validation.
    :return: List of selected attributes.
    :rtype: List[str]
    """
    # Grab the attributes of interest from the attribute config file, OR a .csv file.
    attrs_sel = _id_attrs_sel_wrap(
        attr_cfig=attr_cfig,
        path_cfig=path_cfig,
        name_attr_csv = name_attr_csv,
        colname_attr_csv = colname_attr_csv
    )
    
    # --- VALIDATION: Selected Attributes ---
    if arg_val:
        try:
            schemas.schema_attrs_sel.validate(pd.DataFrame(attrs_sel))
            logging.info("✅ Attributes Selection DataFrame validated successfully.")
        except Exception as e:
            logging.error(f"❌ Validation failed for Attributes Selection: {e}")
            sys.exit(1)
            
    return attrs_sel

def validate_gdf_comid_schema(gdf_comid: gpd.GeoDataFrame, arg_val: bool=False):
    """
    Validates the structure and geometry format of the GeoDataFrame.

    :param gdf_comid: The GeoDataFrame to validate.
    :type gdf_comid: gpd.GeoDataFrame
    :param arg_val: Flag to enable Pandera schema validation.
    :type arg_val: bool
    """
    if arg_val:
        # --- VALIDATION: GDF Comid ---
        try:
            schemas.schema_gdf_comid.validate(gdf_comid)
            logging.info("✅ GDF Comid DataFrame validated successfully.")
        except Exception as e:
            logging.error(f"❌ Validation failed for GDF Comid: {e}")
            sys.exit(1)

def validate_dat_resp_schema(dat_resp: xr.Dataset, valid_metrics: List[str], col_locid: str, arg_val: bool):
    """
    Validates the structure and metric columns of the Xarray Dataset response data.

    :param dat_resp: The Xarray Dataset response data.
    :type dat_resp: xr.Dataset
    :param valid_metrics: List of metrics for validation.
    :type valid_metrics: List[str]
    :param col_locid: The name of the feature ID column ('featureID').
    :type col_locid: str
    :param arg_val: Flag to enable Pandera schema validation.
    :type arg_val: bool
    """
    if arg_val:
        # --- VALIDATION: Response Data (dat_resp) ---
        try:
            # Construct temporary DF matching schema logic (Metric extraction)
            temp_cols = {
                # "basin_name": dat_resp.get("basin_name", xr.DataArray(np.nan)).values if "basin_name" in dat_resp else None,
                "gage_id": dat_resp["gage_id"].values,
                # "comid": dat_resp["comid"].values if "comid" in dat_resp else None, #"comid": dat_resp.get("comid", xr.DataArray(np.nan)).values,
                "featureID": dat_resp.get(col_locid, xr.DataArray(np.nan)).values,
                "featureSource": dat_resp["featureSource"].values,
            }
            
            # Use metrics from Xarray attributes if available, otherwise rely on valid_metrics from config
            current_metrics = dat_resp.attrs.get('respvar_mappings', '').split('|')
            current_metrics = [m for m in current_metrics if m in dat_resp]

            for metr in current_metrics:
                temp_cols[metr] = dat_resp[metr].values
            
            # Filter out None columns and validate
            temp_cols = {k: v for k, v in temp_cols.items() if v is not None and v.ndim == 1}
            tempDF_dat_resp = pd.DataFrame(temp_cols)

            schema_dat_resp = schemas.build_schema_dat_resp(valid_metrics)
            schema_dat_resp.validate(tempDF_dat_resp)
            logging.info("✅ Response Data DataFrame validated successfully.")
        except Exception as e:
            logging.error(f"❌ Validation failed for Response Data: {e}")
            sys.exit(1)

def write_validated_evaluation_output(
    rslt_eval_df: pd.DataFrame, 
    dir_out_alg_ds: Path, 
    ds: str,
    valid_metrics: List[str],
    arg_val: bool = False
):
    """
    Validates the final model evaluation results schema and writes the result to a csv file.

    :param rslt_eval_df: The DataFrame containing model performance metrics.
    :type rslt_eval_df: pd.DataFrame
    :param dir_out_alg_ds: The directory for algorithm output.
    :type dir_out_alg_ds: Path
    :param ds: The dataset ID.
    :type ds: str
    :param valid_metrics: List of metrics for validation.
    :type valid_metrics: List[str]
    :param arg_val: Flag to enable Pandera schema validation.
    :type arg_val: bool
    """
    
    if arg_val:
        try:
            # Pass the dynamically loaded metrics to the schema builder function
            schema_rslt_eval_df = schemas.build_schema_rslt_eval_df(valid_metrics) 
            schema_rslt_eval_df.validate(rslt_eval_df)
            logging.info("✅ Results Evaluation DataFrame validated successfully.")
        except Exception as e:
            logging.error(f"❌ Validation failed for Results Evaluation: {e}")
            sys.exit(1)
            
    rslt_eval_df['dataset'] = ds
    
    # Final write to csv
    path_eval_csv = Path(dir_out_alg_ds)/Path(f'algo_eval_{ds}.csv')
    rslt_eval_df.to_csv(path_eval_csv,index=False)
    logging.info(f'... Wrote training and testing evaluation to file for {ds} at {path_eval_csv}')

# %% PRED ALGO UTILITIES

def load_validated_pipeline(path_algo: Path, arg_val: bool = False) -> Dict[str, Any]:
    """
    Loads a joblib-serialized ML pipeline and validates its contents using Pydantic.

    :param path_algo: Path to the .joblib file containing the trained model pipeline.
    :type path_algo: Path
    :param arg_val: Flag to enable Pydantic validation.
    :type arg_val: bool
    :raises FileNotFoundError: If the path does not exist.
    :raises ValueError: If Pydantic validation fails.
    :return: The loaded dictionary containing the pipeline and metadata.
    :rtype: Dict[str, Any]
    """
    if not path_algo.exists():
        msg_nonexst = f"The following algorithm path does not exist: \n{path_algo}"
        logging.error(msg_nonexst)
        raise FileNotFoundError(msg_nonexst)

    pipeline_data = joblib.load(path_algo)

    # --- VALIDATION: Loaded Pipeline ---
    if arg_val:
        try:
            ModelMetadata(**pipeline_data)
            logging.info("✅ Loaded Algorithm Pipeline validated successfully (Pydantic).")
        except Exception as e:
            logging.error(f"❌ Validation failed for Loaded Pipeline ({path_algo.name}): {e}")
            # Exit here as pipeline corruption is a critical failure
            sys.exit(1)
            
    return pipeline_data

def validate_input_attributes(
    df_attr: pd.DataFrame, 
    arg_val: bool = False):
    
    """
    Validates schema if arg_val==True

    :param df_attr: Initial unvalidated DataFrame of attributes.
    :type df_attr: pd.DataFrame
    :param arg_val: Flag to enable Pandera schema validation.
    :type arg_val: bool
    """

    # --- VALIDATION: Input Attribute Data (df_attr) ---
    if arg_val:
        try:
            schema_df_attr = schemas.schema_df_attr
            schema_df_attr.validate(df_attr)
            logging.info("✅ Input Attribute DataFrame validated successfully.")
        except Exception as e:
            logging.error(f"❌ Validation failed for Input Attribute Data: {e}")
            
            # Exit here as input data schema failure is critical
            sys.exit(1)

def infer_mapie_alphas(columns: list) -> list[float]:
    """
    Dynamically extracts unique MAPIE alpha values from a list of DataFrame columns.
    
    :param columns: List of column names (e.g., from df.columns).
    :return: A sorted list of float alpha values.
    """
    alphas = set()
    for col in columns:
        if col.startswith('mapie_lower_'):
            try:
                alphas.add(float(col.split('_')[-1]))
            except ValueError:
                continue
    return sorted(list(alphas))

def infer_mapie_errors(df: pd.DataFrame, alpha_val: float, colname_data: str = 'prediction') -> dict:
    """
    Infer lower, upper, and total MAPIE errors for a specific alpha value.
    
    :param df: DataFrame containing the prediction and MAPIE bounds.
    :param alpha_val: The specific alpha value to calculate errors for.
    :param colname_data: The name of the central prediction column.
    :return: Dictionary containing the error series and the global min/max for scaling.
    """
    l_col = f"mapie_lower_{alpha_val:.2f}"
    u_col = f"mapie_upper_{alpha_val:.2f}"
    
    if l_col not in df.columns or u_col not in df.columns:
        raise KeyError(f"MAPIE columns for alpha {alpha_val:.2f} not found in DataFrame.")
        
    lower_err = np.abs(df[colname_data] - df[l_col])
    upper_err = np.abs(df[u_col] - df[colname_data])
    total_err = lower_err + upper_err
    
    return {
        'lower_err': lower_err,
        'upper_err': upper_err,
        'total_err': total_err,
        'min_err': total_err.min(),
        'max_err': total_err.max()
    }

def write_validated_prediction_output(
    df_pred_mrge: pd.DataFrame, 
    path_pred_out: Path, 
    arg_val: bool, 
    valid_metrics: List[str],
    mapie_alpha: List[float] = None
):
    """
    Validates the final prediction output schema and writes the result to a Parquet file.

    :param df_pred_mrge: The merged DataFrame containing predictions, metadata, and uncertainty bounds.
    :type df_pred_mrge: pd.DataFrame
    :param path_pred_out: The destination path for the Parquet file.
    :type path_pred_out: Path
    :param arg_val: Flag to enable Pandera schema validation.
    :type arg_val: bool
    :param valid_metrics: List of metrics to use for schema validation checks.
    :type valid_metrics: List[str]
    :param mapie_alpha: List of alpha values used for MAPIE uncertainty columns.
    :type mapie_alpha: List[float], optional
    """
    if arg_val:
        try:
            # Identify uncertainty columns for schema builder
            uncertainty_cols = [
                col for col in df_pred_mrge.columns 
                if col.startswith('forestci') or col.startswith('mapie_')
            ]
            
            # Use the schema builder function defined in schemas.py
            schema_df_pred = schemas.build_schema_df_pred(
                valid_metrics=valid_metrics,
                uncertainty_cols=uncertainty_cols,
                mapie_alphas=mapie_alpha
            )
            
            # Perform validation
            schema_df_pred.validate(df_pred_mrge)
            logging.info("✅ Prediction Output DataFrame validated successfully.")
        except Exception as e:
            logging.error(f"❌ Prediction Output validation failed: {e}")
            sys.exit(1)

    # Write prediction results
    df_pred_mrge.to_parquet(path_pred_out)
    logging.info(f"Wrote prediction output to: {path_pred_out}")

# %% rafts_tfrm_attr UTILITIES
def validate_df_comids(
    df_comids: pd.DataFrame, 
    arg_val: bool = False):
    
    """
    Reads attribute data, performs basic cleaning, and validates schema if requested.

    :param df_comids: Initial unvalidated DataFrame of attributes.
    :type df_comids: pd.DataFrame
    :param arg_val: Flag to enable Pandera schema validation.
    :type arg_val: bool
    """

    # --- VALIDATION: Attribute Data (df_comids) ---
    if arg_val:
        try:
            schema_df_comids = schemas.schema_df_comids  # Load schema from schemas.py
            validated_df_comids = schema_df_comids.validate(df_comids)
            print("✅ DataFrame validated successfully.")
        except Exception as e:
            print(f"❌ Validation failed: {e}")
            sys.exit(1)
    
    return df_comids


# --------------------------------------------------------------------------- #
# --------------------------------- hfATLAS --------------------------------- #
def read_hfatlas_wrap_dask(paths_hfatl: Union[Path, str, List[Union[Path, str]]], attrs_sel: list = None, 
                           map_id_col: str = "divide_id", query_clean: bool = False) -> pd.DataFrame: 
    """
    Highly efficient Dask/PyArrow implementation to read and merge requested attributes.

    Peeks at Parquet metadata to resolve pint-aware tuples before lazy-loading data.

    :param paths_hfatl: A single file path, directory string, or a list of paths pointing to the hfATLAS Parquet files.
    :type paths_hfatl: Union[Path, str, List[Union[Path, str]]]
    :param attrs_sel: A list of specific attribute column names to retrieve. If None, retrieves all available attributes. Defaults to None.
    :type attrs_sel: list, optional
    :param map_id_col: The name of the column used as the unique catchment/location identifier for merging. Defaults to "divide_id".
    :type map_id_col: str, optional
    :param query_clean: A boolean flag indicating whether to clean the resulting dataset (e.g., dropping NAs or handling unit stripping) after querying. Defaults to False.
    :type query_clean: bool, optional
    :return: A dataframe containing the merged and requested hfATLAS attributes.
    :rtype: pd.DataFrame
    """
    # Changelog:
    #  2026-07-21 fix: enforce map_id_col dtype read as str, GL
    #  2026-07-23 fix: ignore a col if colname duplicated in a separate parquet, GL
    #  2026-08-05 feat: dynamically check if map_id_col is an index or column to prevent Dask KeyErrors, Gemini3.1Pro
    if attrs_sel is None:
        attrs_sel = []

    if not isinstance(paths_hfatl, list):
        paths_hfatl = [paths_hfatl]
        
    # 1. Expand directories into a list of specific parquet files
    all_files = []
    for p in paths_hfatl:
        p_obj = Path(p)
        if p_obj.is_dir():
            all_files.extend(list(p_obj.rglob("*.parquet")))
        elif p_obj.is_file() and p_obj.suffix == '.parquet':
            all_files.append(p_obj)
            
    if not all_files:
        logging.error("No valid parquet files found in the provided hfATLAS paths.")
        return pd.DataFrame()

    ddrafts_to_merge = []
    found_attrs = set()

    # 2. PEEK phase: Read only the schema metadata (extremely fast, ~0 RAM)
    for file in all_files:
        try:
            pq_file = pq.ParquetFile(file)
            raw_cols = pq_file.schema.names
            
            # Dynamically extract pandas index metadata
            index_cols = []
            if pq_file.schema_arrow.metadata and b'pandas' in pq_file.schema_arrow.metadata:
                try:
                    pandas_meta = json.loads(pq_file.schema_arrow.metadata[b'pandas'].decode('utf8'))
                    raw_index_cols = pandas_meta.get('index_columns', [])
                    
                    # Ensure we extract the string names safely
                    for ic in raw_index_cols:
                        if isinstance(ic, str):
                            index_cols.append(ic)
                        elif isinstance(ic, dict) and 'name' in ic and ic['name']:
                            index_cols.append(ic['name'])
                except Exception as meta_err:
                    logging.debug(f"Could not parse pandas metadata in {file.name}: {meta_err}")
                    
        except Exception as e:
            logging.warning(f"Could not read schema for {file.name}: {e}")
            continue

        # Build a mapping of clean_name -> raw_name for this specific file
        col_mapping = {}
        for raw_col in raw_cols:
            clean_col, _ = parse_hfatlas_colname(raw_col)
            col_mapping[clean_col] = raw_col

        if query_clean and not attrs_sel:
            attrs_sel = [c for c in col_mapping.keys() if c != map_id_col]

        # Determine which requested attributes actually exist in this file
        available_clean_cols = [col for col in attrs_sel if col in col_mapping]
            
        logging.info(f"Found {len(available_clean_cols)} requested attributes in {file.name}")
        found_attrs.update(available_clean_cols)

        # Determine if map_id_col is an explicit column or an index
        raw_map_id_col = col_mapping.get(map_id_col, map_id_col)
        is_index = raw_map_id_col in index_cols
        is_col = raw_map_id_col in raw_cols

        # The exact raw strings we need to ask Dask to load
        raw_cols_to_load = [col_mapping[col] for col in available_clean_cols]
        rename_dict = {col_mapping[col]: col for col in available_clean_cols}
        
        dask_kwargs = {'engine': 'pyarrow'}

        if is_index:
            # Tell Dask to specifically load this as the index (do NOT add to raw_cols_to_load)
            dask_kwargs['index'] = raw_map_id_col
        elif is_col:
            # Bypass hidden metadata indices to prevent KeyError, load as standard column
            dask_kwargs['index'] = False
            raw_cols_to_load.insert(0, raw_map_id_col)
            rename_dict[raw_map_id_col] = map_id_col
        else:
            logging.warning(f"'{map_id_col}' not found as a column or index in {file.name}")
            continue

        # 3. LAZY LOAD phase: Tell Dask to read *only* the specific columns we need
        try:
            ddf = dd.read_parquet(file, columns=raw_cols_to_load, **dask_kwargs)
        except Exception as e:
            logging.error(f"Failed to read {file.name} with Dask: {e}")
            continue
            
        ddf = ddf.rename(columns=rename_dict)
        
        # Extract map_id_col if it was stored as the index instead of a column
        if is_index:
            if ddf.index.name == raw_map_id_col or ddf.index.name == map_id_col:
                ddf = ddf.reset_index()
            else:
                ddf = ddf.reset_index()
                if 'index' in ddf.columns:
                    ddf = ddf.rename(columns={'index': map_id_col})
                elif '__index_level_0__' in ddf.columns:
                    ddf = ddf.rename(columns={'__index_level_0__': map_id_col})
                else:
                    ddf = ddf.rename(columns={ddf.columns[0]: map_id_col})
                    
            # Just in case the raw name wasn't renamed yet
            if raw_map_id_col in ddf.columns and raw_map_id_col != map_id_col:
                ddf = ddf.rename(columns={raw_map_id_col: map_id_col})
                
            logging.info(f"Extracted '{map_id_col}' from the index of {file.name}.")
        
        # Enforce string type inside the lazy Dask graph BEFORE indexing/merging
        ddf[map_id_col] = ddf[map_id_col].astype(str)

        # Set the index to map_id_col to optimize Dask merges
        ddf = ddf.set_index(map_id_col)
        ddrafts_to_merge.append(ddf)

    if not ddrafts_to_merge:
        logging.warning("None of the requested attributes were found across any of the provided files.")
        return pd.DataFrame(columns=[map_id_col] + attrs_sel)

    # 4. MERGE phase: Let Dask build the graph to outer join all datasets on the index
    logging.info("Building Dask merge graph...")
    combined_ddf = ddrafts_to_merge[0]
    for i in range(1, len(ddrafts_to_merge)):
        right_ddf = ddrafts_to_merge[i]
        # Identify any columns (excluding the index) that already exist in the combined_ddf
        overlapping_cols = set(combined_ddf.columns).intersection(right_ddf.columns)
        
        # Drop the overlapping columns from the right-hand dataframe
        if overlapping_cols:
            right_ddf = right_ddf.drop(columns=list(overlapping_cols))
        combined_ddf = combined_ddf.join(right_ddf, how='outer')

    # 5. COMPUTE phase: Execute the graph and bring the final, slimmed-down table into Pandas RAM
    logging.info("Executing computations and pulling to memory...")
    combined_df = combined_ddf.compute().reset_index()

    # Enforce string dtype on the location identifier to prevent downstream bugs
    if map_id_col in combined_df.columns:
        col_dtype = combined_df[map_id_col].dtype
        if not pd.api.types.is_object_dtype(col_dtype) and not pd.api.types.is_string_dtype(col_dtype):
            warn_str = f"Location identifier column '{map_id_col}' was read as {col_dtype}. Coercing to string to prevent downstream ID-matching errors."
            logging.warning(warn_str)
            print(warn_str)
            combined_df[map_id_col] = combined_df[map_id_col].astype(str)

    # Check for missing columns across the entire batch
    miss_cols = [col for col in attrs_sel if col not in found_attrs]
    if len(miss_cols) > 0:
        logging.warning(f"Problem with attribute selection. The following are missing across all searched files: {miss_cols}")
        
    return combined_df

import ast
from typing import Any

def parse_hfatlas_colname(raw_col: Any) -> tuple[str, str | None]:
    """Parses an hfATLAS column representation (tuple or stringified tuple) into a clean name and unit.

    Returns the original stringified value if parsing fails.

    :param raw_col: The raw column representation to parse (string or actual tuple).
    :type raw_col: Any
    :return: A tuple containing the clean column name and the extracted unit (or None if no unit).
    :rtype: tuple[str, str | None]
    """
    clean_name = str(raw_col)
    unit = None
    
    # 1. Handle actual Python tuples (e.g., from MultiIndex or .pint.dequantify)
    if isinstance(raw_col, tuple):
        if len(raw_col) > 0:
            clean_name = str(raw_col[0])
        if len(raw_col) >= 2:
            unit = str(raw_col[1])
            
    # 2. Handle stringified tuples (e.g., "('TOT_AET_hfa', 'mm')")
    elif isinstance(raw_col, str) and raw_col.startswith("(") and raw_col.endswith(")"):
        try:
            # Safely evaluate the string back into a real tuple
            parsed_tuple = ast.literal_eval(raw_col)
            if isinstance(parsed_tuple, tuple):
                if len(parsed_tuple) > 0:
                    clean_name = str(parsed_tuple[0])
                if len(parsed_tuple) >= 2:
                    unit = str(parsed_tuple[1])
        except Exception:
            # Broadened exception catch to mimic strip_units fallback logic
            pass
            
    return clean_name, unit

def clean_hfatlas_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Parses pint-aware string columns and renames them to standard strings.

    :param df: The dataframe containing raw hfATLAS column names (e.g., "('TOT_AET_hfa', 'millimeter')").
    :type df: pd.DataFrame
    :return: The dataframe with cleaned, standard column names (e.g., 'TOT_AET_hfa').
    :rtype: pd.DataFrame
    """
    new_cols = {}
    for col in df.columns:
        clean_col, _ = parse_hfatlas_colname(col)
        new_cols[col] = clean_col
            
    return df.rename(columns=new_cols)

def create_hfatlas_unit_mapper(raw_columns: list) -> pd.DataFrame:
    """
    Creates a dataframe mapping raw hfATLAS column names to their cleaned names and pint units.
    
    :param raw_columns: A list of raw column names (e.g., from df.columns or parquet schema)
    :type raw_columns: list
    :return: A DataFrame containing 'raw_column', 'clean_column', and 'unit'
    :rtype: pd.DataFrame
    """
    mapping_data = []
    
    for col in raw_columns:
        clean_name, unit = parse_hfatlas_colname(col)
        
        mapping_data.append({
            'raw_column': col,
            'clean_column': clean_name,
            'unit': unit
        })

    return pd.DataFrame(mapping_data)

def std_unit_mapper_path(dir_std_base: str | Path, ds: str, cstm_str:str='') -> Path:
    """
    Standardize the filepath for saving the hfATLAS unit mapper.

    :param dir_std_base: The base directory for standardized data
    :type dir_std_base: str | Path
    :param ds: The dataset string/name
    :type ds: str
    :param cstm_str: A custom string to add to the end of the filename
    :type cstm_str: str
    :return: Full path to the unit mapper CSV
    :rtype: Path
    """
    if cstm_str == '':
        path_mapper = Path(dir_std_base) / ds / f"hfatlas_unit_mapper_{ds}.csv"
    else:
        path_mapper = Path(dir_std_base) / ds / f"hfatlas_unit_mapper_{ds}_{cstm_str}.csv"
    path_mapper.parent.mkdir(parents=True, exist_ok=True)
    return path_mapper

def save_hfatlas_unit_mapper(mapper_df: pd.DataFrame, dir_std_base: str | Path, ds: str, cstm_str:str='') -> Path:
    """
    Writes the unit mapper dataframe to a standardized CSV file.

    :param mapper_df: The unit mapping DataFrame from create_hfatlas_unit_mapper
    :type mapper_df: pd.DataFrame
    :param dir_std_base: The base directory for standardized data
    :type dir_std_base: str | Path
    :param ds: The dataset string/name
    :type ds: str
    :param cstm_str: A custom string to add to the end of the filename
    :type cstm_str: str
    :return: The path where the file was written
    :rtype: Path
    """
    out_path = std_unit_mapper_path(dir_std_base, ds,cstm_str)
    mapper_df.to_csv(out_path, index=False)
    logging.info(f"Wrote hfATLAS pint unit mapper to {out_path}")
    return out_path

def get_middle_vertex(geom) -> Point:
    """Returns the middle existing coordinate/vertex from a line, or a guaranteed internal point for a polygon.
    
    :param geom: The geometry object to evaluate.
    :type geom: shapely.geometry.base.BaseGeometry
    :return: The calculated middle vertex point.
    :rtype: shapely.geometry.Point
    """
    if isinstance(geom, (pd.Series, gpd.GeoSeries)):
        # Recursively apply this exact function to every item in the column
        return geom.apply(get_middle_vertex)

    if geom is None or geom.is_empty:
        return None
        
    if geom.geom_type in ['Polygon', 'MultiPolygon']:
        # representative_point() guarantees the point is safely inside the polygon boundaries
        return geom.representative_point()
        
    # 1. Extract all coordinates into a single list
    if geom.geom_type == 'LineString':
        coords = list(geom.coords)
    elif geom.geom_type == 'MultiLineString':
        # Flatten the coordinates from all line segments into one list
        coords = [coord for line in geom.geoms for coord in line.coords]
    else:
        # Fallback for point geometries or unknown types
        if geom.geom_type == 'Point':
            return geom
        return None 
        
    # 2. Find the middle index
    mid_index = len(coords) // 2
    
    # 3. Return as a shapely Point
    return Point(coords[mid_index])

def generate_algo_points_gpkg_wrap(
    div_ids: pd.Series,
    path_hf_gpkg: str | Path, 
    path_gpkg_rafts_prep: str | Path = None,
    hf_layer: str = 'flowpaths',
    map_id_col: str = 'divide_id',
    featureSource: str = 'hf_id',
    vpu_id_col: str = 'vpuid',
    epsg: int = 4326
) -> gpd.GeoDataFrame:
    """Generates a standardized dataset-specific .gpkg of points for the provided divide_ids using a smart caching and extraction strategy.
      This function is designed to efficiently handle both single GPKG files and directories containing multiple GPKG files, 
      reading only the necessary rows based on the provided divide_ids.
      It also ensures that the resulting GeoDataFrame is standardized with the expected columns and CRS, 
      and can be written to a specified output path for use in downstream processing.

    :param div_ids: A pandas Series of divide_ids for which to extract points.
    :type div_ids: pd.Series
    :param path_hf_gpkg: Path to the hydrofabric GPKG file or directory containing multiple GPKG files.
    :type path_hf_gpkg: str | Path
    :param path_gpkg_rafts_prep: Optional path to write the standardized dataset-specific GPKG of points for downstream use. If None, the GPKG will not be written to disk.
    :type path_gpkg_rafts_prep: str | Path, optional
    :param hf_layer: The layer name in the hydrofabric GPKG to read from, defaults to 'flowpaths'.
    :type hf_layer: str, optional
    :param map_id_col: The column name in the hydrofabric GPKG that corresponds to the divide_ids, defaults to 'divide_id'.
    :type map_id_col: str, optional
    :param featureSource: The value to assign to the 'featureSource' column in the resulting GeoDataFrame, defaults to 'hf_id'.
    :type featureSource: str, optional
    :param vpu_id_col: The column name in the hydrofabric GPKG  that corresponds to the VPUID, defaults to 'vpuid'.
    :type vpu_id_col: str, optional    
    :seealso: rafts_agg_hfatl_basin.py
    :seealso: hfatlas_to_rafts_prep.py
    """
    # Standardized a dataset-specific .gpkg of points using a smart caching and extraction strategy

    path_hf_gpkg = Path(path_hf_gpkg)

    # Read just the divide-ids of interest from the flowpath layer of the hydrofabric GPKG
    formatted_ids = ", ".join([f"'{div_id}'" for div_id in div_ids])
    where_clause = f"{map_id_col} IN ({formatted_ids})"

    gdrafts_to_concat = []
    # Strategy to handle both directories and single files <---
    if path_hf_gpkg.is_dir():
        logging.info(f"Directory detected. Scanning {path_hf_gpkg} for GPKG files...")
        for gpkg_file in path_hf_gpkg.glob("*.gpkg"):
            try:
                # Read ONLY the requested rows for each file
                gdf = gpd.read_file(gpkg_file, layer=hf_layer, where=where_clause, engine="pyogrio")
                if not gdf.empty:
                    gdrafts_to_concat.append(gdf.iloc[[0]]) # Read only the first row
                else: # Try without layer
                    gdf = gpd.read_file(gpkg_file, where=where_clause,engine="pyogrio")
                    if not gdf.empty:
                        gdrafts_to_concat.append(gdf.iloc[[0]]) # Read only the first row
            except Exception as e:
                logging.debug(f"Skipped {gpkg_file.name} or layer '{hf_layer}' not found: {e}")
    elif path_hf_gpkg.is_file():
        logging.info(f"Single GPKG file detected: {path_hf_gpkg.name}")
        # Read ONLY the requested rows
        sub_fp_gdf = gpd.read_file(
            path_hf_gpkg, 
            layer=hf_layer, 
            where=where_clause,
            engine="pyogrio" 
        )
        if not sub_fp_gdf.empty:
            gdrafts_to_concat.append(sub_fp_gdf)
    else:
        raise FileNotFoundError(f"Hydrofabric path is neither a file nor directory: {path_hf_gpkg}")
    
    # Combine the read geometries
    sub_fp_gdf = pd.concat(gdrafts_to_concat, ignore_index=True)
    try:
        crs_gdf = gdrafts_to_concat[0].crs
        active_col = gdrafts_to_concat[0].active_geometry_name
    except:
        crs_gdf = sub_fp_gdf.crs
        active_col = sub_fp_gdf.active_geometry_name
    sub_fp_gdf = gpd.GeoDataFrame(sub_fp_gdf, geometry=active_col, crs=crs_gdf)
    if sub_fp_gdf.active_geometry_name != 'geometry':
        sub_fp_gdf = sub_fp_gdf.rename_geometry('geometry')
    sub_fp_gdf.to_crs(epsg=epsg, inplace=True)

    if vpu_id_col not in sub_fp_gdf.columns:
        logging.warning(f"Expecting {vpu_id_col} vpu col to be in hf dataset {path_hf_gpkg.name}")

    # TODO any additional colname modifications needed? 
    # Historic columns have been: sourceName featureID comid name X Y featureSource  gage_id measure reachcode tot_na
    try:
        sub_fp_gdf['X'] = sub_fp_gdf.geometry.x
        sub_fp_gdf['Y'] = sub_fp_gdf.geometry.y
    except:
        # Apply it to your GeoDataFrame
        gdf = sub_fp_gdf.copy()
        gdf['middle_vertex'] = gdf.geometry.apply(get_middle_vertex)

        # Extract X and Y if you need them in separate columns
        sub_fp_gdf['X'] = gdf['middle_vertex'].apply(lambda p: p.x if p else None)
        sub_fp_gdf['Y'] = gdf['middle_vertex'].apply(lambda p: p.y if p else None)
        
        # Reset the geometry to the middle vertex
        sub_fp_gdf['geometry'] = gdf['middle_vertex']
        sub_fp_gdf = sub_fp_gdf.set_geometry('geometry')

    # TODO figure out what to do about gage_id column.... Should there be an aggregation here based on subsetting???
    sub_fp_gdf['featureID'] = sub_fp_gdf[map_id_col]
    sub_fp_gdf['featureSource'] = featureSource
    sub_fp_gdf['gage_id'] = sub_fp_gdf[map_id_col] # TODO should this always be the same as divide_id?? 
    sub_fp_gdf['comid'] = sub_fp_gdf[map_id_col] # TODO should this always be the same as divide_id?? 
    sub_fp_gdf['tot_na'] = 0 # TODO figure out what to do about tot_na column
    # Write to the dataset-specific gpkg path:
    if path_gpkg_rafts_prep:# The file write location
        sub_fp_gdf.to_file(path_gpkg_rafts_prep, driver="GPKG",  layer = 'outlet')
        logging.info(f"Wrote {sub_fp_gdf.shape[0]} hydrofabric flowpaths to {path_gpkg_rafts_prep}")

    return sub_fp_gdf

def hfatl_std_long(df_hfatlas:pd.DataFrame, map_id_col:str='divide_id', vpu_id_col:str='vpuid')->pd.DataFrame:
    """Melts and standardizes an hfATLAS dataframe into a long format.

    :param df_hfatlas: The raw wide-format hfATLAS dataframe.
    :type df_hfatlas: pd.DataFrame
    :param map_id_col: The unique location/catchment identifier column name, defaults to 'divide_id'.
    :type map_id_col: str, optional
    :param vpu_id_col: The Vector Processing Unit (VPU) identifier column name, defaults to 'vpuid'.
    :type vpu_id_col: str, optional
    :return: The standardized long-format dataframe with 'attribute' and 'value' columns.
    :rtype: pd.DataFrame
    """
    if df_hfatlas.columns.str.contains(vpu_id_col).any():
        df_long = df_hfatlas.melt(id_vars=[map_id_col, vpu_id_col], var_name='attribute', value_name='value')
    else: 
        df_long = df_hfatlas.melt(id_vars=[map_id_col], var_name='attribute', value_name='value')
    df_long['value'] = pd.to_numeric(df_long['value'], errors='coerce')
    df_long = df_long.dropna(subset=['value'])
    return df_long

def hfatl_hf_cmbo_wrap(df_hfatlas:pd.DataFrame, gdf_hf:gpd.GeoDataFrame, 
                        ds:str, dir_db_attrs:str, featureSource:str, map_id_col:str = 'divide_id',vpu_id_col:str = 'vpuid',
                        vpu_mapped:bool = False,data_source:str='hfATLAS')-> pd.DataFrame:
    """Combine attribute data with the gdf containing vpu & write to parquet files within subdirs of vpu

    :param df_hfatlas: The hfatlas attribute data from `read_hfatlas_wrap()`
    :type df_hfatlas: pd.DataFrame
    :param gdf_hf: The hydrofabric flowpath geodataframe from `generate_algo_points_gpkg_wrap()`
    :type gdf_hf: gpd.GeoDataFrame
    :param ds: The dataset name
    :type ds: str
    :param dir_db_attrs: directory where attribute .parquet files live
    :type dir_db_attrs: str
    :param map_id_col: The identifier column in hf & hfatlas datasets, defaults to 'divide_id'
    :type map_id_col: str, optional
    :return: combined hfatlas attributes and hf point geoms. Note point is the midpoint of a flowpath for a divide, not the outlet.
    :rtype: pd.DataFrame
    """
  
    df_hfatlas_geo = df_hfatlas.merge(gdf_hf, how = 'outer', on = map_id_col)
    if df_hfatlas_geo.shape[0] == 0:
        logging.error("Problem: no data generated when combining df_hfatlas with hydrofabric geodataframe")
        sys.exit(1)

    if not vpu_mapped:
        logging.info("No valid VPU mapping found or specified. Defaulting to placeholder VPU ID: 'all'")
        df_hfatlas_geo['vpuid'] = 'all'


    # 4. Reshape to RaFTS Long Format
    logging.info("Reshaping attributes to RaFTS standard schema...")
    # df_long = df_hfatl.melt(id_vars=[map_id_col, vpu_id_col], var_name='attribute', value_name='value')
    # df_long['value'] = pd.to_numeric(df_long['value'], errors='coerce')
    # df_long = df_long.dropna(subset=['value'])
    df_long = hfatl_std_long(df_hfatlas_geo, map_id_col, vpu_id_col)
    df_long = df_long.rename(columns={map_id_col: 'featureID'})
    df_long['featureSource'] = featureSource
    df_long['data_source'] = data_source
    df_long['dl_timestamp'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    
    final_columns = ['vpuid', 'featureID', 'featureSource', 'data_source', 'dl_timestamp', 'attribute', 'value']
    df_long = df_long[final_columns]

    # 5. Distributed I/O: Write parquet files by VPU
    grouped_vpus = df_long.groupby('vpuid')
    total_vpus = len(grouped_vpus)
    
    for i, (vpuid, df_vpu) in enumerate(grouped_vpus):
        df_save = df_vpu.drop(columns=['vpuid'])
        
        save_path = generate_vpu_attr_filepath(dir_db_attrs, ds, vpuid)
        logging.info(f"Writing {save_path.name} ({i+1}/{total_vpus})...")
        df_save.to_parquet(save_path, index=False)
    
    logging.info(f"Wrote parquet files grouped by VPU to {save_path.parent.parent}")
    return df_long

def std_dir_ds_agg(dir_db_attrs: str, ds:str) -> Path:
        """Standardized directory for a basin-aggregated dataset of attributes.
        Intended for aggregating hydrofabric attributes by divide to larger scales

        :param dir_db_attrs: The base directory with {ds} f-string placeholder
        :type dir_db_attrs: str
        :param ds: The dataset string
        :type ds: str
        :return: Directory of the dataset attributes aggregated by basin
        :rtype: Path
        :seealso: rafts_agg_hfatl_basin.py for creation context
        """
        ds_agg_str = ds + '_agg_hfatl'
        dir_db_attrs_agg_save = Path(str(dir_db_attrs).format(ds=ds_agg_str))
        dir_db_attrs_agg_save.mkdir(parents=True, exist_ok=True)
        return dir_db_attrs_agg_save

def std_path_agg_ds(dir_db_attrs_agg_save: Path, ds: str) -> Path:
    """Standardized path for a basin-aggregated dataset of attributes.
        Intended for aggregating hydrofabric attributes by divide to larger scales

    :param dir_db_attrs_agg_save: Directory of the dataset attributes aggregated by basin from std_dir_ds_agg
    :type dir_db_attrs_agg_save: Path
    :param ds: The dataset string
    :type ds: str
    :return: _description_
    :rtype: Path
    """
    out_path = dir_db_attrs_agg_save / f"{ds}_agg_hfatlas.parquet"
    return out_path

def generate_vpu_attr_filepath(dir_db_attrs: Path, dataset_name: str, vpuid: str) -> Path:
    """Creates a standardized filepath grouped by dataset and VPU identifier.

    :param dir_db_attrs: The base directory where attribute parquet files are stored.
    :type dir_db_attrs: Path
    :param dataset_name: The name of the dataset being processed (e.g., 'hfatlas').
    :type dataset_name: str
    :param vpuid: The Vector Processing Unit identifier (e.g., '01', '18', or 'all').
    :type vpuid: str
    :return: The fully resolved Path object for saving the attribute parquet file.
    :rtype: Path
    """
    dir_db_attrs = Path(str(dir_db_attrs).format(ds = dataset_name)) 
    if dataset_name in str(dir_db_attrs):
        # TODO ensure that dir_db_attrs is ALWAYS used here for consistency across codebase attribute storage
        save_dir = dir_db_attrs / str(vpuid)
    else:
        save_dir = dir_db_attrs / dataset_name / str(vpuid)
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir / f"attr_{vpuid}.parquet"

# --- filepaths used in rafts_pair_donors.py:
def std_impute_log_path(dir_regn: str | Path, ds: str, algo: str, resp_var: str) -> Path:
    """Standardize the path for the missing data imputation log."""
    path_out = Path(dir_regn) / ds / f"imputed_locations_{algo}_{resp_var}__{ds}.csv"
    path_out.parent.mkdir(parents=True, exist_ok=True)
    return path_out

def std_donor_pairs_path(dir_regn: str | Path, ds: str, algo: str, resp_var: str) -> Path:
    """Standardize the path for donor-receiver pairings."""
    path_out = Path(dir_regn) / ds / f"donor_pairs_{algo}_{resp_var}__{ds}.csv"
    path_out.parent.mkdir(parents=True, exist_ok=True)
    return path_out

def std_receiver_params_path(dir_regn: str | Path, ds: str, algo: str, resp_var: str) -> Path:
    """Standardize the path for assigned receiver parameters."""
    path_out = Path(dir_regn) / ds / f"receiver_params_{algo}_{resp_var}__{ds}.csv"
    path_out.parent.mkdir(parents=True, exist_ok=True)
    return path_out

def std_receiver_params_mapped_path(dir_regn: str | Path, ds: str, algo: str, resp_var: str, ext: str = ".gpkg") -> Path:
    """Standardize the path for crosswalk-mapped receiver parameters."""
    path_out = Path(dir_regn) / ds / f"receiver_params_mapped_{algo}_{resp_var}__{ds}{ext}"
    path_out.parent.mkdir(parents=True, exist_ok=True)
    return path_out

# --- Safe f-string Resolver for DataFrames (used in hfatlas_to_rafts_prep.py) ---
class SafeDict(dict):
    """Allows safe mapping where missing keys are left as unresolved f-strings (e.g. '{missing}')."""
    def __missing__(self, key):
        return '{' + key + '}'

def resolve_fstrings(val, context_dict, max_depth=3):
    """Recursively formats f-strings natively found inside parsed YAML items.
    
    :param val: The value or string template to evaluate and format.
    :type val: Any
    :param context_dict: A dictionary of context variables used for f-string substitution.
    :type context_dict: dict
    :param max_depth: The maximum recursion depth for nested evaluations, defaults to 3.
    :type max_depth: int, optional
    :return: The fully resolved and formatted value.
    :rtype: Any
    """
    if not isinstance(val, str) or '{' not in val:
        return val
    for _ in range(max_depth):
        new_val = val.format_map(SafeDict(**context_dict))
        if new_val == val:
            return new_val
        val = new_val
    return val
# %% Write regionalized params to standardized gpkg
def register_gpkg_attributes_table(conn: sqlite3.Connection, table_name: str):
    """Registers a raw SQLite table as a non-spatial GeoPackage attributes layer.

    :param conn: An active SQLite database connection.
    :type conn: sqlite3.Connection
    :param table_name: The name of the table to register.
    :type table_name: str
    :return: None
    :rtype: None
    """
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT count(name) FROM sqlite_master WHERE type='table' AND name='gpkg_contents'")
        if cursor.fetchone()[0] == 1:
            query = f"""
            INSERT OR IGNORE INTO gpkg_contents (table_name, data_type, identifier, description)
            VALUES ('{table_name}', 'attributes', '{table_name}', 'Regionalized parameters');
            """
            cursor.execute(query)
            conn.commit()
    except sqlite3.Error as e:
        logging.warning(f"Could not register table '{table_name}' in gpkg_contents: {e}")


def create_sqlite_index(conn: sqlite3.Connection, table_name: str, index_col: str):
    """Creates a formal SQLite database index on the specified column to speed up joins.
    
    :param conn: An active SQLite database connection.
    :type conn: sqlite3.Connection
    :param table_name: The name of the table where the index should be created.
    :type table_name: str
    :param index_col: The target column name to index.
    :type index_col: str
    :return: None
    :rtype: None
    """
    try:
        cursor = conn.cursor()
        index_name = f"idx_{table_name}_{index_col}"
        query = f'CREATE INDEX IF NOT EXISTS "{index_name}" ON "{table_name}" ("{index_col}");'
        cursor.execute(query)
        conn.commit()
    except sqlite3.Error as e:
        logging.warning(f"Could not create SQLite index on '{table_name}' for column '{index_col}': {e}")

def update_database(db_path: Path, df_data: pd.DataFrame, table_name: str, id_col: str, overwrite: bool):
    """Helper function to execute standard SQLite table writing, appending, registration, and indexing.
    
    :param db_path: Path to the SQLite/GeoPackage database file.
    :type db_path: pathlib.Path
    :param df_data: The dataframe containing data to write into the database.
    :type df_data: pd.DataFrame
    :param table_name: The target table name inside the database.
    :type table_name: str
    :param id_col: The primary ID column used for indexing or tracking.
    :type id_col: str
    :param overwrite: Whether to overwrite the existing table if it already exists.
    :type overwrite: bool
    :return: None
    :rtype: None
    """
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT count(name) FROM sqlite_master WHERE type='table' AND name='{table_name}'")
            table_exists = cursor.fetchone()[0] == 1
            
            if not table_exists or overwrite:
                if overwrite and table_exists:
                    logging.info(f"Replacing existing table '{table_name}' in {db_path.name}.")
                else:
                    logging.info(f"Creating new table '{table_name}' in {db_path.name}.")
                    
                df_data.set_index(id_col).to_sql(table_name, conn, if_exists='replace', index=True)
                register_gpkg_attributes_table(conn, table_name)
                create_sqlite_index(conn, table_name, id_col)
                
            else:
                logging.info(f"Table '{table_name}' exists in {db_path.name}. Checking for missing {id_col}s...")
                existing_ids_query = f"SELECT {id_col} FROM '{table_name}'"
                try:
                    df_existing = pd.read_sql_query(existing_ids_query, conn)
                    existing_id_set = set(df_existing[id_col].astype(str))
                    
                    df_data[id_col] = df_data[id_col].astype(str)
                    df_new = df_data[~df_data[id_col].isin(existing_id_set)]
                    
                    if not df_new.empty:
                        logging.info(f"Appending {len(df_new)} new records to '{table_name}'.")
                        df_new.set_index(id_col).to_sql(table_name, conn, if_exists='append', index=True)
                        register_gpkg_attributes_table(conn, table_name)
                        create_sqlite_index(conn, table_name, id_col)
                    else:
                        logging.info(f"No new records to append. Table '{table_name}' is up to date.")
                except (sqlite3.OperationalError, pd.errors.DatabaseError):
                    logging.error(f"Identifier column '{id_col}' missing in the existing table '{table_name}'. Cannot append.")
    except sqlite3.Error as e:
        logging.error(f"SQLite error occurred while writing to {db_path.name}: {e}")
        sys.exit(1)

def get_crosswalk_target_col(df_crosswalk: pd.DataFrame, pred_gpkg_id_col: str, crosswalk_target_col: str = 'divide_id') -> str:
    """
    Dynamically identify the target identifier column in a crosswalk using a strict 
    hierarchy: User Config -> Native Default -> Subtraction Fallback.

    :param df_crosswalk: The crosswalk dataframe containing mapping columns.
    :type df_crosswalk: pd.DataFrame
    :param pred_gpkg_id_col: The prediction dataset's GeoPackage identifier column name.
    :type pred_gpkg_id_col: str
    :param crosswalk_target_col: The default fallback crosswalk target column, defaults to 'divide_id'.
    :type crosswalk_target_col: str, optional
    :return: The resolved target column name string.
    :rtype: str
    """

    
    crosswalk_cols = list(df_crosswalk.columns)
    
    if pred_gpkg_id_col not in crosswalk_cols:
        logging.error(f"Prediction ID '{pred_gpkg_id_col}' not found in crosswalk columns.")
        return None
        
    # 1. PRIORITY: The explicitly configured crosswalk_target_col from the prediction config
    if crosswalk_target_col and crosswalk_target_col in crosswalk_cols:
        return crosswalk_target_col
        
    # 2. DEFAULT: Check standard hydrofabric names if no target was explicitly configured
    for default_target in ['divide_id', 'comid']:
        if default_target in crosswalk_cols:
            return default_target

    # 3. FALLBACK: Dynamically subtract the known prediction ID and common spatial metadata
    ignore_meta = [pred_gpkg_id_col.lower(), 'vpuid', 'areasqkm', 'areasqmi', 'lengthkm', 'gage_id', 'custom_id', 'site_id']
    possible_cols = [c for c in crosswalk_cols if c.lower() not in ignore_meta]
    
    if possible_cols:
        desired_id_col = possible_cols[0]
        if len(possible_cols) > 1:
            logging.warning(f"Multiple unknown columns found in crosswalk {possible_cols}. Defaulting to '{desired_id_col}'.")
        logging.warning(f"Assuming crosswalk target column to be the first column in the crosswalk file: {desired_id_col}")
        return desired_id_col
        
    logging.error("Could not identify a valid topological target identifier column in the crosswalk file.")
    return None
