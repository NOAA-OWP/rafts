"""Perform response variable predictions using trained algorithms that have been trained
using rafts_proc_algo_viz.py. Refers to prediction configuration file, which should have also
been used in a user-generated, custom script that generates the predictor data to be used 
in this prediction script.

Usage:
    >>> python rafts_pred_algo.py "/path/to/datasetshortname_pred_config.yaml"

# Changelog/Contributions:
2024 Originally created, GL
2025-06-10 Generalize to specify featureID and featureSource columns in the prediction output, GL
2025-10-10 refactor to renamed rafts_algo modules, GL
2025-11-20 Integrated validation using schemas.py and pydantic_schemas.py, Soroush Sorourian with the help of AI.
2025-11-25 refactor: move schemas to package structure and adjust import logic, [Soroush Sorourian/AI]
2025-11-26 feat: Dynamically load valid metrics for schema validation, Soroush Sorourian
2025-11-26 refactor: Replaced dynamic metric loading logic with rafts_algo.utils.get_valid_metrics, Soroush Sorourian
2025-12-01 refactor: Formalized loading, processing, and writing into utility functions, Soroush Sorourian with the help of AI.
2025-12-08 refactor: Updated dynamic metric loading logic to be retrieved from the prediction config, Soroush Sorourian
2026-07-21 fix: read custom prediction dataset dir from path_pred_locs via read_hfatlas_wrap_dask, GL
2026-09-18 refactor: Integrate Pydantic PredConfig schema validation to structurally enforce config typing.
"""

import argparse
import joblib
import rafts_algo.utils as raftsutil
import rafts_algo.regions as raftsregions
import pandas as pd
from pathlib import Path
import forestci as fci
from sklearn.model_selection import train_test_split
from logging.handlers import MemoryHandler
import logging
import rafts_prep.proc_eval_metrics as pem
import numpy as np

import rafts_algo.rafts_algo_train as raftsalgt
from rafts_algo.rafts_algo_train import UniversalDistanceClusterer
from rafts_algo.schemas.pydantic_schemas import PredConfig

# Imports for validation
import importlib.util
import sys

# Predict values and evaluate predictions
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'process the prediction config file')
    parser.add_argument('path_pred_config', type=str, help='Path to the YAML configuration file specific for prediction.')
    #parser.add_argument('path_algo_config', type=str, help='Path to the YAML configuration file specific for algorithm training')
    parser.add_argument('--validate', action='store_true', default=False,
                        help='If present, enables schema validation for all input and output data. Defaults to False.')
    parser.add_argument('--region', type=str, default=None,
                        help="Restrict this run to a single configured sub-region's region_id. "
                             "Omit to loop over every region the pred config's `regions:` block "
                             "resolves to (or run once, unregioned, if `regions:` isn't set).")
    # NOTE pred_config should contain the path for path_algo_config
    args = parser.parse_args()

    path_pred_config = Path(args.path_pred_config).expanduser() # Path(f'~/git/rafts/scripts/workflow_configs/legacy/xssa/xssa_pred_config.yaml').expanduser()     

    # --- Conditionally load schemas ---
    arg_val = args.validate 
    if arg_val:
        logging.info("Schema validation enabled. Using statically imported schemas from rafts_algo.schemas.")

    # --- Commence logging before creating the log file
    memory_handler = MemoryHandler(capacity=30)
    # Get the root logger and add the memory handler to it
    # The root logger is the ancestor of all other loggers
    root_logger = logging.getLogger()
    root_logger.addHandler(memory_handler)
    root_logger.setLevel(logging.INFO) # Set the level to capture INFO messages
    logging.info(f"Running rafts_pred_algo.py with \
                {path_pred_config.parent / path_pred_config.name} config file")
    # ---   
        
    # --- 1. Pydantic validation
    validated_pred_cfg = raftsutil.load_validated_config(path_pred_config, PredConfig)

    # Extract top-level scalars strictly from Pydantic model
    uncn_bnd_pred = validated_pred_cfg.uncn_bnd_pred
    
    # READ rafts_categories.yaml if uncn_bnd_pred is True
    logging.info("Reading uncertainty bounds from rafts_categories.yaml...")

    rafts_catg_uncn = pem._conv_ls_dicts_df_long()
    logging.info("Successfully loaded uncertainty bounds.")

    #%%  READ CONTENTS FROM THE ATTRIBUTE CONFIG
    path_attr_config = raftsutil.build_cfig_path(path_pred_config, validated_pred_cfg.name_attr_config)
    path_algo_config = raftsutil.build_cfig_path(path_pred_config, validated_pred_cfg.name_algo_config)

    attr_cfig = raftsutil.AttrConfigAndVars(path_attr_config)
    attr_cfig._read_attr_config()

    dir_base = attr_cfig.attrs_cfg_dict.get('dir_base')
    dir_std_base = attr_cfig.attrs_cfg_dict.get('dir_std_base')
    dir_db_attrs = attr_cfig.attrs_cfg_dict.get('dir_db_attrs')
    datasets = attr_cfig.attrs_cfg_dict.get('datasets') # Identify datasets of interest

    key_names = [key for dictionary in attr_cfig.attr_config['attr_select'] for key in dictionary.keys()] 
    # Parse the prep config for hfatlas specifics
    if 'paths_hfatl' in key_names: # This is the hfATLAS application rather than pre-2026 RaFTS using NLDI
        hfatl = True
        
        # 1. Safely extract name_prep_config
        prep_cfg_entries = [x for x in attr_cfig.attr_config.get('file_io', []) if 'name_prep_config' in x]
        
        if prep_cfg_entries:
            name_prep_config = prep_cfg_entries[0]['name_prep_config']
            path_prep_config = raftsutil.build_cfig_path(path_pred_config, name_prep_config)
            config_df = pem.read_schm_ls_of_dict(path_prep_config)
            raw_config = config_df.iloc[0].dropna().to_dict()
        else:
            # 2. Durable Fallback: Use col_schema directly from the attribute config
            name_prep_config = None
            col_schema_list = attr_cfig.attr_config.get('col_schema', [])
            raw_config = {k: v for item in col_schema_list for k, v in item.items()}

        # 3. Inject the dataset name into the dictionary for f-string resolution
        fio = {k: raftsutil.resolve_fstrings(v, raw_config) for k, v in raw_config.items()}
        map_id_col = fio.get('featureID')

    # if 'paths_hfatl' in key_names: # This is the hfATLAS application rather than pre-2026 RaFTS using NLDI
    #     hfatl = True
    #     name_prep_config = [x for x in attr_cfig.attr_config.get('file_io') if 'name_prep_config' in x][0]['name_prep_config']
    #     path_prep_config = raftsutil.build_cfig_path(path_pred_config,name_prep_config)
    #     config_df = pem.read_schm_ls_of_dict(path_prep_config)
    #     raw_config = config_df.iloc[0].dropna().to_dict()
    #     # Inject the dataset name into the dictionary for f-string resolution
    #     fio = {k: raftsutil.resolve_fstrings(v, raw_config) for k, v in raw_config.items()}
    #     map_id_col = fio.get('featureID')    
    else:
        name_prep_config = None
        hfatl = False

    
    # ---------- Generate path to the log file & initialize logging -----------
    path_log = pem.std_path_log(dir_input=dir_base, 
                                path_config=path_pred_config,
                            script='rafts_pred_algo')
    logging.basicConfig(level=logging.INFO, 
                        filename=path_log, 
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filemode='w', 
                        force = True) # overwrite log file when force=T

    # We need to find the new FileHandler that basicConfig created and set it
    # as the target for our MemoryHandler - aka we can now put previous logs
    # into the log file now that it has been created
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler):
            memory_handler.setTarget(handler)
            memory_handler.flush()
            break  
    logging.info(f"Writing logs to {path_log}")
    root_logger.removeHandler(memory_handler) # Remove the pre-file logger
    # -------------------------------------------------------------------------

    # Initialize algo configuration class for extracting attributes
    algo_cfig = raftsutil.AlgoConfigParser(path_algo_config)
    algo_cfig._read_algo_config()

    task_type = algo_cfig.algo_cfg_unc_dict["algo_cfg_dict"].get("task_type", "regression")
    name_attr_csv = algo_cfig.algo_cfg_unc_dict["algo_cfg_dict"]["name_attr_csv"]
    colname_attr_csv = algo_cfig.algo_cfg_unc_dict["algo_cfg_dict"]["colname_attr_csv"]

    # Determine whether random forest confidence intervals computed during model training:
    forestci = algo_cfig.algo_cfg_unc_dict["algo_unc_dict"]["uncertainty_cfg"].get('forestci',{})
    if len(forestci)>0:
        fci_flag = forestci[0].get('fci_flag',False)
    else:
        fci_flag = False
    # Attributes needed for prediction:
    attrs_sel = raftsutil._id_attrs_sel_wrap(attr_cfig=attr_cfig,
                    path_cfig=path_attr_config,
                    name_attr_csv = name_attr_csv,
                    colname_attr_csv = colname_attr_csv)

    #%% ESTABLISH ALGORITHM FILE I/O
    regions_cfg = validated_pred_cfg.regions  # None unless this config sets a `regions:` block
    region_loop = raftsregions.resolve_region_loop(
        regions_cfg, args.region,
        context={'home_dir': attr_cfig.attrs_cfg_dict.get('home_dir'), 'dir_base': dir_base})
    region_scheme = regions_cfg.scheme if regions_cfg else None
    all_region_specs = [spec for _, spec in region_loop if spec is not None]
    #%% PREDICTION FILE'S COMIDS (IMPLICIT ASSUMPTION: Each dataset processes the same IDS)
    path_meta_pred = validated_pred_cfg.path_meta
    comid_pred_col = validated_pred_cfg.pred_file_comid_colname
    write_type = validated_pred_cfg.write_type
    ds_type = validated_pred_cfg.ds_type

    #%% prediction config
    resp_vars = validated_pred_cfg.algo_response_vars
    algos = validated_pred_cfg.algo_type

    #%% Run prediction
    for ds in datasets:
        # f-string formatting of the attribute metadata's filepath
        path_pred_locs = raftsutil.build_pred_locs_path(path_meta_template=path_meta_pred, dir_std_base=dir_std_base, 
                                                    ds=ds,ds_type=ds_type, write_type=write_type)
        vals = {'dir_std_base':dir_std_base,'ds':ds}
        dir_db_attrs = Path(str(dir_db_attrs).format(**vals))
        #%%  Read in predictor variable data (aka basin attributes) 
        if not hfatl: # The legacy approach using proc.attr.hydfab 
            # Read and validate the predictor variable data (basin attributes) generated by proc.attr.hydfab
            comids_pred = raftsutil._read_pred_comid(path_pred_locs, comid_pred_col)
            df_attr = raftsutil.rafts_read_attr_comid(
                dir_db_attrs, comids_pred, attrs_sel=attrs_sel,
                read_type='filename', _s3=None, storage_options=None
            )
            
            raftsutil.validate_input_attributes(df_attr,arg_val=arg_val)

            df_attr = df_attr.drop(columns='dl_timestamp')

            # Constrain the values in the value column to two digits after the decimal point (to help ID duplicates)
            is_duplicate = df_attr.assign(value=df_attr['value'].round(2)).duplicated()
            df_attr = df_attr[~is_duplicate].copy()
            df_attr.reset_index(inplace=True)

            new_df_attr = df_attr[['featureID', 'featureSource', 'attribute', 'value']]
            map_feat_srce_feat_id = new_df_attr[['featureID', 'featureSource']].drop_duplicates()
            # Convert into wide format for model prediction
            df_attr_wide = new_df_attr.pivot(index='featureID', columns = 'attribute', values = 'value')

        else: # Grab from hydrofabric/hydroATLAS sources
            logging.info(f"Predicting across all locations in files inside {dir_db_attrs} with required attribute data")
            
            # Read hfATLAS-formatted prediction data via the config-specified prediction location (may be a dir)
            df_attr_wide = raftsutil.read_hfatlas_wrap_dask(
                paths_hfatl=path_pred_locs, 
                attrs_sel=attrs_sel,
                map_id_col=comid_pred_col,
                query_clean=True
            )


            if 'featureID' not in df_attr_wide.columns:
                df_attr_wide.rename(columns={comid_pred_col:'featureID'},inplace=True)
            if 'featureSource' not in df_attr_wide.columns:
                # Prefer the prediction config's own featureSource -- training
                # and prediction may legitimately happen at different spatial
                # scales (e.g. trained on gage-basin-aggregated attributes,
                # predicting at HUC10/HUC14/divide scale), so the training-time
                # value from the linked prep config (`fio`) is only a fallback
                # for configs that haven't set a prediction-specific one.
                pred_featureSource = validated_pred_cfg.featureSource
                # `is not None`, not `or` -- an explicitly-configured empty string is
                # still a deliberate (if unusual) choice and must not be silently
                # treated the same as "unset" by falling through to the training-time
                # value. None is PredConfig's actual "unset" sentinel.
                df_attr_wide['featureSource'] = (pred_featureSource if pred_featureSource is not None
                                                  else fio.get('featureSource', 'hf_id'))

            map_feat_srce_feat_id = df_attr_wide[['featureID','featureSource']].drop_duplicates()
            df_attr_wide.set_index('featureID',inplace = True)

        # Region membership is a disjoint partition, so this is computed once for every
        # configured region here rather than once per region inside the loop below --
        # O(n) + O(num_regions) instead of O(num_regions * n) for a potentially large,
        # CONUS-wide attribute index.
        assigned_all = (raftsregions.assign_to_region(df_attr_wide.index.astype(str), all_region_specs)
                         if all_region_specs else None)

        for region_id, region_spec in region_loop:
            dirs_std_dict = raftsutil.rafts_save_algo_dir_struct(dir_base, region=region_id, scheme=region_scheme)
            dir_out = dirs_std_dict.get('dir_out')
            dir_out_alg_base = dirs_std_dict.get('dir_out_alg_base')

            # Region scoping: restrict candidate receivers to this region's CORE only (never
            # the buffer) -- predictions outside a region's own core are meaningless for a
            # model trained to represent exactly that core.
            if region_spec is not None:
                df_attr_wide_region = df_attr_wide[assigned_all.values == region_id]
                if df_attr_wide_region.empty:
                    logging.info(f"No candidate receivers fall in region '{region_id}' for {ds}; skipping.")
                    continue
            else:
                df_attr_wide_region = df_attr_wide

            # Run predictions & save output
            dir_out_alg_ds = Path(dir_out_alg_base/Path(ds))
            logging.info(f"PREDICTING algorithm for {ds}" + (f" (region '{region_id}')" if region_id else ""))
            for resp_var in resp_vars:
                min_lim = None
                max_lim = None
                metric_bounds = rafts_catg_uncn[rafts_catg_uncn['var'] == resp_var]
                if not metric_bounds.empty:
                    min_lim = metric_bounds['min_lim'].iloc[0]
                    max_lim = metric_bounds['max_lim'].iloc[0]
                    logging.warning(f"   Applying bounds for '{resp_var}': min={min_lim}, max={max_lim}")
                else:
                    logging.warning(f"   No bounds found for '{resp_var}'. Predictions will not be clipped.")

                # --- DYNAMIC ALGORITHM DISCOVERY ---
                dynamic_algos = raftsutil.discover_dynamic_algos(
                    search_dir=dir_out_alg_ds,
                    base_algos=algos,
                    metric=resp_var,
                    dataset_id=ds,
                    file_prefix="algo_",
                    file_extension=".joblib"
                )

                if not dynamic_algos:
                    logging.warning(f"No trained models found for {resp_var} in {dir_out_alg_ds}. Skipping.")
                    continue

                logging.info(f"Dynamically discovered algorithms for {resp_var}: {dynamic_algos}")
                # ----------------------------------------

                for algo in dynamic_algos:
                    path_algo = raftsutil.std_algo_path(dir_out_alg_ds, algo=algo, metric=resp_var, dataset_id=ds)

                    # --- Pipeline loading and validation ---
                    pipeline_data = raftsutil.load_validated_pipeline(path_algo, arg_val=arg_val)

                    pipe = pipeline_data['pipeline']
                    X_train_shape = pipeline_data['X_train_shape']  # Retrieve X_train.shape


                    feat_names = list(pipe.feature_names_in_)
                    df_attr_sub = df_attr_wide_region[feat_names]


                    # Force the index to strings to prevent join() TypeErrors
                    if not pd.api.types.is_string_dtype(df_attr_sub.index.dtype):
                        warn_str = f"The {df_attr_sub.index.name} index is expected to be a str dtype!! Check input data, especially if they are USGS gage ids!!!"
                        logging.warning(warn_str)
                        print(warn_str)
                        df_attr_sub.index = df_attr_sub.index.astype(str)

                    # Remove na values
                    df_attr_sub_rmna = df_attr_sub.dropna()
                    if df_attr_sub_rmna.shape[0] < df_attr_sub.shape[0]:
                        # Note - if error raised here about int dtype, then file read needs to be fixed to ensure str
                        ids_na = set(df_attr_sub.index) - set(df_attr_sub_rmna.index)
                        text_join = '\n'.join(ids_na)
                        msg_rm_na = f"Removing the following featureIDs from prediction due " + \
                         f"to NA values:\n{text_join}"
                        logging.warning(msg_rm_na)

                    # Perform prediction
                    resp_pred = pipe.predict(df_attr_sub_rmna)
                    if task_type != 'clustering':
                        # Unconditionally warn if any predictions fall out of the physical range.
                        raftsutil._warn_if_out_of_bounds(
                            predictions=resp_pred,
                            feature_ids=df_attr_sub_rmna.index,
                            min_lim=min_lim,
                            max_lim=max_lim,
                            resp_var=resp_var,
                            correction_is_active=uncn_bnd_pred,
                            prediction_type="values"
                        )
                        # --- Apply bounds to the primary prediction value (resp_pred) ---
                        if uncn_bnd_pred:
                            resp_pred = raftsutil.clip_predictions(resp_pred, min_lim, max_lim)

                    # Initialize DataFrame for storing results
                    df_pred = pd.DataFrame({'featureID': df_attr_sub_rmna.index, 'prediction': resp_pred, 'resp_var': resp_var, 'dataset': ds, 'algo': algo, 'name_algo': Path(path_algo).name})

                    # If using RandomForest, calculate confidence intervals using forestci
                    if algo == 'rf' and forestci:
                        rf_algo = pipe.named_steps['randomforestregressor']  # Use the correct step name
                        forest_ci = fci.random_forest_error(forest=rf_algo, X_train_shape=X_train_shape, X_test=df_attr_sub_rmna.to_numpy())
                        df_pred['forestci'] = forest_ci

                    # If MAPIE is available, compute prediction intervals
                    mapie_alpha = validated_pred_cfg.mapie_alpha
                    if 'mapie' in pipeline_data and mapie_alpha:
                        mapie = pipeline_data['mapie']
                        y_pred_mapie, y_pis = mapie.predict(df_attr_sub_rmna, alpha=mapie_alpha)

                        raftsutil._warn_if_out_of_bounds(
                            predictions=y_pis,
                            feature_ids=df_attr_sub_rmna.index,
                            min_lim=min_lim,
                            max_lim=max_lim,
                            resp_var=resp_var,
                            correction_is_active=uncn_bnd_pred,
                            prediction_type="intervals"
                        )
                        # Apply bounds if the flag was set and bounds were found
                        if uncn_bnd_pred and (min_lim is not None or max_lim is not None):
                            y_pis = raftsutil.clip_pis(y_pis, min_lim, max_lim)

                        # Rename columns based on self.mapie_alpha values
                        for i, alpha in enumerate(mapie_alpha):
                            df_pred[f'mapie_lower_{alpha:.2f}'] = y_pis[:, 0, i]
                            df_pred[f'mapie_upper_{alpha:.2f}'] = y_pis[:, 1, i]
                    elif mapie_alpha and 'mapie' not in pipeline_data:
                        logging.warning("MAPIE prediction interval estimation is not available in the " \
                        "trained algorithm pipeline, but mapie_alpha is specified in the prediction config file." \
                        "If prediction uncertainty desired, re-run the algorithm training rafts_proc_algo_viz.py, " \
                        "with mapie specified in the Uncertainty section of the algo config file.")

                    path_pred_out = raftsutil.std_pred_path(dir_out,algo=algo,metric=resp_var,dataset_id=ds)

                    # Update prediction file with the featureID-featureSource mapping (and NA-filling)
                    df_pred_mrge = pd.merge(df_pred, map_feat_srce_feat_id, how='right', on='featureID')
                    df_pred_mrge.fillna(value={'resp_var': resp_var,'dataset': ds, 'algo': algo,
                                                'name_algo':Path(path_algo).name},inplace=True)

                    col_order = ['featureID', 'featureSource', 'prediction']

                    # Find and sort any uncertainty columns that were added to the dataframe
                    uncertainty_cols = sorted([
                        col for col in df_pred_mrge.columns
                        if col.startswith('forestci') or col.startswith('mapie_')
                    ])

                    # Define the trailing metadata columns
                    meta_cols = ['resp_var', 'dataset', 'algo', 'name_algo']

                    # Combine all column lists for the final order
                    final_col_order = col_order + uncertainty_cols + meta_cols

                    # Reorder the dataframe
                    df_pred_mrge = df_pred_mrge[final_col_order]

                    # --- Validation and file writing of the output dataframe ---
                    raftsutil.write_validated_prediction_output(
                        df_pred_mrge=df_pred_mrge,
                        path_pred_out=path_pred_out,
                        arg_val=arg_val,
                        valid_metrics=resp_vars,
                        mapie_alpha=mapie_alpha
                    )
                    logging.info(f"Wrote {df_pred_mrge.shape[0]} predictions to {path_pred_out}")
                    logging.info(f"   Completed {algo} prediction of {resp_var}")
    logging.info(f"FINISHED algorithm prediction for {path_pred_config.name}")
    logging.shutdown()