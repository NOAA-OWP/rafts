"""Workflow script to train algorithms on catchment attribute data for predicting
    formulation metrics and/or hydrologic signatures.

Example:
    >>> python rafts_proc_algo_viz.py "/path/to/algo_config.yaml"

Changelog/Contributions
2024 Originally created, GL
2025-05-19 oconus refactor using hard-coded default col_locid = 'featureID'
 in lieu of 'comid; add try/except based on read_type, GL
2025-08-21 implement logging, GL
2025-11-19 Integrated pandera schema validation, [Soroush Sorourian/AI]
2025-11-25 refactor: move schemas to package structure and adjust import logic, [Soroush Sorourian/AI]
2025-11-26 feat: Dynamically load valid metrics from associated prep config file, [Soroush Sorourian/AI]
2025-11-26 refactor: Moved dynamic metric loading logic to rafts_algo.utils.get_valid_metrics, [Soroush Sorourian/AI]
2025-12-01 refactor: Consolidated all validation and I/O into rafts_algo.utils functions, [Soroush Sorourian/AI]
2025-12-08 refactor: Updated dynamic metric loading logic to be retrieved from the algo config, Soroush Sorourian
2026-10-03 feat: Add optional sub-regional training (see rafts_algo.regions), mirroring
           rafts_proc_algo_pool.py; CONUS-wide behavior (no `regions:` config) is unchanged, GL
"""
import argparse
import pandas as pd
from pathlib import Path
import rafts_algo.rafts_algo_train as raftsalgt
import rafts_algo.regions as raftsregions
import rafts_algo.utils as raftsutil
import rafts_algo.plots as raftsplot
import rafts_algo.schemas.schemas as schemas
import numpy as np
import os
import matplotlib
matplotlib.use(os.environ.get("MPLBACKEND", "Agg"))
import matplotlib.pyplot as plt
import xarray as xr
import rafts_prep.proc_eval_metrics as pem
from logging.handlers import MemoryHandler
import logging
import sys
from sklearn.inspection import permutation_importance

import yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'process the algorithm config file')
    parser.add_argument('path_algo_config', type=str,
                        help='Path to the YAML configuration file specific for algorithm training')
    parser.add_argument('--validate', action='store_true', default=False,
                        help='If present, enables schema validation for all input and output data. Defaults to False.')
    parser.add_argument('--region', type=str, default=None,
                        help="Restrict this run to a single configured sub-region's region_id. "
                             "Omit to loop over every region the algo config's `regions:` block "
                             "resolves to (or run once, unregioned, if `regions:` isn't set).")
    args = parser.parse_args()

    path_algo_config = Path(args.path_algo_config).expanduser() #Path(f'~/git/rafts/scripts/workflow_configs/legacy/xssa/xssa_algo_config.yaml').expanduser()
    config_dir = path_algo_config.parent

    # --- Conditionally load schemas
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
    logging.info(f"Running rafts_proc_algo_viz.py with \
                {path_algo_config.parent / path_algo_config.name} config file")

    # ---
    logging.info("BEGINNING algorithm training, testing, & evaluation.")
    # Initialize algo configuration class for extracting attributes
    algo_cfig = raftsutil.AlgoConfigParser(path_algo_config)
    algo_cfig._read_algo_config()

    # Extract variables from dictionary created by AlgoConfigParser
    algo_config = algo_cfig.algo_cfg_unc_dict["algo_cfg_dict"]["algo_config"]

    # Generate variable algo_config_og
    algo_config_og = algo_config.copy()

    verbose = algo_cfig.algo_cfg_unc_dict["algo_cfg_dict"]["verbose"]
    test_size = algo_cfig.algo_cfg_unc_dict["algo_cfg_dict"]["test_size"]
    seed = algo_cfig.algo_cfg_unc_dict["algo_cfg_dict"]["seed"]
    read_type = algo_cfig.algo_cfg_unc_dict["algo_cfg_dict"]["read_type"] # Arg for how to read attribute data using comids in rafts_read_attr_comid(). May be 'all' or 'filename'.
    metrics = algo_cfig.algo_cfg_unc_dict["algo_cfg_dict"]["metrics"]
    make_plots = algo_cfig.algo_cfg_unc_dict["algo_cfg_dict"]["make_plots"]
    same_test_ids = algo_cfig.algo_cfg_unc_dict["algo_cfg_dict"]["same_test_ids"]
    path_attr_config = algo_cfig.algo_cfg_unc_dict["algo_cfg_dict"]["path_attr_config"]
    uncertainty_cfg = algo_cfig.algo_cfg_unc_dict["algo_unc_dict"]["uncertainty_cfg"]
    confidence_levels = algo_cfig.algo_cfg_unc_dict["algo_unc_dict"]["uncertainty_cfg"].get("confidence_levels")
    uncn_bnd_algo = algo_cfig.algo_cfg_unc_dict["algo_unc_dict"]["uncertainty_cfg"].get("uncn_bnd_algo",False)
    regions_cfg_raw = algo_cfig.algo_cfg_unc_dict["algo_cfg_dict"].get("regions")  # raw dict or None

    #%% Attribute configuration
    # Initialize attribute configuration class for extracting attributes
    attr_cfig = raftsutil.AttrConfigAndVars(path_attr_config)
    attr_cfig._read_attr_config()

    # READ rafts_categories.yaml
    logging.info("Reading uncertainty bounds from rafts_categories.yaml...")

    rafts_catg_uncn = pem._conv_ls_dicts_df_long()
    logging.info("Successfully loaded uncertainty bounds.")

    # Grab the attributes of interest from the attribute config file,
    #  OR a .csv file if specified in the algo config file.
    name_attr_csv = algo_cfig.algo_cfg_unc_dict["algo_cfg_dict"]["name_attr_csv"]
    colname_attr_csv = algo_cfig.algo_cfg_unc_dict["algo_cfg_dict"]["colname_attr_csv"]
    attrs_sel = raftsutil.read_validated_attribute_selection(
        attr_cfig=attr_cfig,
        path_cfig=path_algo_config,
        name_attr_csv=name_attr_csv,
        colname_attr_csv=colname_attr_csv,
        arg_val=arg_val
    )

    # Define directories/datasets from the attribute config file
    dir_db_attrs = attr_cfig.attrs_cfg_dict.get('dir_db_attrs')
    dir_std_base = attr_cfig.attrs_cfg_dict.get('dir_std_base')
    dir_base = attr_cfig.attrs_cfg_dict.get('dir_base')
    datasets = attr_cfig.attrs_cfg_dict.get('datasets') # Identify datasets of interest
    # Grab variables for building out the path to metadata (which contains comid-gage id mappings)
    ds_type = [x for x in attr_cfig.attr_config.get('file_io') if 'ds_type' in x][0]['ds_type']
    path_meta_fstr = [x for x in attr_cfig.attr_config.get('file_io') if 'path_meta' in x][0]['path_meta']


    # ---------- Generate path to the log file & initialize logging -----------
    path_log = pem.std_path_log(dir_input=dir_base,
                                path_config=path_algo_config,
                            script='rafts_proc_algo_viz')
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
    # Define identifier schemas from attribute config:
    featureID = attr_cfig.attr_config.get('col_schema')[0].get('featureID')
    featureSource = attr_cfig.attr_config.get('col_schema')[0].get('featureSource')
    col_locid = 'featureID' # Enforcing the default column name expected throughout, 'featureID'.
    if same_test_ids:
        # Must first establish which comids to use in the train-test split
        split_dict = raftsutil.split_train_test_comid_wrap(dir_std_base=dir_std_base,
                    datasets=datasets, path_attr_config=path_attr_config,
                    id_col=col_locid, test_size=test_size,
                    random_state=seed)
        # If we use all the same comids for testing, we can make inter-comparisons. This ONE
        # global split is computed before any region loop, and the region loop below only
        # ever *subsets* it -- see rafts_proc_algo_pool.py's identical pattern.
        test_ids_global = split_dict.get('sub_test_ids',None) #If this returns None, we use the test_size for all data
        dict_gdf_comids = split_dict.get('dict_gdf_comids',None) # retrieve the featureID - gage_id - geometry mapping
        # TODO PROBLEM: The raftsutil.rafts_read_attr_comid step can reduce the total number of comids for consideration if data are missing. Thus test_ids would need to be revised
    else:
        test_ids_global = None
        # retrieve the featureID - gage_id - geometry mapping

    # %% Looping over regions, then over datasets within each region -- a `regions:` block
    # that isn't configured resolves to a single implicit (None, None) region (see
    # rafts_algo.regions.resolve_region_loop), so every line below behaves exactly as it did
    # before regions existed in that case.
    region_loop = raftsregions.resolve_region_loop(
        regions_cfg_raw, args.region,
        context={'home_dir': attr_cfig.attrs_cfg_dict.get('home_dir'), 'dir_base': dir_base})
    region_divide_id_col = (regions_cfg_raw or {}).get('divide_id_col', 'divide_id')
    region_min_train_gages = (regions_cfg_raw or {}).get('min_train_gages', 1)
    region_fallback = (regions_cfg_raw or {}).get('fallback', 'skip')
    region_scheme = (regions_cfg_raw or {}).get('scheme') if regions_cfg_raw else None

    for region_id, region_spec in region_loop:
        dirs_std_dict = raftsutil.rafts_save_algo_dir_struct(dir_base, region=region_id, scheme=region_scheme)
        dir_out = dirs_std_dict.get('dir_out')
        dir_out_alg_base = dirs_std_dict.get('dir_out_alg_base')
        dir_out_anlys_base = dirs_std_dict.get('dir_out_anlys_base')
        dir_out_viz_base = dirs_std_dict.get('dir_out_viz_base')

        test_ids = test_ids_global  # reset to the global split at the start of each region's pass

        # %% Looping over datasets
        for ds in datasets:
            logging.info(f'PROCESSING {ds} dataset inside \n {dir_std_base}' +
                         (f' for region {region_id}' if region_id else ''))

            dir_out_alg_ds = Path(dir_out_alg_base/Path(ds))
            dir_out_alg_ds.mkdir(exist_ok=True)

            vals = {'ds_type':ds_type,'write_type':'parquet', 'dir_std_base':dir_std_base,'ds':ds}
            path_meta = path_meta_fstr.format(**vals)
            dir_db_attrs = Path(str(dir_db_attrs).format(**vals))
            if Path(path_meta).exists() and False:
                # TODO allow secondary option where dat_resp and metrics read in from elsewhere.
                # NOTE dataset metadata handling will also need to be considered
                # TODO first check for comids from metadata in the path_attr_config file
                if 'parquet' in Path(path_meta).suffix:
                    df_meta = pd.read_parquet(path_meta)
                elif 'csv' in Path(path_meta).suffix:
                    df_meta = pd.read_csv(path_meta)

                # Select the unique metadata:
                df_meta_uniq = df_meta[['featureSource',col_locid,'gage_id']].drop_duplicates().set_index('gage_id')
                # Read in the original data
                dat_resp = raftsutil._open_response_data_rafts(dir_std_base,ds)

                # Add the featureSource/featureID to the xr dataset's data vars
                for coord in ['featureSource', col_locid]:
                    dat_resp[coord] = (('gage_id'), df_meta_uniq[coord])

                locids_resp = dat_resp[col_locid].data
                # TODO add gdf_comid based on lat/lon
            else:
                # Read in the standardized dataset generated by rafts_prep & grab comids/coords
                dict_resp_gdf = raftsutil.combine_resp_gdf_comid_wrap(dir_std_base=dir_std_base,
                                ds= ds, path_attr_config=path_attr_config)
                dat_resp = dict_resp_gdf['dat_resp'] # TODO why is dat_resp len 36 when it should be 44 with benchmarking FY25?
                gdf_comid = dict_resp_gdf['gdf_comid']

                # Snapshot the full one-row-per-(gage_id, divide_id) structure BEFORE the
                # subset below collapses it -- region-aware gage eligibility needs every
                # basin divide a gage has. See rafts_algo.regions.assign_gages_by_basin_divides
                # and rafts_proc_algo_pool.py's identical use of this structure.
                gdf_comid_basin_rows = gdf_comid

                # Subset to the gage ids only selected for training (just in case some predictions make it into dat_resp)
                gdf_comid = gdf_comid[gdf_comid['gage_id'].astype(str).isin(dat_resp['gage_id'].values)]
                dat_resp["comid"] = (("gage_id"), gdf_comid["comid"].astype(str).values)

                # --- VALIDATION: GDF Comid ---
                raftsutil.validate_gdf_comid_schema(gdf_comid, arg_val=arg_val)

                locids_resp = gdf_comid[col_locid].tolist()

            # === Region scoping: restrict training/donor data to this region's core+buffer ===
            model_scope = 'conus'
            if region_spec is not None:
                eligible_gage_ids = raftsregions.donor_gage_ids_for_region(
                    gdf_comid_basin_rows, region_spec, gage_id_col=col_locid,
                    divide_id_col=region_divide_id_col)
                gdf_comid_region = gdf_comid[gdf_comid[col_locid].astype(str).isin(eligible_gage_ids)]

                if len(gdf_comid_region) < region_min_train_gages:
                    eligible_fn = lambda r: raftsregions.donor_gage_ids_for_region(
                        gdf_comid_basin_rows, r, gage_id_col=col_locid, divide_id_col=region_divide_id_col)
                    gdf_comid_region, region_spec, model_scope = raftsregions.apply_min_train_gages_fallback(
                        region=region_spec, gdf_donors_core_buffer=gdf_comid_region,
                        gdf_donors_global=gdf_comid, min_train_gages=region_min_train_gages,
                        fallback=region_fallback, id_col=col_locid, eligible_ids_fn=eligible_fn)
                    if gdf_comid_region is None:  # fallback == 'skip'
                        logging.warning(f"Region '{region_id}'/{ds}: fewer than "
                                        f"{region_min_train_gages} training gages available; "
                                        f"skipping (fallback='skip').")
                        continue
                else:
                    model_scope = 'region'

                gdf_comid = gdf_comid_region
                locids_resp = gdf_comid[col_locid].tolist()
                raftsregions.write_model_scope_sidecar(dir_out_alg_ds, region_id, model_scope)

                if isinstance(test_ids, pd.Series):
                    core_gage_ids = raftsregions.donor_gage_ids_for_region_core(
                        gdf_comid_basin_rows, region_spec, gage_id_col=col_locid,
                        divide_id_col=region_divide_id_col)
                    test_ids = gdf_comid[col_locid][
                        gdf_comid[col_locid].astype(str).isin(test_ids.astype(str))
                        & gdf_comid[col_locid].astype(str).isin(core_gage_ids)]
            # === end region scoping ===

            if not metrics:
                # The metrics approach. These are all xarray data variables of the response(s)
                metrics = dat_resp.attrs['respvar_mappings'].split('|')

            # --- VALIDATION: Response Data (dat_resp) ---
            raftsutil.validate_dat_resp_schema(dat_resp, metrics, col_locid, arg_val)

            #%%  Read in predictor variable data (aka basin attributes) & NA removal
            # Read the predictor variable data (basin attributes) generated by proc.attr.hydfab
            # NOTE some gage_ids lost inside rafts_read_attr_comid.
            try:
                df_attr = raftsutil.rafts_read_attr_comid(dir_db_attrs, locids_resp, attrs_sel = attrs_sel,
                                                _s3 = None,storage_options=None,read_type=read_type)
            except: # The read_type='filename' approach may not work
                df_attr = raftsutil.rafts_read_attr_comid(dir_db_attrs, locids_resp, attrs_sel = attrs_sel,
                                    _s3 = None,storage_options=None,read_type='all')

            # --- VALIDATION: Attribute Data (df_attr) ---
            raftsutil.validate_input_attributes(df_attr,arg_val=arg_val)

            # Convert into wide format for model training
            try:
                df_attr_wide = df_attr.pivot(index=col_locid, columns = 'attribute', values = 'value')
            except:
                logging.error("Could not convert to long format. A common culprit is duplicated data, perhaps un-detected due to" \
                " 1) multiple sub-directories containing data inside dir_db_attrs" \
                " 2) data across all standard columns are the same but the dl_timestamp differs.")
                sys.exit(1)
            comids_df_attr_wide = df_attr_wide.index.values

            # Prepare attribute correlation matrix w/o NA values (writes to file)
            if df_attr_wide.isna().any().any(): #
                df_attr_wide_dropna = df_attr_wide.dropna()
                locids_with_na_attrs = [x for x in df_attr_wide.index if x not in df_attr_wide_dropna.index]
                logging.warning(f"Dropping {df_attr_wide.shape[0] - df_attr_wide_dropna.shape[0]} total locations from analysis \
                for correlation/PCA assessment due to NA values, reducing dataset to {df_attr_wide_dropna.shape[0]} points")
                missing_locs_str = '\n'.join(locids_with_na_attrs)
                logging.warning(f"Locations with missing attribute data include:\n{missing_locs_str}")
                frac_na = (df_attr_wide.shape[0] - df_attr_wide_dropna.shape[0])/df_attr_wide.shape[0]
                if frac_na > 0.1:
                    logging.warning(f"!!!!{np.round(frac_na*100,1)}%  of data are NA values and will be discarded before training/testing!!!!")
            else:
                df_attr_wide_dropna = df_attr_wide.copy()
            # ---------  UPDATE gdf and comid list after possible data removal ---------- #
            # Data removal comes from from raftsutil.rafts_read_attr_comid & df_attr_wide.dropna():
            remn_comids = list(df_attr_wide_dropna.index) # these are the comids that are left after checking what data are available
            # Revise gdf_comid
            gdf_comid = gdf_comid[gdf_comid[col_locid].isin(remn_comids)].reset_index()

            if isinstance(test_ids,pd.Series): # Revise test_ids
                # This resets the index of test_ids to correspond with gdf_comid
                test_ids = gdf_comid[col_locid][gdf_comid[col_locid].isin(test_ids)]

            #%% Characterize dataset correlations & principal components:
            fig_corr_mat = raftsplot.plot_corr_mat_save_wrap(df_X=df_attr_wide_dropna,
                                        title=f'Correlation matrix from {ds} dataset',
                                        dir_out_viz_base=dir_out_viz_base,
                                        ds=ds)
            plt.clf()
            # Attribute correlation results based on a correlation threshold (writes to file)
            df_corr_rslt = raftsplot.corr_thr_write_table_wrap(df_X=df_attr_wide_dropna,
                                                           dir_out_anlys_base=dir_out_anlys_base,
                                                           ds = ds,
                                                           corr_thr=0.8)


            # Principal component analysis
            pca_rslt = raftsplot.plot_pca_save_wrap(df_X=df_attr_wide_dropna,
                            dir_out_viz_base=dir_out_viz_base,
                            ds = ds,
                            std_scale=True # Apply the StandardScaler.
                            )
            plt.clf()
            # %% Train, test, and evaluate
            task_type = algo_cfig.algo_cfg_unc_dict["algo_cfg_dict"].get("task_type", "regression")
            # Override metrics if clustering (we don't need real metrics)
            if task_type == 'clustering':
                metrics = ['cluster_labels']

            rslt_eval = dict()
            for metr in metrics:
                logging.info(f' - Processing {metr}')
                if len(algo_config) == 0:
                    algo_config = algo_config_og.copy()
                if task_type == 'clustering':
                    # Bypass dat_resp completely. Just use attributes!
                    df_pred_resp = df_attr_wide_dropna.reset_index()
                else:
                    # Subset response data to metric of interest & the comid
                    df_metr_resp = pd.DataFrame({col_locid: dat_resp[col_locid],
                                                'featureSource': dat_resp['featureSource'],
                                                metr : dat_resp[metr].data})
                    # Join attribute data and response data
                    df_pred_resp = df_metr_resp.merge(df_attr_wide_dropna, left_on = col_locid, right_on = col_locid)

                    if df_pred_resp.isna().any().any(): # Check for NA values and remove them if present to avoid errors during evaluation
                        tot_na_dfpred = df_pred_resp.shape[0] - df_pred_resp.dropna().shape[0]
                        pct_na_dfpred = tot_na_dfpred/df_pred_resp.shape[0]*100
                        logging.info(f"Removing {tot_na_dfpred} NA values, which is {pct_na_dfpred}% of total data")
                        df_pred_resp = df_pred_resp.dropna()
                        if pct_na_dfpred > 10:
                            logging.warning(f"!!!!More than 10% of data are NA values!!!!")

                    # TODO may need to add additional distinguishing strings to dataset_id, e.g. in cases of probabilistic simulation

                    # GET MIN/MAX BOUNDS FOR THE CURRENT METRIC
                    min_lim = None
                    max_lim = None
                    metric_bounds = rafts_catg_uncn[rafts_catg_uncn['var'] == metr]
                    if not metric_bounds.empty:
                        min_lim = metric_bounds['min_lim'].iloc[0]
                        max_lim = metric_bounds['max_lim'].iloc[0]
                        logging.warning(f"   Applying bounds for '{metr}': min={min_lim}, max={max_lim}")
                    else:
                        logging.warning(f"   No bounds found for '{metr}'. Predictions will not be clipped.")

                    # Instantiate the training, testing, and evaluation class
                    train_eval = raftsalgt.AlgoTrainEval(df=df_pred_resp,
                                                attrs=attrs_sel,
                                                algo_config=algo_config,
                                                uncertainty=uncertainty_cfg,
                                                dir_out_alg_ds=dir_out_alg_ds, dataset_id=ds,
                                                metr=metr,task_type=task_type,
                                                test_size=test_size, rs = seed,
                                                test_id_col=col_locid,
                                                verbose=verbose,
                                                confidence_levels=confidence_levels,
                                                uncn_bnd_algo=uncn_bnd_algo,
                                                min_lim=min_lim,
                                                max_lim=max_lim
                                                )
                    train_eval.train_eval() # Train, test, eval wrapper

                    # Retrieve evaluation metrics dataframe & write to file
                    rslt_eval[metr] = train_eval.eval_df
                    path_eval_metr = raftsutil.std_eval_metrs_path(dir_out_viz_base, ds,metr)
                    train_eval.eval_df.to_csv(path_eval_metr)

                    #%% Random Forest Feature Importance
                    y_test = train_eval.y_test
                    df_X, y_all = train_eval.all_X_all_y()

                    if make_plots:
                        out_dir = Path(dir_out_viz_base) / ds

                        # Iterate through all trained models to plot feature importances dynamically
                        for algo_str, algo_info in train_eval.algs_dict.items():
                            model = algo_info['algo']
                            imp_train = getattr(model, "feature_importances_", None)

                            if imp_train is None:
                                # Calculate training permutation importance (requires X and y data)
                                try:
                                    result = permutation_importance(
                                        estimator=model,
                                        X=train_eval.X_train,
                                        y=train_eval.y_train,
                                        n_repeats=5,
                                        random_state=seed,
                                        n_jobs=-1 # Uses all processors to speed up the calculation
                                    )

                                    # Extract the mean importances to match the 1D array format of .feature_importances_
                                    imp_train = result.importances_mean

                                    path_feat_imp_train = raftsplot.std_feat_imp_path(dir_out_viz_base=dir_out_viz_base,
                                                                                        ds=ds, algo_str=f"{algo_str}_train", metr=metr)
                                    out_csv_tr = out_dir / f"{algo_str}_feature_importance_train_{ds}_{metr}.csv"
                                    fi_df.to_csv(out_csv_tr, index=False)
                                    logging.info(f"Wrote {algo_str} feature importances to {out_csv}")

                                    fig_train = raftsplot.plot_feature_importance(
                                                    feat_imprt=imp_train,
                                                    attrs=df_X.columns,
                                                    title=f"{algo_str.upper()} Training Feature Importance: {ds}"
                                                    )

                                    fig_train.savefig(path_feat_imp_train, bbox_inches='tight')
                                    plt.close(fig_train)


                                except:
                                    logging.info(f"The {algo_str} does not estimate feature importance")

                            else:
                                # Save features importances from the trained model to csv files
                                fi_df = pd.DataFrame({"feature": df_X.columns, "importance": imp_train})
                                fi_df = fi_df.sort_values("importance", ascending=False)
                                out_csv = out_dir / f"{algo_str}_feature_importance_train_{ds}_{metr}.csv"
                                fi_df.to_csv(out_csv, index=False)
                                logging.info(f"Wrote {algo_str} feature importances to {out_csv}")

                                # Plot PNG of Feature Importance using the generalized wrapper
                                raftsplot.save_feat_imp_fig_wrap(
                                    model=model,
                                    attrs=df_X.columns,
                                    dir_out_viz_base=dir_out_viz_base,
                                    ds=ds,
                                    metr=metr,
                                    algo_str=f"{algo_str}_Training"
                                )

                            try: # Now create the feature importance plot on test data:
                                result_test = permutation_importance(
                                    estimator=model,
                                    X=train_eval.X_test,
                                    y=train_eval.y_test,
                                    n_repeats=5,
                                    random_state=seed,
                                    n_jobs=-1
                                )
                                imp_test = result_test.importances_mean

                                path_feat_imp_test = raftsplot.std_feat_imp_path(dir_out_viz_base=dir_out_viz_base,
                                                            ds=ds, algo_str=f"{algo_str}_test", metr=metr)
                                out_viz_dir = path_feat_imp_test.parent

                                # --- Save and Plot Testing Importances ---
                                fi_test_df = pd.DataFrame({"feature": df_X.columns, "importance": imp_test}).sort_values("importance", ascending=False)
                                fi_test_csv = out_viz_dir / f"{algo_str}_feature_importance_test_{ds}_{metr}.csv"
                                fi_test_df.to_csv(fi_test_csv, index=False)

                                fig_test = raftsplot.plot_feature_importance(
                                    feat_imprt=imp_test,
                                    attrs=df_X.columns,
                                    title=f"{algo_str.upper()} Testing Feature Importance: {ds}"
                                )

                                fig_test.savefig(path_feat_imp_test, bbox_inches='tight')
                                plt.close(fig_test)

                                logging.info(f"Wrote {algo_str} test feature importances to {out_viz_dir}")
                            except Exception as e:
                                    logging.info(f"The {algo_str} does not estimate feature importance. Error: {e}")


                        # Create learning curves for each algorithm
                        algo_plot_lc = raftsalgt.AlgoEvalPlotLC(df_X,y_all)
                        raftsalgt.plot_learning_curve_save_wrap(algo_plot_lc,train_eval,
                                        dir_out_viz_base=dir_out_viz_base,
                                        ds=ds,
                                        cv = 5,n_jobs=-1,
                                        train_sizes = np.linspace(0.1, 1.0, 10),
                                        scoring = 'neg_mean_squared_error',
                                        ylabel_scoring = "Mean Squared Error (MSE)",
                                        training_uncn = False
                                        )

                    # %% Model testing results visualization

                    # Initialize min and max errors
                    if make_plots:
                        # Calculate global min and max for consistent uncertainty scaling across all algorithms (but unique scaling for e/ response variable/metric)
                        min_err = float('inf')  # Initialize with a large value
                        max_err = float('-inf')  # Initialize with a small value
                        for algo_str in train_eval.algs_dict.keys():
                            if train_eval.preds_dict[algo_str].get('y_pis',None) is not None:
                                y_pred = train_eval.preds_dict[algo_str].get('y_pred',None)
                                y_pis = train_eval.preds_dict[algo_str].get('y_pis',None)

                                # Calculate the global min and max errors across all algorithms
                                for alpha_val in next(d['alpha'] for d in uncertainty_cfg.get('mapie', [])):
                                    lower_err = y_pred - np.array([y_pis[i].loc['lower_limit', f'alpha_{alpha_val:.2f}'] for i in range(len(y_pred))])
                                    upper_err = np.array([y_pis[i].loc['upper_limit', f'alpha_{alpha_val:.2f}'] for i in range(len(y_pred))]) - y_pred

                                    total_err = lower_err + upper_err  # Compute total error for this algorithm

                                    # Update global min and max across all algorithms
                                    min_err = min(min_err, total_err.min())
                                    max_err = max(max_err, total_err.max())

                    # ----- Extract y_pred for each algorithm -----
                    dict_test_gdf = dict()
                    for algo_str in train_eval.algs_dict.keys():

                        #%% Evaluation: learning curves
                        y_pred = train_eval.preds_dict[algo_str].get('y_pred')
                        y_obs = train_eval.y_test.values
                        r2_val = train_eval.eval_dict[algo_str].get('r2', None)

                        spearman_val = train_eval.eval_dict[algo_str].get('spearman', None)
                        if make_plots and task_type != 'clustering':
                            # Regression of testing holdout's prediction vs observation
                            if train_eval.preds_dict[algo_str].get('y_pis',None) is not None:
                                y_pis = train_eval.preds_dict[algo_str].get('y_pis')
                                for alpha_val in next(d['alpha'] for d in uncertainty_cfg.get('mapie', [])):
                                    raftsplot.plot_pred_vs_obs_wrap_mapie(y_pred, y_obs, dir_out_viz_base,
                                            ds, metr, algo_str=algo_str,
                                            y_pis = y_pis, alpha_val = alpha_val,
                                            split_type=f'testing{test_size}',r2_val=r2_val,spearman_val=spearman_val)
                            else:
                                raftsplot.plot_pred_vs_obs_wrap(y_pred, y_obs, dir_out_viz_base,
                                        ds, metr, algo_str=algo_str,split_type=f'testing{test_size}',r2_val=r2_val,spearman_val=spearman_val)

                        # PREPARE THE GDF TO ALIGN PREDICTION VALUES BY COMIDS/COORDS
                        # Get the comids corresponding to the testing data/run QA checks
                        comids_test = train_eval.df[col_locid].iloc[train_eval.X_test.index].values
                        test_gdf = gdf_comid[gdf_comid[col_locid].isin(comids_test)].copy()
                        # The comid-y_pred/y_obs mapping:
                        if task_type == 'clustering':
                            df_test = pd.DataFrame({col_locid: comids_test, 'observed': np.nan})
                        else:
                            df_test = train_eval.df.iloc[train_eval.y_test.index][[col_locid,metr]].rename(columns={metr:'observed'})
                        df_test['prediction'] = y_pred

                        # Merge the test_gdf with the prediction dataframe
                        test_gdf = test_gdf.merge(df_test, left_on=col_locid, right_on=col_locid, how='left')

                        # Add details on dataset, response variable, and algorithm
                        test_gdf.loc[:,'dataset'] = ds
                        test_gdf.loc[:,'metric'] = metr
                        test_gdf.loc[:,'algo'] = algo_str
                        test_gdf.loc[:,'region_id'] = region_id if region_id else 'CONUS'
                        test_gdf.loc[:,'model_scope'] = model_scope

                        test_gdf.drop_duplicates(subset=[col_locid,'observed','prediction'],inplace=True)

                        dict_test_gdf[algo_str] = test_gdf
                        if make_plots:
                            raftsplot.plot_map_pred_wrap(test_gdf,
                                            dir_out_viz_base, ds,
                                                metr,algo_str,
                                                split_type='test',
                                                colname_data='prediction',
                                                epsg_reproj=4326,
                                                task_type=task_type)

                            # %% Test Prediction Uncertainty Plotting
                            for algo_str in train_eval.algs_dict.keys():
                                if train_eval.preds_dict[algo_str].get('y_pis',None) is not None:
                                    y_pred = train_eval.preds_dict[algo_str].get('y_pred',None)
                                    y_pis = train_eval.preds_dict[algo_str].get('y_pis',None)
                                                    # Calculate the global min and max errors across all algorithms
                                    for alpha_val in next(d['alpha'] for d in uncertainty_cfg.get('mapie', [])):
                                        # --- Flatten MAPIE intervals into columns for the unified plotter ---
                                        test_gdf[f'mapie_lower_{alpha_val:.2f}'] = [y_pis[i].loc['lower_limit', f'alpha_{alpha_val:.2f}'] for i in range(len(y_pis))]
                                        test_gdf[f'mapie_upper_{alpha_val:.2f}'] = [y_pis[i].loc['upper_limit', f'alpha_{alpha_val:.2f}'] for i in range(len(y_pis))]
                                        raftsplot.plot_map_pred_wrap_uncn(
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

                    # Generate analysis path out:
                    path_pred_obs = raftsutil.std_test_pred_obs_path(dir_out_anlys_base,ds, metr)
                    # TODO why does test_gdf end up with a size larger than total comids? Should be the split test amount
                    df_pred_obs_ds_metr = pd.concat(dict_test_gdf)
                    df_pred_obs_ds_metr.to_csv(path_pred_obs)
                    logging.info(f"Wrote the prediction-observation-coordinates dataset to file\n{path_pred_obs}")

                    del train_eval
                # Compile results and write to file
                rslt_eval_df = pd.concat(rslt_eval).reset_index(drop=True)

                # --- VALIDATION and file writing: Result Eval DF ---
                raftsutil.write_validated_evaluation_output(
                    rslt_eval_df=rslt_eval_df,
                    dir_out_alg_ds=dir_out_alg_ds,
                    ds=ds,
                    valid_metrics=metrics,
                    arg_val=arg_val
                )

                dat_resp.close()
        #%% Cross-comparison across all datasets: determining where the best metric lives
        if same_test_ids and len(datasets)>1:
            logging.info("Cross-comparison across multiple datasets possible.\n"+
            f"Refer to custom script processing example inside scripts/analysis/rafts_proc_viz_best_ealstm.py")

    logging.info("FINISHED algorithm training, testing, & evaluation")
    logging.shutdown()
