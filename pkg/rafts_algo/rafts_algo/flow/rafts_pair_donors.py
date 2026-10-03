"""rafts_pair_donors.py

Following algorithm clustering predictions from rafts_pred_algo.py, the donor-receiver pairing
assigns a donor basin (used in training) to every receiver basin based on sharing the same
cluster and then finding the donor basin within each cluster at the nearest euclidean distance.

Changelog / Contributions
 2026-07-21 Originally created, GL with heavy consultation from Gemini3.1Pro
 2026-09-18 Refactored to integrate Pydantic PredConfig and AlgoConfig schema validation.
"""
import argparse
import pandas as pd
from pathlib import Path
import logging
from logging.handlers import MemoryHandler
import sys
import numpy as np

import rafts_algo.utils as raftsutil
import rafts_algo.regions as raftsregions
import rafts_algo.qa_utils as qa_utils
import rafts_prep.proc_eval_metrics as pem
import rafts_algo.rafts_algo_train as raftsat
import sqlite3

from rafts_algo.schemas.pydantic_schemas import PredConfig, AlgoConfig

"""
Workflow script to pair ungauged receiver basins with gauged donor basins
based on 1:1 attribute nearest-neighbor matching within machine learning clusters.

Usage:
    >>> python rafts_pair_donors.py "/path/to/datasetshortname_pred_config.yaml"
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process the prediction config file for donor pairing.')
    parser.add_argument('path_pred_config', type=str, help='Path to the prediction YAML config.')
    parser.add_argument('--region', type=str, default=None,
                        help="Restrict this run to a single configured sub-region's region_id. "
                             "Omit to loop over every region the pred config's `regions:` block "
                             "resolves to (or run once, unregioned, if `regions:` isn't set).")
    args = parser.parse_args()

    path_pred_config = Path(args.path_pred_config).expanduser()

    # --- Logging Setup ---
    memory_handler = MemoryHandler(capacity=30)
    root_logger = logging.getLogger()
    root_logger.addHandler(memory_handler)
    root_logger.setLevel(logging.INFO)
    logging.info(f"Running rafts_pair_donors.py with {path_pred_config.name}")

    # --- Pydantic Validation ---
    validated_pred_cfg = raftsutil.load_validated_config(path_pred_config, PredConfig)

    path_attr_config = raftsutil.build_cfig_path(path_pred_config, validated_pred_cfg.name_attr_config)
    path_algo_config = raftsutil.build_cfig_path(path_pred_config, validated_pred_cfg.name_algo_config)

    validated_algo_cfg = raftsutil.load_validated_config(path_algo_config, AlgoConfig)

    # --- Parse Configurations (Legacy parsing to retain f-string resolution) ---
    attr_cfig = raftsutil.AttrConfigAndVars(path_attr_config)
    attr_cfig._read_attr_config()

    # Extract validated top-level scalars
    task_type = validated_algo_cfg.task_type
    if task_type != 'clustering':
        logging.error("Donor pairing currently requires clustering predictions. Exiting.")
        sys.exit(1)

    dir_base = attr_cfig.attrs_cfg_dict.get('dir_base')
    dir_std_base = attr_cfig.attrs_cfg_dict.get('dir_std_base')
    dir_db_attrs = attr_cfig.attrs_cfg_dict.get('dir_db_attrs')
    datasets = attr_cfig.attrs_cfg_dict.get('datasets')
    home_dir = attr_cfig.attrs_cfg_dict.get('home_dir')
    
    id_col_pred = validated_pred_cfg.pred_file_comid_colname
    context = {
        'dir_base': str(dir_base),
        'dir_std_base': str(dir_std_base),
        'dir_db_attrs': str(dir_db_attrs),
        'home_dir': str(home_dir),
    }

    path_meta_raw = validated_pred_cfg.path_meta
    path_meta = Path(raftsutil.resolve_fstrings(path_meta_raw, context))
    if not path_meta.exists():
        logging.error(f"Training metadata missing: {path_meta}. Cannot identify donors.")


    regions_cfg = validated_pred_cfg.regions  # None unless this config sets a `regions:` block
    region_loop = raftsregions.resolve_region_loop(regions_cfg, args.region, context=context)
    region_divide_id_col = regions_cfg.divide_id_col if regions_cfg else 'divide_id'

    # Resolve Attributes using validated config
    name_attr_csv = validated_algo_cfg.name_attr_csv
    colname_attr_csv = validated_algo_cfg.colname_attr_csv
    attrs_sel = raftsutil._id_attrs_sel_wrap(attr_cfig=attr_cfig, path_cfig=path_attr_config, 
                                          name_attr_csv=name_attr_csv, colname_attr_csv=colname_attr_csv)

    path_log = pem.std_path_log(dir_input=dir_base, path_config=path_pred_config, script='rafts_pair_donors')
    logging.basicConfig(level=logging.INFO, filename=path_log, format='%(asctime)s - %(levelname)s - %(message)s', filemode='w', force=True)
    print(f"Logging to {path_log}")
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler):
            memory_handler.setTarget(handler)
            memory_handler.flush()
            break
    root_logger.removeHandler(memory_handler)

    resp_vars = validated_pred_cfg.algo_response_vars
    algos = validated_pred_cfg.algo_type

    # --- Processing ---
    for ds in datasets:

        logging.info(f"PROCESSING Donor-Receiver Pairing for dataset: {ds}")

        # 1. Load DONOR Attributes (Gauged Basins from Training)
        vals_train = {'ds_type': 'training', 'write_type': 'parquet', 'dir_std_base': dir_std_base, 'ds': ds}
        # Path to the training data (donors)
        dir_db_attrs = Path(str(dir_db_attrs).format(**vals_train))


        logging.info(f"Reading donor metadata from {path_meta}")
        df_meta = raftsutil.read_hfatlas_wrap_dask(
            paths_hfatl=[dir_db_attrs],
            attrs_sel=[], # Empty list forces it to only pull the map_id_col
            map_id_col='featureID'
        )
        # Read in the donor ids, meaning those that were trained.
        gageids_donor = df_meta['featureID'].drop_duplicates().tolist()

        # Load the donor response variables (.nc file) generated during prep
        dat_resp = raftsutil._open_response_data_rafts(dir_std_base, ds)
        df_resp = dat_resp.to_dataframe()
        gage_ids_raw = df_resp.index.tolist()

        # One row per (gage_id, divide_id) pair -- needed to resolve which gages are
        # eligible donors for a given region, since a gage's calibration basin can span many
        # divides (see rafts_algo.regions.donor_gage_ids_for_region's docstring, and
        # rafts_proc_algo_pool.py's identical use of this same structure during training).
        gdf_comid_basin_rows = raftsutil.combine_resp_gdf_comid_wrap(
            dir_std_base=dir_std_base, ds=ds, path_attr_config=path_attr_config)['gdf_comid']

        df_attr_donor = raftsutil.rafts_read_attr_comid(dir_db_attrs,
                                                  gage_ids_raw, attrs_sel=attrs_sel, read_type='all')
        df_donor_wide_all = df_attr_donor.pivot(index='featureID', columns='attribute', values='value').dropna()
        logging.info(f"Ingested donor attribute data. Total locations = {df_donor_wide_all.shape[0]}")

        for region_id, region_spec in region_loop:
            scheme = regions_cfg.scheme if regions_cfg else None
            dirs_std_dict = raftsutil.rafts_save_algo_dir_struct(dir_base, region=region_id, scheme=scheme)
            dir_out = dirs_std_dict.get('dir_out')
            dir_out_alg_base = dirs_std_dict.get('dir_out_alg_base')
            dir_regionalization = Path(dir_out) / "regionalization"
            dir_regionalization.mkdir(exist_ok=True)
            dir_out_alg_ds = Path(dir_out_alg_base) / ds

            # Region scoping: a region's donor pool for pairing must match the pool its
            # model was actually trained on (core+buffer gages) -- otherwise a receiver
            # could get paired with a donor from a completely different region, defeating
            # the point of regionalizing in the first place.
            if region_spec is not None:
                eligible_gage_ids = raftsregions.donor_gage_ids_for_region(
                    gdf_comid_basin_rows, region_spec, gage_id_col='featureID',
                    divide_id_col=region_divide_id_col)
                df_donor_wide_region = df_donor_wide_all[df_donor_wide_all.index.astype(str).isin(eligible_gage_ids)]
                if df_donor_wide_region.empty:
                    logging.warning(f"No eligible donors for region '{region_id}' in {ds}; skipping.")
                    continue
            else:
                df_donor_wide_region = df_donor_wide_all

            for resp_var in resp_vars:
                dynamic_algos = raftsutil.discover_dynamic_algos(
                    search_dir=dir_out_alg_ds, base_algos=algos, metric=resp_var,
                    dataset_id=ds, file_prefix="algo_", file_extension=".joblib"
                )

                for algo in dynamic_algos:
                    logging.info(f"Pairing locations for algorithm: {algo} | Response: {resp_var}" +
                                 (f" | Region: {region_id}" if region_id else ""))

                    # 2. Predict Clusters for DONORS using the loaded pipeline
                    path_algo = raftsutil.std_algo_path(dir_out_alg_ds, algo=algo, metric=resp_var, dataset_id=ds)
                    pipeline_data = raftsutil.load_validated_pipeline(path_algo, arg_val=False)
                    pipe = pipeline_data['pipeline']
                    expected_features = pipe.feature_names_in_

                    # Sort the region's donor pool by expected features:
                    df_donor_wide = df_donor_wide_region[expected_features]

                    # Predict donor clusters
                    donor_clusters = pipe.predict(df_donor_wide)
                    df_donors_paired = df_donor_wide.copy().reset_index()
                    df_donors_paired['prediction'] = donor_clusters

                    # Determine distance metric
                    dist_metric = 'gower' if 'gower' in algo.lower() else 'euclidean'

                    # Spatial pairing preference (optional): merge in donor centroid
                    # coordinates only when actually needed -- neither donor nor receiver
                    # dataframe carries a coordinate column otherwise.
                    max_spa_dist = regions_cfg.max_spa_dist_km if regions_cfg else None
                    if max_spa_dist is not None:
                        donor_coords = (gdf_comid_basin_rows[['featureID', 'geometry']]
                                         .drop_duplicates('featureID').copy())
                        donor_coords['lon'] = donor_coords.geometry.x
                        donor_coords['lat'] = donor_coords.geometry.y
                        df_donors_paired = df_donors_paired.merge(
                            donor_coords[['featureID', 'lon', 'lat']], on='featureID', how='left')

                    # 3. Load RECEIVER Predictions (Ungauged Basins from rafts_pred_algo.py)
                    path_pred_in = raftsutil.std_pred_path(dir_out=dir_out, algo=algo, metric=resp_var, dataset_id=ds)
                    if not path_pred_in.exists():
                        logging.warning(f"Receiver predictions missing: {path_pred_in}")
                        print(f"PROBLEM: No Receiver predictions from the prediction step!! {path_pred_in}")
                        continue
                
                    # Read the cluster predictions across all locations 
                    path_pred_out = raftsutil.std_pred_path(dir_out,algo=algo,metric=resp_var,dataset_id=ds)
                    df_receivers = pd.read_parquet(path_pred_out)
                    df_recv_attrs = raftsutil.read_hfatlas_wrap_dask(
                                        paths_hfatl=[path_meta], 
                                        attrs_sel=attrs_sel, # Empty list forces it to only pull the map_id_col
                                        map_id_col=id_col_pred#,query_clean=True
                                        )
                

                    if 'featureID' not in df_receivers.columns:
                        df_receivers.rename(columns={id_col_pred:'featureID'},inplace=True)

                    df_receivers_mrge = df_receivers.merge(df_recv_attrs, left_on = 'featureID', right_on = id_col_pred, how = 'inner')
                    df_receivers_mrge = df_receivers_mrge[['featureID','prediction']+attrs_sel]

                    logging.info(f"Ingested receiver attribute data. Total locations = {df_receivers.shape[0]}")

                    if max_spa_dist is not None:
                        # The divides layer's own id column is PredConfig.map_divide_id_col
                        # (e.g. 'divide_id'), not literally 'featureID' -- but for the hfATLAS
                        # workflow, a receiver's 'featureID' (renamed from pred_file_comid_colname
                        # above) IS a divide_id value, so the two align once renamed.
                        divide_id_col = validated_pred_cfg.map_divide_id_col
                        gdf_recv_divides = qa_utils.resolve_divides_layer(
                            {'path_hf_finl_gpkg': validated_pred_cfg.path_hf_finl_gpkg,
                             'layr_hf_finl_gpkg': validated_pred_cfg.layr_hf_finl_gpkg},
                            context, divide_id_col=divide_id_col)
                        recv_coords = gdf_recv_divides.drop_duplicates(divide_id_col).copy()
                        recv_centroids = recv_coords.geometry.centroid
                        recv_coords['lon'] = recv_centroids.x
                        recv_coords['lat'] = recv_centroids.y
                        recv_coords = recv_coords.rename(columns={divide_id_col: 'featureID'})
                        df_receivers_mrge = df_receivers_mrge.merge(
                            recv_coords[['featureID', 'lon', 'lat']], on='featureID', how='left')

                    # 4. MISSING DATA IMPUTATION & CLUSTER ASSIGNMENT
                    nan_pred_mask = df_receivers_mrge['prediction'].isna()
                    if nan_pred_mask.any():
                        from sklearn.impute import KNNImputer
                    
                        num_missing = nan_pred_mask.sum()
                        logging.warning(f"Found {num_missing} receivers with NaN cluster predictions due to missing attributes.")
                    
                        # A. Document the locations that will be imputed
                        path_impute_log = raftsutil.std_impute_log_path(dir_regionalization, ds, algo, resp_var)
                        df_receivers_mrge.loc[nan_pred_mask, ['featureID']].to_csv(path_impute_log, index=False)
                        logging.info(f"Wrote list of imputed locations to {path_impute_log}")
                    
                        # B. Impute missing attributes using K-Nearest Neighbors
                        # Fit and transform across all receivers to find the nearest attribute matches
                        imputer = KNNImputer(n_neighbors=5, weights='distance')
                        df_receivers_mrge.loc[:, attrs_sel] = imputer.fit_transform(df_receivers_mrge[attrs_sel])
                    
                        # C. Predict the missing clusters using the imputed attributes
                        # Ensure we pass the features in the exact order the pipeline expects
                        X_imputed = df_receivers_mrge.loc[nan_pred_mask, expected_features]
                        imputed_clusters = pipe.predict(X_imputed)
                    
                        # Update the prediction column with the new cluster assignments
                        df_receivers_mrge.loc[nan_pred_mask, 'prediction'] = imputed_clusters
                        logging.info("Successfully imputed attributes and assigned clusters for missing locations.")               

                    # 5. Execute 1:1 Pairing. max_spa_dist is only passed through when
                    # configured -- omitting it entirely (rather than passing None) keeps
                    # this call identical to pre-regions behavior for every existing config.
                    pairing_kwargs = {}
                    if max_spa_dist is not None:
                        pairing_kwargs = {'max_spa_dist': max_spa_dist, 'spatial_cols': ('lon', 'lat')}
                    df_pairings = raftsat.assign_donors_to_receivers(
                        df_donors=df_donors_paired,
                        df_receivers=df_receivers_mrge,
                        attrs=attrs_sel,
                        metric=dist_metric,
                        **pairing_kwargs
                    )

                    if not df_pairings.empty:
                        df_pairings['dataset'] = ds
                        df_pairings['algo'] = algo
                        df_pairings['resp_var'] = resp_var
                    
                        # 6. Save Output
                        path_pair_out = raftsutil.std_donor_pairs_path(dir_regionalization, ds, algo, resp_var)
                        path_pair_out.parent.mkdir(parents=True, exist_ok=True)
                        df_pairings.to_csv(path_pair_out, index=False)
                        logging.info(f"Saved {len(df_pairings)} donor-receiver pairings to {path_pair_out}")


                        # 7. Assign parameter sets to receivers from donors
                        logging.info(f"Assigning donor parameters to receiver locations for {algo}...")
                        try:
                            # Identify the ID column in df_resp (typically 'gage_id' or 'featureID')
                            id_col_resp = df_resp.index.name
                            if not id_col_resp:
                                id_col_resp = 'gage_id' if 'gage_id' in df_resp.columns else  \
                                    logging.error(f"Could not determine the location identifier column in the prepared response variable dataset {ds}") and \
                                    print(f"Could not determine the location identifier column in the prepared response variable dataset {ds}")

                            # # Extract the parameter columns (data variables in the .nc file)
                            # param_cols = list(dat_resp.data_vars.keys())
                            # df_params_only = df_resp[[id_col_resp] + param_cols].copy()
                        
                            # Enforce string types to prevent merge failures
                            #df_params_only[id_col_resp] = df_params_only[id_col_resp].astype(str)
                            df_pairings['donor_id'] = df_pairings['donor_id'].astype(str)
                        
                            # Merge parameters onto the pairing DataFrame based on donor_id
                            df_receiver_params = df_pairings[['receiver_id', 'donor_id']].merge(
                                df_resp, 
                                left_on='donor_id', 
                                right_on=id_col_resp, 
                                how='left'
                            )

                            # Insert check on gage_ids
                            tot_gage_ids = len(df_resp.index)
                            tot_donor_ids = df_receiver_params['donor_id'].nunique()
                            tot_cmmn_ids = len(np.intersect1d(df_resp.index.unique(),df_receiver_params['donor_id'].unique()))
                            if tot_cmmn_ids < tot_donor_ids:
                                logging.error('Something is wrong with the donor ids. Some are unknown, having no parameter sets.')


                            # Format as wide: featureID (receiver), and the parameters
                            df_receiver_params = df_receiver_params.rename(columns={'receiver_id': 'featureID'})
                            # cols_to_keep = ['featureID'] + param_cols
                            # df_receiver_params_wide = df_receiver_params[cols_to_keep]
                        
                            # Reformat columns response variable columns to have pint units
                            path_mapper = raftsutil.std_unit_mapper_path(dir_std_base=dir_std_base, ds=ds, cstm_str='resp_vars')
                            logging.info(f'Writing mapping of pint units/colnames to {path_mapper}')
                            mapper_df = pd.read_csv(path_mapper)

                            # Rename response variable columns based on options in the mapper column's 'raw' field
                            rename_dict = dict(zip(mapper_df['clean_column'], mapper_df['raw_column']))

                            # Apply the renaming to your dataframe
                            df_receiver_params = df_receiver_params.rename(columns=rename_dict)

                            # Save output
                            path_params_out = raftsutil.std_receiver_params_path(dir_regionalization, ds, algo, resp_var)
                            df_receiver_params.to_csv(path_params_out, index=False)
                            logging.info(f"Saved assigned receiver parameters to {path_params_out}")

                            # 8. OPTIONAL CROSSWALK MAPPING
                            # Fetch from the pre-parsed prediction configuration using validated schema
                            path_crosswalk_ids_raw = validated_pred_cfg.path_crosswalk_ids
                            crosswalk_target_col = validated_pred_cfg.crosswalk_target_col # The target id column in the crosswalk file (e.g. 'divide_id')
                            pred_gpkg_id_col = validated_pred_cfg.pred_gpkg_id_col
                        
                            if path_crosswalk_ids_raw:
                                # Safely resolve any f-strings (like {dir_std_base}) in the path
                                path_crosswalk_ids = Path(raftsutil.resolve_fstrings(path_crosswalk_ids_raw, context))
                            
                                if path_crosswalk_ids.exists():
                                    logging.info(f"Applying crosswalk mapping from {path_crosswalk_ids}")
                                
                                    # Ensure crosswalk is strictly read as string to prevent integer coercion
                                    if str(path_crosswalk_ids).endswith('.csv'):
                                        df_crosswalk = pd.read_csv(path_crosswalk_ids, dtype=str)
                                    elif str(path_crosswalk_ids).endswith('.parquet'):
                                        df_crosswalk = pd.read_parquet(path_crosswalk_ids).astype(str)
                                    else:
                                        print("ERROR: Crosswalk file must be a .csv or .parquet")
                                        logging.error("Crosswalk file must be a .csv or .parquet")
                                        continue
                                    
                                    # Identify the desired new identifier column (the one that isn't pred_gpkg_id_col)
                                    desired_id_col = raftsutil.get_crosswalk_target_col(df_crosswalk, pred_gpkg_id_col, crosswalk_target_col)
                                    if desired_id_col:
                                        df_receiver_params['featureID'] = df_receiver_params['featureID'].astype(str)
                                    
                                        # Merge crosswalk onto the assigned parameters
                                        df_mapped_params = df_receiver_params.merge(
                                            df_crosswalk, 
                                            left_on='featureID', 
                                            right_on=pred_gpkg_id_col, 
                                            how='inner'
                                        )
                                    
                                        # Clean up columns: drop the old featureID and the crosswalk join key
                                        cols_to_drop = ['featureID']
                                        if pred_gpkg_id_col in df_mapped_params.columns and pred_gpkg_id_col != desired_id_col:
                                            cols_to_drop.append(pred_gpkg_id_col)
                                        
                                        df_mapped_params = df_mapped_params.drop(columns=cols_to_drop, errors='ignore')
                                    
                                        # Move the new desired identifier column to the front of the DataFrame
                                        new_col_order = [desired_id_col] + [c for c in df_mapped_params.columns if c != desired_id_col]
                                        df_mapped_params = df_mapped_params[new_col_order]
                                    
                                        # Save the final mapped parameters as a Parquet file
                                        path_params_cw_out = raftsutil.std_receiver_params_mapped_path(dir_regionalization, ds, algo,
                                                                                                     resp_var, ext=".parquet")
                                        df_mapped_params.to_parquet(path_params_cw_out, index=False)
                                        logging.info(f"Saved crosswalk-mapped receiver parameters to {path_params_cw_out}")

                                        path_params_cw_out_gpkg = raftsutil.std_receiver_params_mapped_path(dir_regionalization, ds, algo, 
                                                                                                         resp_var, ext=".gpkg")
                                  
                                        with sqlite3.connect(path_params_cw_out_gpkg) as conn:
                                            df_mapped_params.to_sql("parameters", conn, if_exists='replace', index=False)
                                        logging.info(f"Saved crosswalk-mapped receiver parameters to {path_params_cw_out_gpkg}")
                                    
                                    else:
                                        print(f"ERROR: Crosswalk file missing the specified pred_gpkg_id_col: {pred_gpkg_id_col}")
                                        logging.error(f"Crosswalk file missing the specified pred_gpkg_id_col: {pred_gpkg_id_col}")
                                else:
                                    print(f"Crosswalk file path was provided but does not exist: {path_crosswalk_ids}")
                                    logging.error(f"ERROR: Crosswalk file path was provided but does not exist: {path_crosswalk_ids}")
                            else:
                                print("WARN: NOT performing a crosswalk on aggregated identifiers.")
                                logging.warning("NOT performing a crosswalk on aggregated identifiers.")
                        except Exception as e:
                            print(f"ERROR: Failed to assign donor parameters to receivers: {e}")
                            logging.error(f"Failed to assign donor parameters to receivers: {e}")

    logging.info("FINISHED Donor-Receiver Pairing & Parameter Assignment.")
    logging.shutdown()
    