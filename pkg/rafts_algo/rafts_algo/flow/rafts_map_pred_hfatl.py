"""rafts_map_pred_hfatl_huc_div.py
Workflow script to generate a map of predictions

Make sure to add in a unique string pertaining to the prediction of interest!
Example: 
    >>> python rafts_map_pred_hfatl.py "/path/to/pred_config.yaml" --analysis_str "huc08"

# Changelog/contributions
    2025-08-21 added logging, GL
    2025-10-10 refactor to renamed rafts_algo modules, GL
    2026-05-01 adapted to support dynamic ID joins for hfATLAS and mapie uncertainties
    2026-07-21 add custom pred gpkg capability, GL
    2026-08-07 feat: auto-generate divide-level map if crosswalk and master GPKG are present, Gemini3.1Pro
    2026-09-18 refactor: Integrate Pydantic PredConfig schema validation.
"""
import argparse
import pandas as pd
from pathlib import Path
import rafts_algo.utils as raftsutil
import rafts_algo.regions as raftsregions
import rafts_algo.plots as raftsplot
import geopandas as gpd
import logging
from logging.handlers import MemoryHandler
import rafts_prep.proc_eval_metrics as pem
import gc

from rafts_algo.schemas.pydantic_schemas import PredConfig

# Predict values and evaluate predictions
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'process the prediction config file')
    parser.add_argument('path_pred_config', type=str, help='Path to the YAML configuration file specific for prediction.')
    parser.add_argument('--analysis_str', type=str, default='',required=False) # The string to add on to end of each plotting file name
    parser.add_argument('--region', type=str, default=None,
                         help="Restrict this run to a single configured sub-region's region_id. "
                              "Omit to loop over every resolved sub-region, or run once "
                              "unregioned if `regions:` isn't set.")
    args = parser.parse_args()

    path_pred_config = Path(args.path_pred_config).expanduser() 
    
    # --- Commence logging before creating the log file
    memory_handler = MemoryHandler(capacity=30)
    root_logger = logging.getLogger()
    root_logger.addHandler(memory_handler)
    root_logger.setLevel(logging.INFO) 
    logging.info(f"Running rafts_map_pred_hfatl.py with {path_pred_config.name} config file")
    # ---
    
    analysis_str = args.analysis_str

    # --- Pydantic Validation ---
    validated_pred_cfg = raftsutil.load_validated_config(path_pred_config, PredConfig)

    #%% prediction config var extract (Via Pydantic where applicable)
    resp_vars = validated_pred_cfg.algo_response_vars
    algos = validated_pred_cfg.algo_type
    path_gpkg_pred = validated_pred_cfg.path_gpkg_pred
    pred_gpkg_lyr = validated_pred_cfg.pred_gpkg_lyr
    pred_gpkg_id_col = validated_pred_cfg.pred_gpkg_id_col
    crosswalk_target_col = validated_pred_cfg.crosswalk_target_col

    #%%  READ CONTENTS FROM THE ATTRIBUTE CONFIG
    path_attr_config = raftsutil.build_cfig_path(path_pred_config, validated_pred_cfg.name_attr_config)
    path_algo_config = raftsutil.build_cfig_path(path_pred_config, validated_pred_cfg.name_algo_config)

    algo_cfig = raftsutil.AlgoConfigParser(path_algo_config)
    algo_cfig._read_algo_config()
    algo_config = algo_cfig.algo_cfg_unc_dict["algo_cfg_dict"]["algo_config"]
    metrics = algo_cfig.algo_cfg_unc_dict["algo_cfg_dict"]["metrics"]
    task_type = algo_cfig.algo_cfg_unc_dict["algo_cfg_dict"].get("task_type", "regression")

    attr_cfig = raftsutil.AttrConfigAndVars(path_attr_config)
    attr_cfig._read_attr_config()

    dir_base = attr_cfig.attrs_cfg_dict.get('dir_base')
    dir_std_base = attr_cfig.attrs_cfg_dict.get('dir_std_base')
    home_dir = attr_cfig.attrs_cfg_dict.get('home_dir')
    datasets = attr_cfig.attrs_cfg_dict.get('datasets')

    regions_cfg = validated_pred_cfg.regions  # None unless this config sets a `regions:` block
    region_loop = raftsregions.resolve_region_loop(
        regions_cfg, args.region, context={'home_dir': home_dir, 'dir_base': dir_base})
    region_scheme = regions_cfg.scheme if regions_cfg else None

    # ---------- Generate path to the log file & initialize logging -----------
    path_log = pem.std_path_log(dir_input=dir_base, path_config=path_pred_config, script='rafts_map_pred')
    logging.basicConfig(level=logging.INFO, filename=path_log, format='%(asctime)s - %(levelname)s - %(message)s', filemode='w', force=True)

    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler):
            memory_handler.setTarget(handler)
            memory_handler.flush()
            
    logging.info(f"Writing logs to {path_log}")
    root_logger.removeHandler(memory_handler) 
    # -------------------------------------------------------------------------
    
    path_hf_finl_gpkg_raw = validated_pred_cfg.path_hf_finl_gpkg
    layr_hf_finl_gpkg = validated_pred_cfg.layr_hf_finl_gpkg
    path_crosswalk_ids_raw = validated_pred_cfg.path_crosswalk_ids

    for ds in datasets: 
        print(f"Mapping predictions for {ds} dataset")
        vals = {'dir_std_base':dir_std_base,'ds':ds, 'home_dir':home_dir}
        path_rafts_dat_resp =  raftsutil._std_rafts_prep_ds_paths(dir_std_base=dir_std_base,ds=ds,mtch_str='*.nc')
        path_gpkg_rafts_prep = raftsutil._std_rafts_prep_ds_companion_gpkg_path(path_rafts_dat_resp[0])

        # Initialize safe placeholder variables
        df_crosswalk = None
        gdf_divides = None
        desired_id_col = None

        if path_crosswalk_ids_raw and path_hf_finl_gpkg_raw:
            path_crosswalk_ids = Path(raftsutil.resolve_fstrings(path_crosswalk_ids_raw, vals))
            
            if path_crosswalk_ids.exists():
                df_crosswalk = pd.read_parquet(path_crosswalk_ids).astype(str) if str(path_crosswalk_ids).endswith('.parquet') else pd.read_csv(path_crosswalk_ids, dtype=str)
                crosswalk_cols = list(df_crosswalk.columns)


                # Use the centralized hierarchy to find the target column
                desired_id_col = raftsutil.get_crosswalk_target_col(df_crosswalk, pred_gpkg_id_col, crosswalk_target_col)
                
                if not desired_id_col:
                    logging.error("Could not identify a valid target identifier column in crosswalk.")
                    continue
                    
                path_hf_finl_gpkg = Path(raftsutil.resolve_fstrings(path_hf_finl_gpkg_raw, vals))
                if not path_hf_finl_gpkg.exists():
                    logging.warning(f"Master GPKG does not exist: {path_hf_finl_gpkg}")
            
                try:  # Read hydrofabric divides
                    gdf_divides = gpd.read_file(path_hf_finl_gpkg, layer=layr_hf_finl_gpkg, columns=[desired_id_col, 'geometry'], engine='pyogrio')
                    gdf_divides[desired_id_col] = gdf_divides[desired_id_col].astype(str)
                except:
                    try: # Try out the divides layer
                        gdf_divides = gpd.read_file(path_hf_finl_gpkg, layer='divides', columns=[desired_id_col, 'geometry'], engine='pyogrio')
                        gdf_divides[desired_id_col] = gdf_divides[desired_id_col].astype(str)
                    except Exception as e:
                        logging.warning(f"Could not read '{str(layr_hf_finl_gpkg)}' nor 'divides' layer names from {path_hf_finl_gpkg}. Skipping secondary map. Error: {e}")

        # ---
        layers = gpd.list_layers(path_gpkg_rafts_prep)
        lyr = None
        if len(layers) > 0:
            if layers['name'].str.contains('outlet').any():
                lyr = 'outlet' 
                
        if path_gpkg_pred:
            path_gpkg_pred = Path(str(path_gpkg_pred).format(**vals))
            gdf_all = gpd.read_file(path_gpkg_pred,layer=pred_gpkg_lyr,
                                    columns=[pred_gpkg_id_col], engine='pyogrio')
            gdf_all['featureID'] = gdf_all[pred_gpkg_id_col]
        else:
            logging.warning(f"Falling back on reading the gpkg used in the response variable preparation: {path_gpkg_rafts_prep}")
            gdf_all = gpd.read_file(path_gpkg_rafts_prep,layer=lyr)

        if 'featureID' not in gdf_all.columns and 'comid' in gdf_all.columns:
            gdf_all['featureID'] = gdf_all['comid']
            gdf_all['featureSource'] = pred_gpkg_id_col
        
        if 'featureID' not in gdf_all.columns:
            logging.error(f'Expecting featureID column to be in the gdf_all geodataframe')

        for region_id, region_spec in region_loop:
            dirs_std_dict = raftsutil.rafts_save_algo_dir_struct(dir_base, region=region_id, scheme=region_scheme)
            dir_out_viz_base = dirs_std_dict.get('dir_out_viz_base')
            dir_out = dirs_std_dict.get('dir_out')

            for metr in resp_vars:
                dir_preds_ds = Path(dir_out) / 'algorithm_predictions' / ds
                dynamic_algos = raftsutil.discover_dynamic_algos(
                    search_dir=dir_preds_ds,
                    base_algos=algos,
                    metric=metr,
                    dataset_id=ds,
                    file_prefix="pred_",
                    file_extension=".parquet"
                )
            
                if not dynamic_algos:
                    logging.warning(f"No prediction files found for {metr} to map. Skipping.")
                    continue

                for algo_str in dynamic_algos:
                    logging.info(f"Generating prediction map for dataset: {ds}\nAlgorithm: {algo_str}\nResponse variable: {metr}")
                
                    path_pred_in = raftsutil.std_pred_path(dir_out=dir_out,algo=algo_str,metric=metr,dataset_id=ds)
                
                    if not Path(path_pred_in).exists():
                        logging.warning(f"Prediction file not found: {path_pred_in}. Skipping.")
                        continue
                    
                    df_pred = pd.read_parquet(path_pred_in)

                    possible_joins = [('featureID', 'featureID'), ('divide_id', 'featureID'), ('comid', 'featureID')]
                    join_col_gpkg = None
                
                    for gpkg_c, pred_c in possible_joins:
                        if gpkg_c in gdf_all.columns and pred_c in df_pred.columns:
                            join_col_gpkg = gpkg_c
                            break
                        
                    if not join_col_gpkg:
                        logging.error(f"Could not find matching ID columns to join GPKG {gdf_all.columns.tolist()} and Preds {df_pred.columns.tolist()}")
                        continue

                    gdf_all[join_col_gpkg] = gdf_all[join_col_gpkg].astype(str)
                    df_pred['featureID'] = df_pred['featureID'].astype(str)

                    gdf_pred = gdf_all.merge(df_pred, how='inner', left_on=join_col_gpkg, right_on='featureID')
                
                    if gdf_pred.empty:
                        logging.error("Merge resulted in an empty GeoDataFrame. IDs did not match.")
                        continue

                    # =========================================================================
                    # HELPER: Execution sequence for map plotting
                    # =========================================================================
                    def execute_mapping(gdf_to_plot, current_analysis_str, gdf_missing=None):
                        logging.info(f"Plotting predictions for {current_analysis_str}")
                        raftsplot.plot_map_pred_wrap(
                            test_gdf=gdf_to_plot,
                            dir_out_viz_base=dir_out_viz_base,
                            ds=ds, metr=metr, algo_str=algo_str,
                            split_type=current_analysis_str,
                            colname_data='prediction', epsg_reproj=3857, task_type=task_type,
                            gdf_missing=gdf_missing
                        )
                    
                        mapie_alphas = raftsutil.infer_mapie_alphas(gdf_to_plot.columns)
                        for alpha_val in mapie_alphas:
                            logging.info(f"Generating MAPIE uncertainty map for alpha={alpha_val}")
                            raftsplot.plot_map_pred_wrap_uncn(
                                test_gdf=gdf_to_plot, dir_out_viz_base=dir_out_viz_base, 
                                ds=ds, metr=metr, algo_str=algo_str, alpha_val=alpha_val, uncn_col=None,
                                split_type=current_analysis_str, colname_data='prediction', epsg_reproj=3857
                            )

                        if 'forestci' in gdf_to_plot.columns:
                            logging.info("Generating ForestCI uncertainty map")
                            raftsplot.plot_map_pred_wrap_uncn(
                                test_gdf=gdf_to_plot, dir_out_viz_base=dir_out_viz_base, 
                                ds=ds, metr=metr, algo_str=algo_str, alpha_val=None, uncn_col='forestci',
                                split_type=current_analysis_str, colname_data='prediction', epsg_reproj=3857
                            )
                        del gdf_to_plot
                        gc.collect()
                    # -----------------------------------------------------
                    # 1. PLOT PRIMARY GEOMETRY
                    # -----------------------------------------------------
                    execute_mapping(gdf_pred, analysis_str)

                    # -----------------------------------------------------
                    # 2. PLOT SECONDARY GEOMETRY (DIVIDES via CROSSWALK)
                    # -----------------------------------------------------
                    if df_crosswalk is not None and gdf_divides is not None and desired_id_col:
                        logging.info("Crosswalk and master GPKG found. Generating secondary divide-level map.")

                        # Broadcast the aggregated predictions down to the divide scale.
                        # NOTE: left (not inner) merges here on purpose -- an inner merge
                        # silently drops every divide_id absent from df_crosswalk, with no
                        # record of how many or which ones were lost (only an all-or-nothing
                        # gdf_pred_divides.empty check below). A real gap like this was
                        # found and confirmed empirically: 13,952 of 555,866 divides in the
                        # master GPKG (2.5%) have no entry at all in the huc12 crosswalk, so
                        # their huc12 parent's cluster prediction (which does exist) never
                        # reaches them -- they used to vanish from the map without a trace,
                        # even though "the predictions populate cluster values for all
                        # divides" upstream of this broadcast step.
                        df_pred_mapped = df_pred.merge(df_crosswalk, left_on='featureID', right_on=pred_gpkg_id_col, how='left')
                        n_pred_unmatched = df_pred_mapped[desired_id_col].isna().sum()
                        if n_pred_unmatched:
                            logging.warning(
                                f"{n_pred_unmatched} of {len(df_pred_mapped)} predicted {pred_gpkg_id_col} "
                                f"rows have no {desired_id_col} in the crosswalk and cannot be broadcast to divides."
                            )

                        gdf_pred_divides = gdf_divides.merge(df_pred_mapped, on=desired_id_col, how='left')
                        missing_mask = gdf_pred_divides['prediction'].isna() if 'prediction' in gdf_pred_divides.columns else gdf_pred_divides[pred_gpkg_id_col].isna()
                        gdf_missing_divides = gdf_pred_divides[missing_mask]
                        gdf_pred_divides = gdf_pred_divides[~missing_mask]
                        if not gdf_missing_divides.empty:
                            logging.warning(
                                f"{len(gdf_missing_divides)} of {len(gdf_divides)} divides in the master GPKG have "
                                f"no matching prediction after the crosswalk broadcast (e.g. {gdf_missing_divides[desired_id_col].head(5).tolist()}); "
                                f"rendering them as a distinct 'no crosswalk data' layer rather than leaving them blank."
                            )

                        if not gdf_pred_divides.empty:
                            div_analysis_str = f"{analysis_str}_divides" if analysis_str else "divides"
                            execute_mapping(gdf_pred_divides, div_analysis_str, gdf_missing=gdf_missing_divides)
                        else:
                            logging.warning("Divide-level merge resulted in an empty GeoDataFrame.")
        
            logging.info(f"Prediction map plots stored inside {dir_out_viz_base}")
        logging.info(f"Completed prediction map generation for {path_pred_config}")
        
    logging.shutdown()