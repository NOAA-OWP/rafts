"""rafts_map_donor_receiver.py

Workflow script to map donor-receiver pairings for one or more configured regions
(e.g. a state or group of states): receivers are shown as a categorical choropleth of
their predicted cluster, donor gage locations as solid points, receiver centroids as
hollow squares, and a line connecting each receiver to its assigned donor.

Donor-receiver pairing is an unsupervised (clustering) concept only -- rafts_pair_donors.py
itself refuses to run for a supervised task_type. This script mirrors that constraint as a
graceful no-op (not an error) rather than requiring every prediction config in a shared
shell loop to know in advance whether pairing applies to it.

Regions default to Florida and the Pacific Northwest (`DEFAULT_DONOR_MAP_REGIONS`) when a
prediction config doesn't set `donor_map_states` at all. A config can override this with:
    donor_map_states: ['FL']                          # one region, named by its states
    donor_map_states: {FL: ['FL'], PNW: ['WA', 'OR']}  # multiple named regions
    donor_map_states: []                               # explicit opt-out (no map at all)

A HUC12 with no huc12<->divide_id crosswalk entry gets no predictor row and so never
appears in the donor-pairs output at all -- normally rendered as a hatched "No donor
pairing" gap. Where that gap is caused by a single hydrofabric divide bigger than the
HUC12 itself (see scripts/qa/crosswalk_gap_diagnosis/), this substitutes the divide's own
geometry as a proxy receiver, borrowing the pairing from whichever HUC12 the crosswalk
actually assigned that divide to -- see build_divide_proxy_receivers(). Best-effort: any
failure in this enhancement just falls back to the plain hatched-gap behavior.

Usage:
    >>> python rafts_map_donor_receiver.py "/path/to/pred_config.yaml"

# Changelog/Contributions
    2026-09-22 Originally created, GL
    2026-09-22 Default to FL + Pacific Northwest regions when unconfigured; support
               multiple named regions per run, GL
    2026-09-22 Substitute divide-shaped proxy receivers (borrowed pairing) for unpaired
               HUC12s dominated by a larger divide, instead of leaving them hatched, GL
"""
import argparse
import sys
import logging
from logging.handlers import MemoryHandler
from pathlib import Path

import pandas as pd
import geopandas as gpd

import rafts_algo.utils as raftsutil
import rafts_algo.regions as raftsregions
import rafts_algo.plots as raftsplot
import rafts_prep.proc_eval_metrics as pem
from rafts_algo.qa_utils import resolve_divides_layer, find_huc12_divide_overlaps

# Applied only when a prediction config has no `donor_map_states` key at all (a config
# that sets it -- including to an empty value -- always takes precedence over this).
# Unrelated to the sub-regional `regions:` config block (rafts_algo.regions) -- this is a
# purely cosmetic display-grouping default for this map, not a training/pairing scope.
DEFAULT_DONOR_MAP_REGIONS = {
    'FL': ['FL'],
    'PNW': ['WA', 'OR'],
}


def build_divide_proxy_receivers(gdf_no_pairing: gpd.GeoDataFrame, gdf_divides: gpd.GeoDataFrame,
                                  divide_to_assigned_huc: pd.Series, divide_area_sqkm: pd.Series,
                                  dp: pd.DataFrame, pred_gpkg_id_col: str, divide_id_col: str,
                                  min_coverage_frac: float = 0.05) -> tuple:
    """Resolve unpaired HUC12s dominated by a larger hydrofabric divide into divide-shaped
    proxy receivers, borrowing the pairing from whichever HUC12 the crosswalk actually
    assigned that divide to.

    A HUC12 with no crosswalk entry gets no predictor row and so never appears in the
    donor-pairs output at all -- it shows up as a row in `gdf_no_pairing` (hatched "No
    donor pairing" on the map). When that gap is caused by a single hydrofabric divide
    larger than the HUC12 itself, the huc12<->divide_id crosswalk (strictly 1:1) can only
    assign that divide to ONE of the HUC12s it overlaps -- see
    scripts/qa/crosswalk_gap_diagnosis/ for the full diagnosis this reuses. The physically
    correct receiver shape for that area is the divide, not the undersized HUC12 polygon
    that lost the crosswalk assignment, and a real, data-backed pairing already exists for
    it: whatever the crosswalk assigned that divide to. This substitutes the divide's own
    geometry in for those cases, carrying over the assigned HUC12's cluster_id/donor_id.

    :param gdf_no_pairing: Region HUC12s with no donor-pairs entry, with a `pred_gpkg_id_col`
     column and 'areasqkm' (used to test "the divide is bigger than the HUC12").
    :type gdf_no_pairing: gpd.GeoDataFrame
    :param gdf_divides: Hydrofabric divide polygons, with `divide_id_col`.
    :type gdf_divides: gpd.GeoDataFrame
    :param divide_to_assigned_huc: Crosswalk lookup, divide_id -> the HUC12 it's actually assigned to.
    :type divide_to_assigned_huc: pd.Series
    :param divide_area_sqkm: Divide footprint area (sqkm) by divide_id, computed in an equal-area CRS.
    :type divide_area_sqkm: pd.Series
    :param dp: The full donor_pairs table for this ds/algo/resp_var (receiver_id, cluster_id, donor_id).
    :type dp: pd.DataFrame
    :param pred_gpkg_id_col: Column name holding the HUC12 identifier in `gdf_no_pairing`.
    :type pred_gpkg_id_col: str
    :param divide_id_col: Column name holding the divide identifier.
    :type divide_id_col: str
    :param min_coverage_frac: Minimum fraction of the HUC12's own area a divide must cover
     to be considered its dominant divide, defaults to 0.05
    :type min_coverage_frac: float, optional
    :return: A 2-tuple: (divide-shaped proxy receivers, in `gdf_no_pairing`'s CRS, with
     `pred_gpkg_id_col`/`divide_id_col`/'cluster_id'/'donor_id'/'is_divide_proxy'/geometry;
     the remaining unresolved rows of `gdf_no_pairing`, same schema as the input).
    :rtype: tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]
    """
    empty_proxy = gpd.GeoDataFrame(
        columns=[pred_gpkg_id_col, divide_id_col, 'cluster_id', 'donor_id', 'is_divide_proxy', 'geometry'],
        geometry='geometry', crs=gdf_no_pairing.crs)
    if gdf_no_pairing.empty or gdf_divides.empty or 'areasqkm' not in gdf_no_pairing.columns:
        return empty_proxy, gdf_no_pairing

    overlaps = find_huc12_divide_overlaps(gdf_no_pairing, gdf_divides, divide_id_col,
                                           min_frac=min_coverage_frac, huc12_col=pred_gpkg_id_col)
    if overlaps.empty:
        return empty_proxy, gdf_no_pairing

    # Dominant divide per HUC12: the one covering the largest share of that HUC12's own area.
    dominant = overlaps.sort_values('frac_of_huc', ascending=False).drop_duplicates(pred_gpkg_id_col)
    dominant = dominant.merge(gdf_no_pairing[[pred_gpkg_id_col, 'areasqkm']], on=pred_gpkg_id_col, how='left')
    dominant['divide_area_sqkm'] = dominant[divide_id_col].map(divide_area_sqkm)
    dominant = dominant[dominant['divide_area_sqkm'] > dominant['areasqkm']]
    if dominant.empty:
        return empty_proxy, gdf_no_pairing

    dominant['assigned_huc12'] = dominant[divide_id_col].map(divide_to_assigned_huc)
    # Join on the actual key (assigned_huc12 <-> receiver_id) rather than building
    # `borrowed` via dp_by_huc.loc[list-like] and zipping its .values against
    # resolvable's .values by position -- that relied on .loc preserving the given
    # list's order, an implicit pandas behavior rather than something asserted here.
    # dp's caller already enforces unique receiver_id (.drop_duplicates('receiver_id')
    # in __main__), so this merge can't fan out rows.
    resolvable = dominant.merge(
        dp[['receiver_id', 'cluster_id', 'donor_id']],
        left_on='assigned_huc12', right_on='receiver_id', how='inner',
    )
    if resolvable.empty:
        return empty_proxy, gdf_no_pairing

    divide_geom = gdf_divides.set_index(divide_id_col).geometry

    gdf_proxy = gpd.GeoDataFrame({
        pred_gpkg_id_col: resolvable[pred_gpkg_id_col].values,
        divide_id_col: resolvable[divide_id_col].values,
        'cluster_id': resolvable['cluster_id'].values,
        'donor_id': resolvable['donor_id'].values,
        'is_divide_proxy': True,
    }, geometry=[divide_geom[d] for d in resolvable[divide_id_col]], crs=gdf_divides.crs)
    gdf_proxy['cluster_id'] = gdf_proxy['cluster_id'].astype('Int64')
    # Match gdf_no_pairing's CRS before the caller concats this with other receiver
    # geometries -- gdf_divides is often in a different native CRS than the WBD HUC12
    # layer, and a plain pd.concat/GeoDataFrame append does not reconcile CRS on its own.
    gdf_proxy = gdf_proxy.to_crs(gdf_no_pairing.crs)

    gdf_remaining = gdf_no_pairing[~gdf_no_pairing[pred_gpkg_id_col].isin(resolvable[pred_gpkg_id_col])]
    return gdf_proxy, gdf_remaining


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Map donor-receiver pairings for one or more configured regions.')
    parser.add_argument('path_pred_config', type=str, help='Path to the YAML configuration file specific for prediction.')
    parser.add_argument('--region', type=str, default=None,
                         help="Restrict this run to a single configured sub-region's region_id "
                              "(rafts_algo.regions; unrelated to donor_map_states below). Omit to "
                              "loop over every resolved sub-region, or run once unregioned if "
                              "`regions:` isn't set.")
    args = parser.parse_args()

    path_pred_config = Path(args.path_pred_config).expanduser()

    # --- Commence logging before creating the log file
    memory_handler = MemoryHandler(capacity=30)
    root_logger = logging.getLogger()
    root_logger.addHandler(memory_handler)
    root_logger.setLevel(logging.INFO)
    logging.info(f"Running rafts_map_donor_receiver.py with {path_pred_config.name} config file")
    # ---

    pred_cfg = raftsutil.PredConfigParser(path_pred_config)
    pred_cfg._read_pred_config()

    path_attr_config = raftsutil.build_cfig_path(pred_cfg.pred_cfg_dict.get('path_pred_config'), pred_cfg.pred_cfg_dict.get('name_attr_config'))
    path_algo_config = raftsutil.build_cfig_path(pred_cfg.pred_cfg_dict.get('path_pred_config'), pred_cfg.pred_cfg_dict.get('name_algo_config'))

    algo_cfig = raftsutil.AlgoConfigParser(path_algo_config)
    algo_cfig._read_algo_config()
    task_type = algo_cfig.algo_cfg_unc_dict["algo_cfg_dict"].get("task_type", "regression")

    # --- Guard 1: donor-receiver pairing (rafts_pair_donors.py) is clustering-only.
    # This script is wired unconditionally into the shared regn_all_proc.sh loop across
    # both supervised and unsupervised model configs, so it must recognize on its own
    # when pairing doesn't apply rather than erroring the whole pipeline out.
    if task_type != 'clustering':
        msg = (f"task_type='{task_type}' has no donor-receiver pairing step "
               f"(rafts_pair_donors.py is clustering-only); skipping donor-receiver map.")
        logging.info(msg)
        print(f"Skipping donor-receiver map for {path_pred_config.name}: {msg}")
        sys.exit(0)

    # --- Guard 2: regions default to DEFAULT_DONOR_MAP_REGIONS when `donor_map_states`
    # is absent from the config entirely. An explicit empty/falsy value (e.g. `[]`) is
    # a deliberate opt-out and is honored as such, distinct from "not set."
    donor_map_states_cfg = pred_cfg.pred_cfg_dict.get('donor_map_states')
    if donor_map_states_cfg is None:
        regions = dict(DEFAULT_DONOR_MAP_REGIONS)
        logging.info(f"'donor_map_states' not set in prediction config; defaulting to built-in regions: {regions}.")
    elif not donor_map_states_cfg:
        msg = "'donor_map_states' explicitly set empty; skipping donor-receiver map."
        logging.info(msg)
        print(f"Skipping donor-receiver map for {path_pred_config.name}: {msg}")
        sys.exit(0)
    else:
        regions = raftsregions.normalize_named_state_groups(donor_map_states_cfg)

    attr_cfig = raftsutil.AttrConfigAndVars(path_attr_config)
    attr_cfig._read_attr_config()

    dir_base = attr_cfig.attrs_cfg_dict.get('dir_base')
    dir_std_base = attr_cfig.attrs_cfg_dict.get('dir_std_base')
    home_dir = attr_cfig.attrs_cfg_dict.get('home_dir')
    datasets = attr_cfig.attrs_cfg_dict.get('datasets')

    regions_cfg = pred_cfg.pred_cfg_dict.get('regions')  # None unless this config sets a block
    sub_region_loop = raftsregions.resolve_region_loop(
        regions_cfg, args.region, context={'home_dir': home_dir, 'dir_base': dir_base})
    sub_region_scheme = regions_cfg.get('scheme') if regions_cfg else None

    # ---------- Generate path to the log file & initialize logging -----------
    path_log = pem.std_path_log(dir_input=dir_base, path_config=path_pred_config, script='rafts_map_donor_receiver')
    logging.basicConfig(level=logging.INFO, filename=path_log, format='%(asctime)s - %(levelname)s - %(message)s', filemode='w', force=True)
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler):
            memory_handler.setTarget(handler)
            memory_handler.flush()
            break
    logging.info(f"Writing logs to {path_log}")
    root_logger.removeHandler(memory_handler)
    # -------------------------------------------------------------------------

    path_gpkg_pred = pred_cfg.pred_cfg_dict.get('path_gpkg_pred')
    pred_gpkg_lyr = pred_cfg.pred_cfg_dict.get('pred_gpkg_lyr')
    pred_gpkg_id_col = pred_cfg.pred_cfg_dict.get('pred_gpkg_id_col')
    path_hf_finl_gpkg_raw = pred_cfg.pred_cfg_dict.get('path_hf_finl_gpkg')

    # --- Guard 3: the receiver polygons (for the choropleth) and donor gage locations
    # (from the hydrofabric's hydrolocations layer) both need a real source path.
    if not (path_gpkg_pred and pred_gpkg_lyr and pred_gpkg_id_col):
        logging.error("donor_map_states resolved to region(s) but path_gpkg_pred/pred_gpkg_lyr/pred_gpkg_id_col "
                       "are not all configured; cannot locate receiver geometries. Skipping.")
        sys.exit(0)
    if not path_hf_finl_gpkg_raw:
        logging.error("donor_map_states resolved to region(s) but path_hf_finl_gpkg is not configured; "
                       "cannot locate donor gage geometries. Skipping.")
        sys.exit(0)

    resp_vars = pred_cfg.pred_cfg_dict.get('algo_response_vars', [])
    algos = pred_cfg.pred_cfg_dict.get('algo_type', [])

    for ds in datasets:
        vals = {'dir_std_base': dir_std_base, 'ds': ds, 'home_dir': home_dir}

        path_gpkg_pred_rslv = Path(str(path_gpkg_pred).format(**vals))
        if not path_gpkg_pred_rslv.exists():
            logging.warning(f"Prediction-locations GPKG not found: {path_gpkg_pred_rslv}. Skipping {ds}.")
            continue
        gdf_recv_all = gpd.read_file(path_gpkg_pred_rslv, layer=pred_gpkg_lyr, engine='pyogrio')
        if pred_gpkg_id_col not in gdf_recv_all.columns:
            logging.error(f"'{pred_gpkg_id_col}' not found in {pred_gpkg_lyr} layer of {path_gpkg_pred_rslv}. Skipping {ds}.")
            continue
        # This region filter is designed for USGS WBD HUC12 layers (12-digit ids, a
        # 'states' column) -- the same schema `pred_gpkg_pred` already points to
        # elsewhere in this workflow. zfill is a no-op for already-correct-length ids.
        gdf_recv_all[pred_gpkg_id_col] = gdf_recv_all[pred_gpkg_id_col].astype(str).str.zfill(12)
        if gdf_recv_all.crs is None:
            gdf_recv_all = gdf_recv_all.set_crs(epsg=4326)

        if 'states' not in gdf_recv_all.columns:
            logging.error(f"'donor_map_states' requires a 'states' column in the {pred_gpkg_lyr} layer "
                           f"(as in USGS WBD HUC12 layers) to filter by region; not found. Skipping {ds}.")
            continue

        path_hf_finl_gpkg = Path(raftsutil.resolve_fstrings(path_hf_finl_gpkg_raw, vals))
        if not path_hf_finl_gpkg.exists():
            logging.warning(f"Master hydrofabric GPKG not found: {path_hf_finl_gpkg}. Skipping {ds}.")
            continue
        try:
            hl = gpd.read_file(path_hf_finl_gpkg, layer='hydrolocations', engine='pyogrio')
        except Exception as e:
            logging.warning(f"Could not read 'hydrolocations' layer from {path_hf_finl_gpkg}: {e}. Skipping {ds}.")
            continue
        if not {'hl_type', 'hl_link'}.issubset(hl.columns):
            logging.warning(f"'hydrolocations' layer in {path_hf_finl_gpkg} lacks hl_type/hl_link columns "
                             f"needed to locate donor gages. Skipping {ds}.")
            continue

        hl_nwis = hl[hl['hl_type'].astype(str).str.contains('nwis', na=False)].copy()
        # hl_link may combine multiple agencies' identifiers pipe-delimited (e.g.
        # '02378170|2430166'), so split and match against each part individually --
        # confirmed empirically: an exact-match lookup against the raw field silently
        # dropped a donor gage (and, with it, 255 of 1258 receivers) that only matched
        # after this split.
        hl_nwis['gage_id_parts'] = hl_nwis['hl_link'].astype(str).str.split('|')
        hl_nwis_exp = hl_nwis.explode('gage_id_parts')
        hl_nwis_exp['gage_id'] = hl_nwis_exp['gage_id_parts'].astype(str).str.zfill(8)
        if hl_nwis_exp.crs is None:
            hl_nwis_exp = hl_nwis_exp.set_crs(epsg=4326)

        # Optional enhancement: for HUC12s with no donor pairing because a larger
        # hydrofabric divide swallowed their crosswalk entry (see
        # scripts/qa/crosswalk_gap_diagnosis/), substitute the divide's own geometry as a
        # proxy receiver instead of leaving a hatched gap -- see
        # build_divide_proxy_receivers(). Best-effort: any failure here just falls back
        # to the existing hatched-gap behavior, since this is on top of an already-useful map.
        divide_id_col = pred_cfg.pred_cfg_dict.get('crosswalk_target_col') or 'divide_id'
        gdf_divides_ds = None
        divide_to_assigned_huc = None
        divide_area_sqkm = None
        path_crosswalk_ids_raw = pred_cfg.pred_cfg_dict.get('path_crosswalk_ids')
        if path_crosswalk_ids_raw:
            try:
                path_crosswalk_ids = Path(raftsutil.resolve_fstrings(path_crosswalk_ids_raw, vals))
                df_crosswalk = pd.read_csv(path_crosswalk_ids) if path_crosswalk_ids.suffix == '.csv' \
                    else pd.read_parquet(path_crosswalk_ids, columns=[divide_id_col, pred_gpkg_id_col])
                df_crosswalk[divide_id_col] = df_crosswalk[divide_id_col].astype(str)
                df_crosswalk[pred_gpkg_id_col] = df_crosswalk[pred_gpkg_id_col].astype(str).str.zfill(12)
                # The crosswalk is documented elsewhere as strictly 1:1 (one row per
                # divide_id), but that's a data-quality expectation, not something
                # enforced here -- a duplicated divide_id would make the two .map()
                # calls below raise InvalidIndexError deep inside build_divide_proxy_receivers
                # instead of failing loudly where the data was actually loaded. Keep
                # the first occurrence and warn, rather than let a corrupt crosswalk
                # silently disable this whole best-effort enhancement.
                if df_crosswalk[divide_id_col].duplicated().any():
                    n_dup = int(df_crosswalk[divide_id_col].duplicated().sum())
                    logging.warning(f"Crosswalk has {n_dup:,} duplicate {divide_id_col} rows "
                                     f"(expected strictly 1:1); keeping the first occurrence of each.")
                    df_crosswalk = df_crosswalk.drop_duplicates(divide_id_col, keep='first')
                divide_to_assigned_huc = df_crosswalk.set_index(divide_id_col)[pred_gpkg_id_col]

                gdf_divides_ds = resolve_divides_layer(pred_cfg.pred_cfg_dict, vals, divide_id_col)
                if gdf_divides_ds[divide_id_col].duplicated().any():
                    n_dup = int(gdf_divides_ds[divide_id_col].duplicated().sum())
                    logging.warning(f"Divides layer has {n_dup:,} duplicate {divide_id_col} rows; "
                                     f"keeping the first occurrence of each.")
                    gdf_divides_ds = gdf_divides_ds.drop_duplicates(divide_id_col, keep='first')
                divide_area_sqkm = gdf_divides_ds.to_crs('EPSG:5070').set_index(divide_id_col).geometry.area / 1e6
                logging.info(f"Loaded {len(gdf_divides_ds):,} divides for the divide-proxy receiver enhancement ({ds}).")
            except Exception as e:
                logging.warning(f"Divide-proxy receiver enhancement unavailable for {ds}: {e}. "
                                 f"Unpaired HUC12s will render as plain hatched gaps.")
                gdf_divides_ds = None

        for region_id, region_spec in sub_region_loop:
            dirs_std_dict = raftsutil.rafts_save_algo_dir_struct(
                dir_base, region=region_id, scheme=sub_region_scheme)
            dir_out = dirs_std_dict.get('dir_out')
            dir_out_viz_base = dirs_std_dict.get('dir_out_viz_base')
            dir_regionalization = Path(dir_out) / "regionalization"

            dir_regn_ds = dir_regionalization / ds
            for resp_var in resp_vars:
                # Discover algos by which donor_pairs_*.csv files actually exist -- i.e. only
                # algo variants where donor-receiver pairing actually ran and succeeded, not
                # just the base algo strings listed in the config.
                dynamic_algos = raftsutil.discover_dynamic_algos(
                    search_dir=dir_regn_ds, base_algos=algos, metric=resp_var,
                    dataset_id=ds, file_prefix="donor_pairs_", file_extension=".csv"
                )
                if not dynamic_algos:
                    logging.info(f"No donor_pairs files found for {resp_var} in {dir_regn_ds}; "
                                 f"skipping (pairing may not have been run yet for {ds}/{resp_var}).")
                    continue

                for algo_str in dynamic_algos:
                    path_pairs = raftsutil.std_donor_pairs_path(dir_regionalization, ds, algo_str, resp_var)
                    if not path_pairs.exists():
                        logging.warning(f"Donor pairs file not found: {path_pairs}. Skipping.")
                        continue

                    # Read once per algo/response-var; region filtering below is cheap and
                    # avoids re-reading this file (up to several MB) once per region.
                    dp = pd.read_csv(path_pairs, dtype={'receiver_id': str, 'donor_id': str})
                    dp['receiver_id'] = dp['receiver_id'].str.zfill(12)
                    dp['donor_id'] = dp['donor_id'].str.zfill(8)
                    dp = dp.drop_duplicates('receiver_id')

                    for region_str, region_states in regions.items():
                        gdf_region = gdf_recv_all[raftsregions.states_mask(gdf_recv_all, 'states', region_states)]
                        if gdf_region.empty:
                            logging.warning(f"No {pred_gpkg_lyr} features matched region {region_str}={region_states} "
                                             f"for {ds}. Skipping.")
                            continue

                        region_pairs = dp[dp['receiver_id'].isin(set(gdf_region[pred_gpkg_id_col]))]
                        if region_pairs.empty:
                            logging.warning(f"No donor pairings matched region {region_str} for {ds}/{algo_str}/{resp_var}. Skipping.")
                            continue

                        gdf_region_run = gdf_region.merge(region_pairs[['receiver_id', 'cluster_id', 'donor_id']],
                                                           left_on=pred_gpkg_id_col, right_on='receiver_id', how='left')
                        gdf_region_run['cluster_id'] = gdf_region_run['cluster_id'].astype('Int64')
                        gdf_no_pairing = gdf_region_run[gdf_region_run['cluster_id'].isna()]

                        # Resolve as many of those unpaired HUC12s as possible into divide-shaped
                        # proxy receivers (see build_divide_proxy_receivers) before locating donor
                        # gages, so a borrowed donor from outside the region's own pairing set still
                        # gets included below. Wrapped in try/except so this stays genuinely
                        # best-effort (per the comment where gdf_divides_ds is loaded above) --
                        # an unexpected failure here (e.g. unresolvable geometry) falls back to
                        # plain hatched gaps for this one region/algo/resp_var instead of crashing
                        # the whole run.
                        gdf_proxy = gpd.GeoDataFrame(columns=['donor_id'], geometry=gpd.GeoSeries([], crs=gdf_region.crs))
                        if gdf_divides_ds is not None and not gdf_no_pairing.empty:
                            try:
                                gdf_proxy, gdf_no_pairing = build_divide_proxy_receivers(
                                    gdf_no_pairing, gdf_divides_ds, divide_to_assigned_huc, divide_area_sqkm,
                                    dp, pred_gpkg_id_col, divide_id_col,
                                )
                            except Exception as e:
                                logging.warning(f"Divide-proxy receiver resolution failed for {ds}/{algo_str}/"
                                                 f"{resp_var}/{region_str}: {e}. Falling back to plain hatched gaps.")

                        donors_needed = set(region_pairs['donor_id']) | set(gdf_proxy['donor_id'].dropna())
                        donors_pts = (hl_nwis_exp[hl_nwis_exp['gage_id'].isin(donors_needed)]
                                      .drop_duplicates('gage_id')[['gage_id', 'geometry']].copy())
                        missing_donor_ids = donors_needed - set(donors_pts['gage_id'])
                        if missing_donor_ids:
                            logging.warning(f"Could not locate {len(missing_donor_ids)} donor gage(s) in hydrolocations "
                                             f"for {ds}/{algo_str}/{resp_var}/{region_str}: {sorted(missing_donor_ids)}")

                        gdf_valid = gdf_region_run[gdf_region_run['cluster_id'].notna()
                                                    & gdf_region_run['donor_id'].isin(donors_pts['gage_id'])].copy()
                        gdf_valid['is_divide_proxy'] = False
                        if not gdf_proxy.empty:
                            gdf_proxy = gdf_proxy[gdf_proxy['donor_id'].isin(donors_pts['gage_id'])]
                        if gdf_valid.empty and gdf_proxy.empty:
                            logging.warning(f"No fully-locatable donor-receiver pairs for {ds}/{algo_str}/{resp_var} "
                                             f"in region {region_str}. Skipping.")
                            continue

                        gdf_receivers = gpd.GeoDataFrame(
                            pd.concat([gdf_valid, gdf_proxy], ignore_index=True), geometry='geometry', crs=gdf_valid.crs)

                        logging.info(f"Mapping donor-receiver pairing for {ds}/{algo_str}/{resp_var} in {region_str}: "
                                     f"{len(gdf_receivers)} of {len(gdf_region)} receivers paired to "
                                     f"{gdf_receivers['donor_id'].nunique()} donors "
                                     f"({len(gdf_proxy)} via divide-shaped proxy).")

                        raftsplot.plot_donor_receiver_map_wrap(
                            gdf_receivers=gdf_receivers, gdf_donors=donors_pts, dir_out_viz_base=dir_out_viz_base,
                            ds=ds, metr=resp_var, algo_str=algo_str, region_str=region_str,
                            gdf_no_pairing=gdf_no_pairing,
                        )

    logging.info(f"Completed donor-receiver map generation for {path_pred_config}")
    logging.shutdown()
