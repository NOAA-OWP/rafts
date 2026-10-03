"""qa_check_missing_huc12_divide_existence.py

QA check: for every HUC12 with no entry in the huc12<->divide_id crosswalk
(a "missing" HUC12 -- it can never receive a prediction, since
rafts_regn_params_gpkg.py's crosswalk merge is an inner join), does the
hydrofabric actually contain a divide overlapping it?

This splits "missing" HUC12s into two very different buckets:
    - no_divide_overlap: no hydrofabric divide covers any real share of this
      HUC12's own area. Likely open water, an island, or a closed/
      non-contributing basin with no flowline-connected divide to begin
      with -- probably a structural, not fixable, gap.
    - divides_exist_not_crosswalked: one or more real divides cover a
      meaningful share of this HUC12, but none of them made it into the
      crosswalk. This is a real, actionable data-coverage gap -- see
      qa_check_huc12_divide_reassignment.py for the follow-up diagnostic
      that explains *where* those divides went instead.

Coverage is measured as (intersection area) / (HUC12's own area), NOT
normalized against the divide's area. This matters because hfv4 has some
divides that are much larger than a single HUC12 -- e.g. the Everglades'
large non-contributing ("landscape"-type, no flowline) divides span many
HUC12s at once. Normalizing by the divide's footprint would make a HUC12
that's fully covered by one such divide register as a negligible sliver of
that divide's total area and get wrongly discarded as "no overlap," even
though the HUC12 itself is completely covered.

Run scoped to one region (e.g. --states FL) for a close look, or with no
--states at all for a CONUS-wide sizing pass (excludes AK/HI/PR/VI/GU/AS and
Canada/Mexico-touching HUC12s, which are out of scope for a CONUS
hydrofabric and would otherwise dominate the "missing" count for reasons
unrelated to crosswalk coverage).

Path resolution:
    --path_pred_config is the only required input. path_crosswalk_ids,
    path_gpkg_pred/pred_gpkg_lyr/pred_gpkg_id_col (the WBD HUC12 layer used
    elsewhere in this workflow, e.g. rafts_map_donor_receiver.py), and
    path_hf_finl_gpkg/layr_hf_finl_gpkg (the hydrofabric divides layer) are
    all read from that config via rafts_algo.qa_utils.

Exit code is always 0 (see scripts/qa/README.md): this is a diagnostic, not
a pass/fail gate, so it never aborts a calling shell pipeline.

Example:
    >>> cd /path/to/rafts/pkg
    >>> uv run python rafts_algo/flow/qa/qa_check_missing_huc12_divide_existence.py \\
    ...     --path_pred_config ../scripts/workflow_configs/regn_fy26_hfv4_conus_clst/regn_casam_pred_config_hf4.yaml \\
    ...     --states FL

# Changelog/Contributions
    2026-09-22 Originally created, GL
    2026-09-22 fix: normalize overlap fraction by the HUC12's own area, not the
               divide's -- hfv4's large non-contributing divides (e.g. the
               Everglades) are much bigger than a single HUC12, GL
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import geopandas as gpd

import rafts_algo.regions as raftsregions
import rafts_algo.plots as raftsplot
from rafts_algo.qa_utils import (
    resolve_qa_context, resolve_huc12_layer, resolve_divides_layer, find_huc12_divide_overlaps
)

# Excluded from the CONUS-wide default scope (no --states given): a CONUS
# hydrofabric structurally has no divides here, so every HUC12 touching one
# of these would register as "missing" regardless of crosswalk coverage.
# Kept as a documented reference/secondary filter -- see is_conus_huc12()
# below for why this alone is not a reliable primary filter.
NON_CONUS_STATES = {'AK', 'HI', 'PR', 'VI', 'GU', 'AS', 'CN', 'MX'}

# USGS HUC2 region codes 01-18 are CONUS; 19=Alaska, 20=Hawaii, 21=Caribbean,
# 22=Pacific Islands/Oceania (Guam, American Samoa, etc).
CONUS_HUC2_CODES = {f'{i:02d}' for i in range(1, 19)}


def is_conus_huc12(gdf_huc12: gpd.GeoDataFrame, huc12_col: str = 'huc12') -> pd.Series:
    """Boolean mask for HUC12s in the CONUS-wide default scope.

    Filters on the HUC12 id's own 2-digit HUC2 region prefix rather than the WBD
    layer's free-text 'states' column -- confirmed empirically: HUC2 region 22
    (Pacific Islands/Oceania) rows had a blank/NaN 'states' field in this WBD
    data, so a `states`-regex exclusion with `na=False` silently let them
    through (NaN never matches the exclusion pattern) and stretched a CONUS
    map's bounding box out to include far-flung Pacific points. The HUC2
    prefix is always populated (it's the first 2 digits of the id itself), so
    it doesn't have this failure mode. `NON_CONUS_STATES` is applied too, as a
    secondary filter for any US-territory HUC12s that fall within a CONUS
    HUC2 code but are still tagged with a non-CONUS state abbreviation.

    :param gdf_huc12: HUC12 polygons with a `huc12_col` column and a 'states' column.
    :type gdf_huc12: gpd.GeoDataFrame
    :param huc12_col: Column name holding the HUC12 identifier, defaults to 'huc12'.
     Pass the workflow's actual `pred_gpkg_id_col` here if it differs.
    :type huc12_col: str, optional
    :return: True for rows in the CONUS-wide default scope.
    :rtype: pd.Series
    """
    huc2 = gdf_huc12[huc12_col].str[:2]
    is_conus_huc2 = huc2.isin(CONUS_HUC2_CODES)
    not_excluded_state = ~gdf_huc12['states'].astype(str).str.contains(
        '|'.join(NON_CONUS_STATES), regex=True, na=False)
    return is_conus_huc2 & not_excluded_state


def classify_missing_huc12s(gdf_missing_huc12: gpd.GeoDataFrame, gdf_divides: gpd.GeoDataFrame,
                             divide_id_col: str, min_coverage_frac: float = 0.05,
                             huc12_col: str = 'huc12') -> pd.DataFrame:
    """Classify each missing HUC12 by whether real hydrofabric divide coverage exists inside it.

    Uses an intersects-predicate spatial join to find candidate (huc12, divide)
    pairs cheaply via spatial indexing, then computes exact intersection area only
    for those candidates (not a full cross product). Coverage is summed across every
    divide overlapping a given HUC12 (several divides can tile one HUC12) and
    expressed as a fraction of *that HUC12's own area* -- see the module docstring
    for why the divide's area is the wrong denominator here.

    :param gdf_missing_huc12: HUC12 polygons with no crosswalk entry, with a `huc12_col` column.
    :type gdf_missing_huc12: gpd.GeoDataFrame
    :param gdf_divides: Hydrofabric divide polygons, with `divide_id_col`.
    :type gdf_divides: gpd.GeoDataFrame
    :param divide_id_col: Column name holding the divide identifier in `gdf_divides`.
    :type divide_id_col: str
    :param min_coverage_frac: Minimum fraction of a HUC12's own area that must be
        covered by real divide(s), summed, to count as "a divide exists here", defaults to 0.05
    :type min_coverage_frac: float, optional
    :param huc12_col: Column name holding the HUC12 identifier, defaults to 'huc12'.
     Pass the workflow's actual `pred_gpkg_id_col` here if it differs.
    :type huc12_col: str, optional
    :return: One row per missing HUC12: `huc12_col`, n_divides_overlapping,
        overlapping_divide_ids, total_coverage_frac, classification.
    :rtype: pd.DataFrame
    """
    base_result = gdf_missing_huc12[[huc12_col]].copy()
    base_result['n_divides_overlapping'] = 0
    base_result['overlapping_divide_ids'] = ''
    base_result['total_coverage_frac'] = 0.0
    base_result['classification'] = 'no_divide_overlap'

    # Noise floor (0.001 = 0.1% of the HUC12's area) discards boundary-vertex
    # digitization slivers; the real classification threshold is min_coverage_frac.
    df_ov = find_huc12_divide_overlaps(gdf_missing_huc12, gdf_divides, divide_id_col,
                                        min_frac=0.001, huc12_col=huc12_col)
    if df_ov.empty:
        return base_result

    agg = df_ov.groupby(huc12_col).agg(
        n_divides_overlapping=(divide_id_col, 'count'),
        overlapping_divide_ids=(divide_id_col, lambda s: ';'.join(s)),
        total_coverage_frac=('frac_of_huc', 'sum'),
    )
    agg['total_coverage_frac'] = agg['total_coverage_frac'].clip(upper=1.0)
    agg['classification'] = agg['total_coverage_frac'].apply(
        lambda f: 'divides_exist_not_crosswalked' if f >= min_coverage_frac else 'no_divide_overlap'
    )

    result = base_result.set_index(huc12_col)
    result.update(agg)
    result['n_divides_overlapping'] = result['n_divides_overlapping'].astype(int)
    return result.reset_index()


def run(path_pred_config: Path, states: list = None, min_coverage_frac: float = 0.05,
        path_out_csv: Path = None, id_zfill_width: int = 12) -> int:
    """Run the missing-HUC12 divide-existence check.

    :param id_zfill_width: Zero-pad the aggregation-unit id to this width (e.g. 12 for
        standard HUC12 codes) before comparing against the crosswalk, since a plain
        str() cast can otherwise drop a leading zero lost to an int/float dtype on
        read. Pass 0/None if the configured `pred_gpkg_id_col` isn't a fixed-width
        zero-padded code. Defaults to 12.
    :type id_zfill_width: int, optional
    :return: Count of missing HUC12s classified as 'divides_exist_not_crosswalked'
        (the actionable subset; 0 means every gap in scope is a real structural
        no-divide case, or there was no gap at all).
    :rtype: int
    """
    ctx = resolve_qa_context(path_pred_config)
    if ctx.path_crosswalk_ids is None:
        print(f"[SKIP] {path_pred_config.name} has no 'path_crosswalk_ids' set -- nothing to check.")
        return 0
    if not ctx.path_crosswalk_ids.exists():
        print(f"[SKIP] Crosswalk file does not exist: {ctx.path_crosswalk_ids}")
        return 0

    pred_cfg_dict = ctx.pred_cfg.pred_cfg_dict
    context = {'dir_std_base': str(pred_cfg_dict.get('dir_std_base')), 'home_dir': str(pred_cfg_dict.get('home_dir'))}
    pred_gpkg_id_col = pred_cfg_dict.get('pred_gpkg_id_col')
    divide_id_col = pred_cfg_dict.get('crosswalk_target_col') or 'divide_id'
    # The aggregation-unit id column is whatever the workflow's own pred_config
    # names it -- never assumed to be literally 'huc12'.
    huc12_col = pred_gpkg_id_col

    try:
        gdf_huc12 = resolve_huc12_layer(pred_cfg_dict, context, id_zfill_width=id_zfill_width)
    except (FileNotFoundError, KeyError) as e:
        print(f"[SKIP] Could not read HUC12 layer: {e}")
        return 0

    if states:
        states_norm = [s.strip().upper() for s in states]
        gdf_region = gdf_huc12[raftsregions.states_mask(gdf_huc12, 'states', states_norm)]
        region_label = "-".join(states_norm)
    else:
        gdf_region = gdf_huc12[is_conus_huc12(gdf_huc12, huc12_col=huc12_col)]
        region_label = "CONUS"

    if gdf_region.empty:
        print(f"[SKIP] No HUC12 features matched region={region_label}.")
        return 0

    df_crosswalk = pd.read_parquet(ctx.path_crosswalk_ids, columns=[pred_gpkg_id_col])
    crosswalk_hucs = df_crosswalk[pred_gpkg_id_col].astype(str)
    if id_zfill_width:
        crosswalk_hucs = crosswalk_hucs.str.zfill(id_zfill_width)
    crosswalk_hucs = set(crosswalk_hucs)

    gdf_missing = gdf_region[~gdf_region[huc12_col].isin(crosswalk_hucs)].copy()
    print(f"path_pred_config: {path_pred_config}")
    print(f"path_crosswalk_ids: {ctx.path_crosswalk_ids}")
    print(f"Region: {region_label} ({len(gdf_region):,} HUC12s in scope)")
    print(f"Missing from crosswalk: {len(gdf_missing):,} of {len(gdf_region):,}\n")

    if gdf_missing.empty:
        print("None -- every HUC12 in this region has a crosswalk entry.")
        return 0

    try:
        gdf_divides = resolve_divides_layer(pred_cfg_dict, context, divide_id_col)
    except FileNotFoundError as e:
        print(f"[SKIP] Could not read divides layer: {e}")
        return 0
    print(f"Read {len(gdf_divides):,} divides for spatial join (this can take a while at CONUS scale)...")

    result = classify_missing_huc12s(gdf_missing, gdf_divides, divide_id_col, min_coverage_frac, huc12_col=huc12_col)
    # Keep geometry through the merge (a plain column selection off a GeoDataFrame
    # drops it unless 'geometry' is explicitly included) -- the classification map
    # below needs it, and result would otherwise end up geometry-less.
    merge_cols = [huc12_col, 'states', 'areasqkm', 'geometry']
    result = result.merge(gdf_missing[merge_cols].astype({huc12_col: str}), on=huc12_col, how='left')
    result = gpd.GeoDataFrame(result, geometry='geometry', crs=gdf_missing.crs)
    for optional_col in ('hutype', 'tohuc', 'name'):
        if optional_col in gdf_missing.columns:
            result = result.merge(gdf_missing[[huc12_col, optional_col]], on=huc12_col, how='left')

    counts = result['classification'].value_counts()
    print("\nClassification of missing HUC12s:")
    for cls, n in counts.items():
        print(f"  {cls}: {n:,} ({100 * n / len(result):.1f}%)")

    if 'hutype' in result.columns:
        print("\nBy hutype:")
        print(pd.crosstab(result['hutype'], result['classification']).to_string())

    n_actionable = int((result['classification'] == 'divides_exist_not_crosswalked').sum())
    print(f"\nActionable (divides exist but weren't crosswalked): {n_actionable:,} of {len(result):,}")
    partial = result[(result['classification'] == 'divides_exist_not_crosswalked') & (result['total_coverage_frac'] < 0.95)]
    if not partial.empty:
        print(f"  ({len(partial):,} of those are only PARTIALLY covered (<95%) by the overlapping divide(s) -- "
              f"worth a closer look at whether the rest of the HUC12 is a real additional gap.)")

    if path_out_csv is None:
        path_out_csv = ctx.dir_qa_out / f"qa_out_missing_huc12_divide_existence_{region_label}_{path_pred_config.stem}.csv"
    path_out_csv = Path(path_out_csv)
    path_out_csv.parent.mkdir(parents=True, exist_ok=True)
    # Geometry excluded here (kept only in the in-memory `result` for the map below) --
    # a WKB/shapely-repr geometry column would bloat this CSV and isn't reader-friendly.
    pd.DataFrame(result.drop(columns='geometry')).to_csv(path_out_csv, index=False)
    print(f"\nWrote full classification to {path_out_csv}")

    # Visual plot of problem areas, for human inspection alongside the CSV -- a region
    # (or CONUS) map with actionable gaps in red and likely-structural gaps hatched blue.
    try:
        path_plot = raftsplot.plot_huc12_gap_classification_map_wrap(
            gdf_region=gdf_region, gdf_classified=result, dir_out_qa=ctx.dir_qa_out,
            dir_out_basemap=ctx.dir_out_viz_base, region_str=region_label,
        )
        print(f"Wrote classification map to {path_plot}")
    except Exception as e:
        # Plotting is a nice-to-have on top of the CSV, which is already written above --
        # never let a rendering failure (e.g. basemap download hiccup) turn a successful
        # diagnostic run into one that looks like it failed.
        print(f"[WARN] Could not render classification map: {e}")

    return n_actionable


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--path_pred_config", type=Path, required=True,
                         help="Path to a prediction config YAML.")
    parser.add_argument("--states", type=str, nargs='+', default=None,
                         help="2-letter state codes to scope the check to (e.g. --states FL). "
                              "Omit for a CONUS-wide sizing pass (excludes AK/HI/PR/VI/GU/AS and "
                              "Canada/Mexico-touching HUC12s).")
    parser.add_argument("--min_coverage_frac", type=float, default=0.05,
                         help="Minimum fraction of a HUC12's own area that must be covered by real "
                              "divide(s) (summed) to count as 'a divide exists here'. Default 0.05 (5%%). "
                              "Normalized against the HUC12's area, not the divide's -- see module docstring.")
    parser.add_argument("--path_out_csv", type=Path, default=None,
                         help="Where to write the classification CSV. Defaults to "
                              "dir_out/analysis/<dataset_name>/qa_out_missing_huc12_divide_existence_<region>_<pred_config_stem>.csv.")
    parser.add_argument("--id_zfill_width", type=int, default=12,
                         help="Zero-pad the aggregation-unit id (pred_config's pred_gpkg_id_col) to this "
                              "width before comparing against the crosswalk -- 12 for standard HUC12 codes. "
                              "Pass 0 if the configured id column isn't a fixed-width zero-padded code. Default 12.")
    args = parser.parse_args()
    n_actionable = run(args.path_pred_config.expanduser(), args.states, args.min_coverage_frac,
                        args.path_out_csv, args.id_zfill_width)
    sys.exit(0)  # diagnostic only -- see scripts/qa/README.md
