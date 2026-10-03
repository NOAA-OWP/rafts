"""qa_check_huc12_divide_reassignment.py

Follow-up diagnostic to qa_check_missing_huc12_divide_existence.py's
'divides_exist_not_crosswalked' bucket: for a HUC12 that has no crosswalk
entry despite a real hydrofabric divide sitting inside it, where did that
divide actually get crosswalked to instead? The huc12<->divide_id crosswalk
is strictly 1:1 (one row per divide_id), so any divide overlapping a
"missing" HUC12 is either entirely absent from the crosswalk, or was
assigned to some *other* HUC12.

For each such divide this reports TWO different overlap metrics, because
they answer different questions and can disagree:
    - frac_of_*_huc: how much of the HUC12's own area does this divide
      fill? (the metric qa_check_missing_huc12_divide_existence.py uses to
      decide "does a divide exist here")
    - frac_of_divide_in_*_huc: how much of the DIVIDE's own footprint falls
      inside this HUC12? This is the metric that actually explains a
      "majority share" crosswalk assignment rule -- confirmed empirically:
      an early version of this check compared frac_of_*_huc instead and
      flagged 57 of 100 FL pairs as apparent defects, but the top offenders
      were all the same single divide (a ~4,089 sqkm hfv4 "landscape"-type
      divide near the Everglades, 42x the region's median HUC12 area)
      covering >99% of FOUR different HUC12s' own area while its actual
      crosswalk assignment held only 82.75% of ITS OWN area from that
      divide -- frac_of_*_huc simply cannot distinguish "this HUC12 is
      small and the divide swallows it" from "this HUC12 is where most of
      the divide actually sits." frac_of_divide_in_*_huc does, and by that
      metric the assigned HUC12 correctly holds the largest share.

Classification is based on frac_of_divide_in_*_huc:
    - divide_itself_not_in_crosswalk: the divide has no crosswalk row at
      all -- a deeper gap than "assigned elsewhere."
    - assigned_elsewhere_majority_share_of_divide: the HUC12 the divide was
      actually assigned to holds >= the share of the divide's footprint
      that the missing HUC12 does. Consistent with a majority-share
      assignment rule -- for a divide much larger than a single HUC12, this
      is an expected consequence of a 1:1 crosswalk, not a defect.
    - assigned_elsewhere_minority_share_of_divide: the assigned HUC12 holds
      a *smaller* share of the divide's footprint than the missing one
      does, yet was still chosen. This is inconsistent with a majority-share
      rule and is the strongest signal of an actual crosswalk-build defect
      (stale source data, a different/buggy assignment rule, etc.).
    - indeterminate_missing_area_data: the divide-normalized share couldn't
      be computed for the missing HUC12, the assigned HUC12, or both (e.g.
      a zero/negative divide area, or an assigned HUC12 id that doesn't
      resolve in the loaded HUC12 layer at all). Reported separately rather
      than risking a false majority/minority call -- NaN comparisons are
      always False in Python, so without this bucket such a row would
      silently fall into "minority_share" (the strongest-defect signal)
      regardless of whether a real defect exists.

Path resolution: identical to qa_check_missing_huc12_divide_existence.py --
--path_pred_config is the only required input; everything else is read from
the pred_config via rafts_algo.qa_utils.

Exit code is always 0 (see scripts/qa/README.md): this is a diagnostic, not
a pass/fail gate, so it never aborts a calling shell pipeline.

Example:
    >>> cd /path/to/rafts/pkg
    >>> uv run python rafts_algo/flow/qa/qa_check_huc12_divide_reassignment.py \\
    ...     --path_pred_config ../scripts/workflow_configs/regn_fy26_hfv4_conus_clst/regn_casam_pred_config_hf4.yaml \\
    ...     --states FL

# Changelog/Contributions
    2026-09-22 Originally created, GL
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
from rafts_algo.flow.qa.qa_check_missing_huc12_divide_existence import is_conus_huc12


def run(path_pred_config: Path, states: list = None, min_coverage_frac: float = 0.05,
        path_out_csv: Path = None, top_n_plot: int = 10, id_zfill_width: int = 12) -> int:
    """Run the HUC12-divide reassignment diagnostic.

    :param top_n_plot: How many of the strongest-evidence defect rows (by
        'margin_frac_divide', descending) to render as visual close-up panels,
        defaults to 10. Pass 0 to skip plotting.
    :type top_n_plot: int, optional
    :param id_zfill_width: Zero-pad the aggregation-unit id (pred_config's
        pred_gpkg_id_col) to this width before comparing against the crosswalk --
        12 for standard HUC12 codes. Pass 0/None if the configured id column isn't
        a fixed-width zero-padded code. Defaults to 12.
    :type id_zfill_width: int, optional
    :return: Count of rows classified 'assigned_elsewhere_smaller_overlap_there'
        -- the strongest evidence of an actual crosswalk-build defect (0 means
        every explainable divide either has no crosswalk entry at all, or was
        assigned to the HUC12 it overlaps most, consistent with a one-divide-
        spans-many-HUC12s limitation rather than a defect).
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
        gdf_huc12_all = resolve_huc12_layer(pred_cfg_dict, context, id_zfill_width=id_zfill_width)
    except (FileNotFoundError, KeyError) as e:
        print(f"[SKIP] Could not read HUC12 layer: {e}")
        return 0

    if states:
        states_norm = [s.strip().upper() for s in states]
        gdf_region = gdf_huc12_all[raftsregions.states_mask(gdf_huc12_all, 'states', states_norm)]
        region_label = "-".join(states_norm)
    else:
        gdf_region = gdf_huc12_all[is_conus_huc12(gdf_huc12_all, huc12_col=huc12_col)]
        region_label = "CONUS"

    if gdf_region.empty:
        print(f"[SKIP] No HUC12 features matched region={region_label}.")
        return 0

    df_crosswalk = pd.read_parquet(ctx.path_crosswalk_ids, columns=[divide_id_col, pred_gpkg_id_col])
    df_crosswalk[divide_id_col] = df_crosswalk[divide_id_col].astype(str)
    df_crosswalk[pred_gpkg_id_col] = df_crosswalk[pred_gpkg_id_col].astype(str)
    if id_zfill_width:
        df_crosswalk[pred_gpkg_id_col] = df_crosswalk[pred_gpkg_id_col].str.zfill(id_zfill_width)
    divide_to_assigned_huc = df_crosswalk.set_index(divide_id_col)[pred_gpkg_id_col]
    crosswalk_hucs = set(df_crosswalk[pred_gpkg_id_col])

    gdf_missing = gdf_region[~gdf_region[huc12_col].isin(crosswalk_hucs)].copy()
    print(f"path_pred_config: {path_pred_config}")
    print(f"path_crosswalk_ids: {ctx.path_crosswalk_ids}")
    print(f"Region: {region_label} ({len(gdf_region):,} HUC12s in scope, "
          f"median area {gdf_region['areasqkm'].median():.1f} sqkm)")
    print(f"Missing from crosswalk: {len(gdf_missing):,} of {len(gdf_region):,}\n")

    if gdf_missing.empty:
        print("None -- every HUC12 in this region has a crosswalk entry. Nothing to reassign-diagnose.")
        return 0

    try:
        gdf_divides = resolve_divides_layer(pred_cfg_dict, context, divide_id_col)
    except FileNotFoundError as e:
        print(f"[SKIP] Could not read divides layer: {e}")
        return 0
    print(f"Read {len(gdf_divides):,} divides for spatial join (this can take a while at CONUS scale)...")

    df_ov_missing = find_huc12_divide_overlaps(gdf_missing, gdf_divides, divide_id_col,
                                                min_frac=min_coverage_frac, huc12_col=huc12_col)
    df_ov_missing = df_ov_missing.rename(columns={'frac_of_huc': 'frac_of_missing_huc'})
    if df_ov_missing.empty:
        print(f"No divide covers >= {min_coverage_frac:.0%} of any missing HUC12's area in this region "
              f"-- nothing to reassign-diagnose (see qa_check_missing_huc12_divide_existence.py's "
              f"'no_divide_overlap' bucket instead).")
        return 0

    # Divide footprint size (sqkm), for interpreting whether an apparent
    # discrepancy is simply because this divide is much bigger than a typical
    # HUC12 (see module docstring).
    equal_area_crs = 'EPSG:5070'
    divide_area_sqkm = (gdf_divides.set_index(divide_id_col).to_crs(equal_area_crs).geometry.area / 1e6)

    # HUC12 areas (sqkm), for both the missing HUC12s and whatever HUC12s the
    # candidate divides were actually assigned to -- needed to convert the
    # huc-normalized overlap fractions above into absolute intersection areas,
    # and from there into divide-normalized shares (see module docstring for
    # why that's the metric that actually explains the assignment).
    huc_area_sqkm = (gdf_huc12_all.set_index(huc12_col).to_crs(equal_area_crs).geometry.area / 1e6)

    # Assigned-HUC12 overlap, computed only for the (small) set of candidate
    # divides found above -- not a bulk join against the full HUC12 layer.
    divide_ids_needed = set(df_ov_missing[divide_id_col])
    assigned_hucs_needed = {divide_to_assigned_huc.get(d) for d in divide_ids_needed} - {None}
    gdf_assigned_hucs = gdf_huc12_all[gdf_huc12_all[huc12_col].isin(assigned_hucs_needed)]
    gdf_divides_needed = gdf_divides[gdf_divides[divide_id_col].isin(divide_ids_needed)]
    df_ov_assigned = find_huc12_divide_overlaps(gdf_assigned_hucs, gdf_divides_needed, divide_id_col,
                                                 min_frac=0.0, huc12_col=huc12_col)
    frac_of_assigned_huc = df_ov_assigned.set_index([divide_id_col, huc12_col])['frac_of_huc']

    rows = []
    for row in df_ov_missing.itertuples(index=False):
        divide_id = getattr(row, divide_id_col)
        missing_huc = getattr(row, huc12_col)
        assigned_huc = divide_to_assigned_huc.get(divide_id)
        d_area = float(divide_area_sqkm.get(divide_id, float('nan')))

        missing_huc_area = float(huc_area_sqkm.get(missing_huc, float('nan')))
        inter_area_missing = row.frac_of_missing_huc * missing_huc_area
        frac_divide_missing = inter_area_missing / d_area if d_area > 0 else float('nan')

        if assigned_huc is None:
            rows.append((missing_huc, divide_id, row.frac_of_missing_huc, frac_divide_missing,
                         None, None, None, d_area, 'divide_itself_not_in_crosswalk'))
            continue

        frac_assigned = frac_of_assigned_huc.get((divide_id, assigned_huc), 0.0)
        assigned_huc_area = float(huc_area_sqkm.get(assigned_huc, float('nan')))
        inter_area_assigned = frac_assigned * assigned_huc_area
        frac_divide_assigned = inter_area_assigned / d_area if d_area > 0 else float('nan')

        # A NaN on either side (d_area <= 0, or assigned_huc not resolving in
        # huc_area_sqkm -- e.g. an id-format mismatch between the crosswalk and the
        # loaded HUC12 layer) must not reach the >= comparison below: Python's NaN
        # comparisons are always False, so `frac_divide_assigned >= frac_divide_missing`
        # would silently fall into the else branch and mislabel a case where the
        # share just couldn't be computed as 'assigned_elsewhere_minority_share_of_divide'
        # -- this script's own strongest "confirmed crosswalk defect" signal -- when
        # it's actually missing data, not evidence of anything.
        if pd.isna(frac_divide_missing) or pd.isna(frac_divide_assigned):
            classification = 'indeterminate_missing_area_data'
        else:
            classification = ('assigned_elsewhere_majority_share_of_divide'
                               if frac_divide_assigned >= frac_divide_missing
                               else 'assigned_elsewhere_minority_share_of_divide')
        rows.append((missing_huc, divide_id, row.frac_of_missing_huc, frac_divide_missing,
                     assigned_huc, frac_assigned, frac_divide_assigned, d_area, classification))

    result = pd.DataFrame(rows, columns=[
        huc12_col, divide_id_col, 'frac_of_missing_huc', 'frac_of_divide_in_missing_huc',
        'assigned_huc12', 'frac_of_assigned_huc', 'frac_of_divide_in_assigned_huc',
        'divide_area_sqkm', 'classification',
    ])
    # How much of the divide's footprint the missing HUC12 holds beyond what the
    # assigned HUC12 holds -- a small margin (e.g. lnd-573107 above: 2.4% vs 2.2%,
    # margin 0.2pp) is more likely noise from a multi-way contest (this divide
    # overlaps far more than just these two HUC12s; the pairwise test here never
    # sees the actual winner if it's a third one) than a real defect. A large
    # margin is the clearer, more actionable signal.
    result['margin_frac_divide'] = result['frac_of_divide_in_missing_huc'] - result['frac_of_divide_in_assigned_huc']
    result = result.sort_values('margin_frac_divide', ascending=False, na_position='last').reset_index(drop=True)

    counts = result['classification'].value_counts()
    print("\nClassification of missing-HUC12 <-> overlapping-divide pairs (by majority share of the divide):")
    for cls, n in counts.items():
        print(f"  {cls}: {n:,} ({100 * n / len(result):.1f}%)")

    median_huc_area = gdf_region['areasqkm'].median()
    large_divides = result[result['divide_area_sqkm'] > 3 * median_huc_area]
    if not large_divides.empty:
        print(f"\n(FYI: {len(large_divides):,} of those rows involve a divide larger than 3x the region's "
              f"median HUC12 area ({median_huc_area:.1f} sqkm) -- the majority-share classification above "
              f"already accounts for this, unlike a simpler same-area comparison would.)")

    n_true_gap = int((result['classification'] == 'assigned_elsewhere_minority_share_of_divide').sum())
    print(f"\nStrongest evidence of an actual crosswalk-build defect "
          f"(assigned elsewhere despite holding a SMALLER share of the divide): {n_true_gap:,} of {len(result):,}")
    if n_true_gap:
        print(result[result['classification'] == 'assigned_elsewhere_minority_share_of_divide'].head(10).to_string(index=False))

    if path_out_csv is None:
        path_out_csv = ctx.dir_qa_out / f"qa_out_huc12_divide_reassignment_{region_label}_{path_pred_config.stem}.csv"
    path_out_csv = Path(path_out_csv)
    path_out_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path_out_csv, index=False)
    print(f"\nWrote full reassignment diagnostic to {path_out_csv}")

    # Visual close-up panels for the strongest-evidence defects, for human inspection
    # alongside the CSV -- each panel shows the divide's real footprint against both
    # HUC12s so the geometry itself, not just the area-fraction numbers, can be checked.
    # Only the 'minority_share' bucket is plotted (the actionable, likely-defect rows);
    # 'majority_share' rows are already explained by a large-divide/1:1-crosswalk
    # limitation and 'divide_itself_not_in_crosswalk' rows have no assigned HUC12 to
    # compare against -- both are still fully captured in the CSV above.
    if top_n_plot > 0 and n_true_gap:
        try:
            df_defects = result[result['classification'] == 'assigned_elsewhere_minority_share_of_divide']
            path_plot = raftsplot.plot_divide_reassignment_defects_wrap(
                df_defects=df_defects, gdf_missing_huc=gdf_missing, gdf_assigned_huc=gdf_assigned_hucs,
                gdf_divides=gdf_divides_needed, divide_id_col=divide_id_col, dir_out_qa=ctx.dir_qa_out,
                region_str=region_label, top_n=top_n_plot, huc12_col=huc12_col,
            )
            print(f"Wrote top-{min(top_n_plot, n_true_gap)} defect close-up panels to {path_plot}")
        except Exception as e:
            # Plotting is a nice-to-have on top of the CSV, which is already written
            # above -- never let a rendering failure turn a successful diagnostic run
            # into one that looks like it failed.
            print(f"[WARN] Could not render defect close-up panels: {e}")

    return n_true_gap


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--path_pred_config", type=Path, required=True,
                         help="Path to a prediction config YAML.")
    parser.add_argument("--states", type=str, nargs='+', default=['FL'],
                         help="2-letter state codes to scope the check to. Defaults to ['FL'] "
                              "(this diagnostic is a close-look follow-up, not a CONUS-wide sizing pass -- "
                              "see qa_check_missing_huc12_divide_existence.py for that).")
    parser.add_argument("--min_coverage_frac", type=float, default=0.05,
                         help="Minimum fraction of a missing HUC12's own area a divide must cover to be "
                              "included here. Should match the value used with "
                              "qa_check_missing_huc12_divide_existence.py for consistent results. Default 0.05 (5%%).")
    parser.add_argument("--path_out_csv", type=Path, default=None,
                         help="Where to write the reassignment CSV. Defaults to "
                              "dir_out/analysis/<dataset_name>/qa_out_huc12_divide_reassignment_<region>_<pred_config_stem>.csv.")
    parser.add_argument("--top_n_plot", type=int, default=10,
                         help="How many of the strongest-evidence defect rows to render as visual "
                              "close-up panels (divide footprint vs. both HUC12 boundaries). Default 10. "
                              "Pass 0 to skip plotting.")
    parser.add_argument("--id_zfill_width", type=int, default=12,
                         help="Zero-pad the aggregation-unit id (pred_config's pred_gpkg_id_col) to this "
                              "width before comparing against the crosswalk -- 12 for standard HUC12 codes. "
                              "Pass 0 if the configured id column isn't a fixed-width zero-padded code. Default 12.")
    args = parser.parse_args()
    n_true_gap = run(args.path_pred_config.expanduser(), args.states, args.min_coverage_frac,
                      args.path_out_csv, args.top_n_plot, args.id_zfill_width)
    sys.exit(0)  # diagnostic only -- see scripts/qa/README.md
