"""qa_check_region_coverage.py

QA check for a configured `regions:` block (see rafts_algo.regions): are the sub-regions a
clean partition of the hydrofabric, and does each have enough donors to train on?

Three checks, modeled on the existing crosswalk-coverage QA scripts in this directory (same
two-way set-difference shape, same always-exit-0 diagnostic convention):

    overlaps: a divide_id assigned to more than one region_id in the regions crosswalk
        (path_regions_crosswalk) -- always reported, since RegionSpec membership is meant to
        be an exact, non-overlapping assignment (unlike donor *eligibility*, which is
        deliberately inclusive -- see rafts_algo.regions.donor_gage_ids_for_region's
        docstring for that distinction).
    gaps: a divide in the hydrofabric GPKG with no region_id at all in the crosswalk.
        Severity is 'info' when `regions.include_conus` is true (the default -- an
        unregioned divide is expected, not an error) and 'error' otherwise.
    too_few_donors: a region whose divide_id count falls below `regions.min_train_gages`.
        This is a rough proxy, not an exact donor-gage count -- the real gage-basin mapping
        used at training time (rafts_algo.regions.donor_gage_ids_for_region) needs the
        attribute config's standardized dataset, which this pred_config-only check doesn't
        load. Treat this as a quick heads-up, not a substitute for the `fallback` policy
        rafts_proc_algo_pool.py already applies at training time.

Path resolution:
    --path_pred_config is the only required input. Its `regions:` block supplies
    path_regions_crosswalk and path_hf_finl_gpkg directly -- no extra derivation needed, since
    (unlike the huc12 crosswalk checks) regions.py's crosswalk mechanism doesn't vary by
    workflow layout.

Exit code is always 0 (see scripts/qa/README.md): this is a diagnostic, not a pass/fail gate.

Example:
    >>> cd /path/to/rafts/pkg
    >>> uv run python rafts_algo/flow/qa/qa_check_region_coverage.py \\
    ...     --path_pred_config ../scripts/workflow_configs/regn_fy26_hfv4_conus_clst/regn_casam_pred_config_hf4.yaml
"""
import argparse
import sys
from pathlib import Path

import rafts_algo.regions as raftsregions
import rafts_algo.utils as raftsutil
from rafts_algo.qa_utils import resolve_divides_layer
from rafts_algo.schemas.pydantic_schemas import PredConfig


def run(path_pred_config: Path) -> int:
    """Run the region-overlap, region-gap, and too-few-donors checks.

    :param path_pred_config: Path to a prediction config YAML with a `regions:` block.
    :type path_pred_config: Path
    :return: Count of overlap divide_ids plus error-severity gap divide_ids (0 means clean).
    :rtype: int
    """
    validated_pred_cfg = raftsutil.load_validated_config(path_pred_config, PredConfig)
    regions_cfg = validated_pred_cfg.regions

    if regions_cfg is None or regions_cfg.scheme is None:
        print(f"[SKIP] {path_pred_config.name} has no `regions:` block (or scheme is unset) -- nothing to check.")
        return 0
    if regions_cfg.scheme not in ('vpu', 'huc2', 'custom'):
        print(f"[SKIP] scheme='{regions_cfg.scheme}' has no crosswalk to check (states-scheme regions "
              f"don't use rafts_algo.regions' crosswalk mechanism).")
        return 0

    attr_cfig_dict = {'home_dir': None}
    try:
        path_attr_config = raftsutil.build_cfig_path(path_pred_config, validated_pred_cfg.name_attr_config)
        attr_cfig = raftsutil.AttrConfigAndVars(path_attr_config)
        attr_cfig._read_attr_config()
        context = {'home_dir': attr_cfig.attrs_cfg_dict.get('home_dir'),
                   'dir_base': attr_cfig.attrs_cfg_dict.get('dir_base')}
    except Exception as e:
        print(f"[WARN] Could not resolve the linked attr_config for f-string context ({e}); "
              f"proceeding with an empty context, which will fail if path fields use placeholders.")
        context = {}

    print(f"path_pred_config: {path_pred_config}")
    print(f"regions.scheme: {regions_cfg.scheme}")
    print(f"regions.path_regions_crosswalk: {regions_cfg.path_regions_crosswalk}")

    regions = raftsregions.load_regions(regions_cfg, context=context)
    print(f"Resolved {len(regions)} region(s): {[r.region_id for r in regions]}\n")

    # --- Overlaps ---
    overlaps = raftsregions.find_region_overlaps(regions_cfg, context=context)
    print(f"Overlaps (divide_id assigned to >1 region_id): {len(overlaps)}")
    if not overlaps.empty:
        print(overlaps.head(10).to_string(index=False))
    n_overlap_ids = len(overlaps)

    # --- Gaps: every divide in the hydrofabric GPKG vs. every divide in the crosswalk ---
    # find_region_gaps calls assign_to_region internally, which raises on exactly the
    # condition find_region_overlaps just reported above -- a crosswalk with real overlaps
    # can't be meaningfully gap-checked via per-region divide_ids until those are resolved,
    # so skip gracefully (reporting the overlaps already found) rather than letting a
    # diagnostic script crash on the very inconsistency it exists to surface.
    n_error_gaps = 0
    if n_overlap_ids > 0:
        print(f"\nGaps: skipped -- {n_overlap_ids} overlapping divide_id(s) found above must be "
              f"resolved in {regions_cfg.path_regions_crosswalk} before gaps can be checked.")
    else:
        gdf_divides = resolve_divides_layer(
            {'path_hf_finl_gpkg': regions_cfg.path_hf_finl_gpkg, 'layr_hf_finl_gpkg': regions_cfg.layr_hf_finl_gpkg},
            context, regions_cfg.divide_id_col)
        gaps = raftsregions.find_region_gaps(
            gdf_divides, regions, id_col=regions_cfg.divide_id_col, include_conus=regions_cfg.include_conus)
        n_error_gaps = int((gaps['severity'] == 'error').sum()) if not gaps.empty else 0
        pct = 100 * len(gaps) / max(len(gdf_divides), 1)
        print(f"\nGaps (divide in hydrofabric GPKG, no region_id in crosswalk): {len(gaps):,} of "
              f"{len(gdf_divides):,} ({pct:.3f}%), severity="
              f"{'info (regions.include_conus=true)' if regions_cfg.include_conus else 'error'}")

    # --- Too-few-donors (rough proxy -- see module docstring) ---
    print(f"\nPer-region divide_id count vs. min_train_gages={regions_cfg.min_train_gages} (rough proxy, "
          f"not an exact donor-gage count):")
    for region in regions:
        flag = " [LOW]" if len(region.divide_ids) < regions_cfg.min_train_gages else ""
        print(f"  {region.region_id}: {len(region.divide_ids):,} divides{flag}")

    return n_overlap_ids + n_error_gaps


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--path_pred_config", type=Path, required=True,
                         help="Path to a prediction config YAML with a `regions:` block.")
    args = parser.parse_args()
    run(args.path_pred_config.expanduser())
    sys.exit(0)  # diagnostic only -- see scripts/qa/README.md
