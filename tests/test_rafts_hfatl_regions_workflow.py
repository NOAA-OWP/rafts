"""Integration test for sub-regional training (rafts_algo.regions), layered on top of the
same 30-gage/1797-divide hfATLAS fixture test_rafts_hfatl_full_workflow.py uses.

Builds a real divide_id -> region_id crosswalk from the real hfv4_x30.gpkg divides layer
(grouped by its real `vpuid` column -- not hand-typed ids), covering two regions:
  - 'EAST' (vpuids 02/03N/03W/04/05): 15 of the 30 gages, 593 divides -- large enough to
    train normally (model_scope == 'region').
  - 'SMALL' (vpuids 16/17): 2 gages, 11 divides -- deliberately below min_train_gages, to
    exercise the fallback='parent' policy (model_scope == 'region_parent_fallback'), which
    falls back to the full 30-gage donor pool rather than failing or silently training on
    too few samples.
Gages in every other vpuid are left out of the crosswalk entirely (neither region claims
them) -- confirming an unregioned gage simply isn't used as training data for either named
region, without that being treated as an error. EAST and SMALL's divide_id sets turn out to
genuinely overlap (some SMALL-vpuid gages' basins share divides with EAST-vpuid gages', a
real nested-catchment relationship in this data) -- see test_crosswalk_fixture_shape for why
that's expected, not a bug.

This test's own config/crosswalk files are generated into a scratch directory and cleaned up
by this test, rather than committed as static fixtures -- the crosswalk's content depends on
hfv4_x30.gpkg's real divide_id/vpuid values, so generating it from that file at test time
(instead of hand-typing ~600 ids) is what keeps it correct if that file ever changes.
"""
import shutil
import subprocess
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
import yaml

HERE = Path(__file__).resolve().parent
DIR_REPO = HERE.parent
DIR_CONFIG = DIR_REPO / "tests" / "config" / "hfatl"
DIR_DATA = DIR_REPO / "tests" / "data"
DIR_RUN = DIR_DATA / "run_data"
DIR_PREP = DIR_REPO / "pkg" / "rafts_prep" / "rafts_prep" / "flow"
DIR_PY = DIR_REPO / "pkg" / "rafts_algo" / "rafts_algo" / "flow"

PREP_CONFIG = DIR_CONFIG / "hfatl_prep_config.yaml"
ATTR_CONFIG = DIR_CONFIG / "hfatl_attr_config.yaml"
PATH_HF_GPKG = DIR_DATA / "hfv4_x30.gpkg"

DATASET = "hfatl_test_x30"
DIR_STD_BASE = DIR_RUN / "input" / "user_data_std" / DATASET
DIR_ALGOS = DIR_RUN / "output" / "regions" / "custom"

EAST_VPUIDS = ['02', '03N', '03W', '04', '05']
SMALL_VPUIDS = ['16', '17']

# Files this test generates itself (see module docstring) and cleans up afterward.
PATH_CROSSWALK = DIR_CONFIG / "_test_regions_crosswalk.csv"
PATH_ALGO_CONFIG_REGN = DIR_CONFIG / "_test_hfatl_algo_config_regn.yaml"


def run_uv_command(script_path, *args):
    cmd = ["uv", "run", "--project", str(DIR_REPO / "pkg"), "python", str(script_path)] + list(args)
    result = subprocess.run(cmd, cwd=DIR_CONFIG, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        pytest.fail(f"Pipeline step failed: {' '.join(cmd)}\nSTDERR: {result.stderr}")
    return result


@pytest.fixture(scope="class", autouse=True)
def prepare_and_clean(request):
    """Run the shared prep+attribute-aggregation steps once (identical to
    test_rafts_hfatl_full_workflow.py's steps 1-2), build this test's crosswalk + algo
    config, then clean everything up afterward.
    """
    shutil.rmtree(DIR_RUN, ignore_errors=True)
    run_uv_command(DIR_CONFIG / "prep_regn_test_agg.py", str(PREP_CONFIG))
    run_uv_command(DIR_PREP / "rafts_agg_hfatl_basin.py",
                    "--path_prep_config", str(PREP_CONFIG), "--path_attr_config", str(ATTR_CONFIG))

    gdf_divides = gpd.read_file(PATH_HF_GPKG, layer='divides', engine='pyogrio')
    df_east = gdf_divides[gdf_divides['vpuid'].isin(EAST_VPUIDS)][['divide_id']].assign(region_id='EAST')
    df_small = gdf_divides[gdf_divides['vpuid'].isin(SMALL_VPUIDS)][['divide_id']].assign(region_id='SMALL')
    pd.concat([df_east, df_small], ignore_index=True).to_csv(PATH_CROSSWALK, index=False)

    with open(DIR_CONFIG / "hfatl_algo_config.yaml") as f:
        algo_cfg = yaml.safe_load(f)
    algo_cfg['regions'] = {
        'scheme': 'custom',
        'path_regions_crosswalk': str(PATH_CROSSWALK),
        'path_hf_finl_gpkg': str(PATH_HF_GPKG),
        'min_train_gages': 5,
        'fallback': 'parent',
    }
    with open(PATH_ALGO_CONFIG_REGN, 'w') as f:
        yaml.dump(algo_cfg, f)

    def _cleanup():
        shutil.rmtree(DIR_RUN, ignore_errors=True)
        PATH_CROSSWALK.unlink(missing_ok=True)
        PATH_ALGO_CONFIG_REGN.unlink(missing_ok=True)
    request.addfinalizer(_cleanup)


class TestHfatlRegionsWorkflow:

    def test_crosswalk_fixture_shape(self):
        """Sanity-check this test's own generated crosswalk before trusting results built on it.

        Note: EAST and SMALL's divide_id sets genuinely overlap here -- some SMALL-vpuid
        gages' basins share divides with EAST-vpuid gages' basins (real nested-catchment
        data, not a bug in this fixture or in rafts_algo.regions). That's exactly why donor
        eligibility (donor_mask/donor_gage_ids_for_region) is an inclusive `isin()` test
        per region, not assign_to_region's exclusive dict lookup -- the latter is reserved
        for contexts (receiver assignment, the region-coverage QA check) where exclusivity
        actually matters.
        """
        df_cw = pd.read_csv(PATH_CROSSWALK)
        gdf_divides = gpd.read_file(PATH_HF_GPKG, layer='divides', engine='pyogrio')

        east_gages = gdf_divides[gdf_divides['vpuid'].isin(EAST_VPUIDS)]['gageID_plain'].nunique()
        small_gages = gdf_divides[gdf_divides['vpuid'].isin(SMALL_VPUIDS)]['gageID_plain'].nunique()
        assert east_gages == 15, f"Expected 15 EAST gages, got {east_gages}"
        assert small_gages == 2, f"Expected 2 SMALL gages, got {small_gages}"
        assert set(df_cw.loc[df_cw['region_id'] == 'EAST', 'divide_id']) == set(
            gdf_divides[gdf_divides['vpuid'].isin(EAST_VPUIDS)]['divide_id'])
        assert set(df_cw.loc[df_cw['region_id'] == 'SMALL', 'divide_id']) == set(
            gdf_divides[gdf_divides['vpuid'].isin(SMALL_VPUIDS)]['divide_id'])

    def test_train_algorithms_with_regions(self):
        """Running rafts_proc_algo_pool.py against a `regions:`-configured algo config
        trains each region separately into its own output/regions/custom/{region_id}/ tree,
        with the tiny SMALL region falling back to the full donor pool instead of failing.
        """
        run_uv_command(DIR_PY / "rafts_proc_algo_pool.py", str(PATH_ALGO_CONFIG_REGN), "--chunk_size", "4")

        for region_id in ('EAST', 'SMALL'):
            dir_region_algos = DIR_ALGOS / region_id / "trained_algorithms" / DATASET
            joblib_files = list(dir_region_algos.glob("algo_*_cluster_labels__*.joblib"))
            assert joblib_files, f"No trained models found for region '{region_id}' in {dir_region_algos}"

            path_eval = dir_region_algos / f"algo_eval_{DATASET}.csv"
            assert path_eval.exists(), f"No eval summary for region '{region_id}' at {path_eval}"
            df_eval = pd.read_csv(path_eval)
            assert not df_eval.empty, f"Empty eval summary for region '{region_id}'"

    def test_single_region_cli_arg_trains_only_that_region(self):
        """--region SMALL trains only SMALL, leaving EAST's output tree untouched/absent."""
        shutil.rmtree(DIR_ALGOS, ignore_errors=True)
        run_uv_command(DIR_PY / "rafts_proc_algo_pool.py", str(PATH_ALGO_CONFIG_REGN),
                       "--chunk_size", "4", "--region", "SMALL")

        assert (DIR_ALGOS / "SMALL" / "trained_algorithms" / DATASET).exists()
        assert not (DIR_ALGOS / "EAST").exists(), "EAST should not have been trained with --region SMALL"
