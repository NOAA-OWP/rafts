"""Unit tests for qa_check_missing_huc12_divide_existence.py.

Follows the project convention (see README.claude "Unit tests should avoid mocking" and
test_qa_utils.py/test_regions.py): real, small GeoDataFrames and a real temp-dir
attr_config/pred_config/GPKG chain -- no mocking.
"""
import tempfile
import unittest
from pathlib import Path

import geopandas as gpd
import pandas as pd
import yaml
from shapely.geometry import box

import rafts_algo.flow.qa.qa_check_missing_huc12_divide_existence as qa_missing


class TestIsConusHuc12(unittest.TestCase):

    def test_excludes_non_conus_huc2_prefix(self):
        gdf = pd.DataFrame({
            'huc12': ['010100000101', '220100000101'],  # HUC2 01 (conus), 22 (Pacific Islands)
            'states': ['ME', ''],
        })
        mask = qa_missing.is_conus_huc12(gdf, huc12_col='huc12')
        self.assertEqual(mask.tolist(), [True, False])

    def test_excludes_non_conus_state_even_with_conus_huc2_prefix(self):
        # A HUC2 prefix alone isn't a reliable filter -- the states column is a
        # secondary filter for territories tagged within a CONUS HUC2 code.
        gdf = pd.DataFrame({
            'huc12': ['010100000101', '010100000102'],
            'states': ['ME', 'PR'],
        })
        mask = qa_missing.is_conus_huc12(gdf, huc12_col='huc12')
        self.assertEqual(mask.tolist(), [True, False])

    def test_blank_states_field_does_not_leak_through(self):
        # Confirmed empirically (module docstring): HUC2 22 rows had a blank/NaN
        # 'states' field, so a states-only exclusion would have let them through.
        gdf = pd.DataFrame({
            'huc12': ['220100000101'],
            'states': [None],
        })
        mask = qa_missing.is_conus_huc12(gdf, huc12_col='huc12')
        self.assertEqual(mask.tolist(), [False])


class TestClassifyMissingHuc12s(unittest.TestCase):

    def setUp(self):
        # H1: fully covered by a divide (>= default 5% threshold) -> actionable.
        # H2: no divide anywhere near it -> no_divide_overlap.
        self.gdf_missing = gpd.GeoDataFrame({
            'huc12': ['H1', 'H2'],
            'geometry': [box(-82.0, 27.0, -81.9, 27.1), box(-70.0, 40.0, -69.9, 40.1)],
        }, crs='EPSG:4326')
        self.gdf_divides = gpd.GeoDataFrame({
            'divide_id': ['D1'],
            'geometry': [box(-82.0, 27.0, -81.9, 27.1)],  # exactly covers H1
        }, crs='EPSG:4326')

    def test_divide_covers_huc_is_actionable(self):
        result = qa_missing.classify_missing_huc12s(
            self.gdf_missing, self.gdf_divides, divide_id_col='divide_id', huc12_col='huc12')
        row_h1 = result[result['huc12'] == 'H1'].iloc[0]
        self.assertEqual(row_h1['classification'], 'divides_exist_not_crosswalked')
        self.assertEqual(row_h1['n_divides_overlapping'], 1)
        self.assertAlmostEqual(row_h1['total_coverage_frac'], 1.0, places=2)

    def test_huc_with_no_overlapping_divide_is_structural_gap(self):
        result = qa_missing.classify_missing_huc12s(
            self.gdf_missing, self.gdf_divides, divide_id_col='divide_id', huc12_col='huc12')
        row_h2 = result[result['huc12'] == 'H2'].iloc[0]
        self.assertEqual(row_h2['classification'], 'no_divide_overlap')
        self.assertEqual(row_h2['n_divides_overlapping'], 0)

    def test_partial_coverage_below_threshold_stays_no_divide_overlap(self):
        # A sliver divide covering far less than min_coverage_frac of H1's area.
        gdf_divides_sliver = gpd.GeoDataFrame({
            'divide_id': ['D_sliver'],
            'geometry': [box(-82.0, 27.0, -81.999, 27.1)],  # ~1% of H1's width
        }, crs='EPSG:4326')
        result = qa_missing.classify_missing_huc12s(
            self.gdf_missing, gdf_divides_sliver, divide_id_col='divide_id',
            min_coverage_frac=0.05, huc12_col='huc12')
        row_h1 = result[result['huc12'] == 'H1'].iloc[0]
        self.assertEqual(row_h1['classification'], 'no_divide_overlap')


class TestRunEndToEnd(unittest.TestCase):
    """Builds a real, minimal attr_config/pred_config chain plus a HUC12 GPKG and a
    divides GPKG, mirroring test_qa_utils.py's config fixture and test_regions.py's
    GPKG-building pattern. H1/H2 are crosswalked; H3 is deliberately left out so it
    surfaces as a 'missing' HUC12 dominated by an uncrosswalked divide.
    """

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_path = Path(self.temp_dir.name)

        self.dir_base = self.test_path / "base"
        self.dir_base.mkdir()
        self.dir_std_base = self.dir_base / "std"
        self.dir_std_base.mkdir()
        self.dir_db_attrs = self.dir_base / "db_attrs"
        self.dir_db_attrs.mkdir()

        attr_config = {
            'file_io': [
                {'home_dir': str(self.test_path)},
                {'dir_base': str(self.dir_base)},
                {'dir_std_base': str(self.dir_std_base)},
                {'dir_db_attrs': str(self.dir_db_attrs)},
            ],
            'formulation_metadata': [{'datasets': ['test_dataset']}],
            'attr_select': [{'static_vars': ['slope']}],
        }
        self.path_attr_config = self.test_path / "attr_config.yaml"
        with open(self.path_attr_config, 'w') as f:
            yaml.dump(attr_config, f)

        # HUC12 layer: H1/H2 crosswalked, H3 left out (the 'missing' one).
        gdf_huc12 = gpd.GeoDataFrame({
            'huc12': ['H1', 'H2', 'H3'],
            'states': ['FL', 'FL', 'FL'],
            'areasqkm': [100.0, 100.0, 100.0],
            'geometry': [
                box(-82.0, 27.0, -81.9, 27.1),
                box(-81.9, 27.0, -81.8, 27.1),
                box(-81.8, 27.0, -81.7, 27.1),
            ],
        }, crs='EPSG:4326')
        self.path_gpkg_pred = self.test_path / "pred_locs.gpkg"
        gdf_huc12.to_file(self.path_gpkg_pred, layer='huc12', driver='GPKG')

        # Divides layer: D_small sits entirely in H1 (crosswalked to H1); D_big spans
        # half of H3 and all of H2 (crosswalked to H2 only) -- so H3 is "missing" and
        # dominated by an uncrosswalked divide.
        gdf_divides = gpd.GeoDataFrame({
            'divide_id': ['D_small', 'D_big'],
            'geometry': [
                box(-81.98, 27.02, -81.92, 27.08),
                box(-81.95, 27.0, -81.75, 27.1),
            ],
        }, crs='EPSG:4326')
        self.path_hf_gpkg = self.test_path / "hydrofabric.gpkg"
        gdf_divides.to_file(self.path_hf_gpkg, layer='divides', driver='GPKG')

        self.path_crosswalk = self.test_path / "crosswalk.parquet"
        pd.DataFrame({
            'divide_id': ['D_small', 'D_big'],
            'huc12': ['H1', 'H2'],
        }).to_parquet(self.path_crosswalk, index=False)

        self.pred_config = {
            'name_attr_config': self.path_attr_config.name,
            'name_algo_config': 'algo.yaml',
            'ds_type': 'eval',
            'path_meta': 'path/meta/{ds}',
            'pred_file_comid_colname': 'huc12',
            'home_dir': str(self.test_path),
            'dir_std_base': str(self.dir_std_base),
            'path_crosswalk_ids': str(self.path_crosswalk),
            'path_gpkg_pred': str(self.path_gpkg_pred),
            'pred_gpkg_lyr': 'huc12',
            'pred_gpkg_id_col': 'huc12',
            'crosswalk_target_col': 'divide_id',
            'path_hf_finl_gpkg': str(self.path_hf_gpkg),
            'layr_hf_finl_gpkg': 'divides',
        }
        self.path_pred_config = self.test_path / "pred_config.yaml"
        with open(self.path_pred_config, 'w') as f:
            yaml.dump(self.pred_config, f)

        self.path_out_csv = self.test_path / "out.csv"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_h3_is_flagged_actionable_with_dominant_divide(self):
        n_actionable = qa_missing.run(
            self.path_pred_config, states=['FL'], min_coverage_frac=0.05,
            path_out_csv=self.path_out_csv, id_zfill_width=0)
        self.assertEqual(n_actionable, 1)
        result = pd.read_csv(self.path_out_csv)
        self.assertEqual(result.loc[result['huc12'] == 'H3', 'classification'].iloc[0],
                          'divides_exist_not_crosswalked')

    def test_no_crosswalk_set_skips_cleanly(self):
        del self.pred_config['path_crosswalk_ids']
        with open(self.path_pred_config, 'w') as f:
            yaml.dump(self.pred_config, f)
        n_actionable = qa_missing.run(self.path_pred_config, states=['FL'],
                                       path_out_csv=self.path_out_csv, id_zfill_width=0)
        self.assertEqual(n_actionable, 0)
        self.assertFalse(self.path_out_csv.exists())


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
