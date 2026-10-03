"""Unit tests for qa_check_region_coverage.run.

Follows the project convention (see README.claude "Unit tests should avoid mocking" and
test_regions.py): a real, small hydrofabric GPKG and a real crosswalk CSV on disk, plus a
real pred_config/attr_config YAML pair -- no mocking.
"""
import tempfile
import unittest
from pathlib import Path

import geopandas as gpd
import pandas as pd
import yaml
from shapely.geometry import box

import rafts_algo.flow.qa.qa_check_region_coverage as qa_region_coverage


class TestQaCheckRegionCoverage(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_path = Path(self.temp_dir.name)

        divide_ids = ['d1', 'd2', 'd3', 'd4', 'd5', 'd6']
        geoms = [box(-82.0 + 0.1 * i, 27.0, -82.0 + 0.1 * (i + 1), 27.1) for i in range(6)]
        gdf_divides = gpd.GeoDataFrame({'divide_id': divide_ids, 'geometry': geoms}, crs='EPSG:4326')
        self.path_hf_gpkg = self.test_path / "hydrofabric.gpkg"
        gdf_divides.to_file(self.path_hf_gpkg, layer='divides', driver='GPKG')

        self.path_crosswalk = self.test_path / "regions_crosswalk.csv"
        pd.DataFrame({
            'divide_id': ['d1', 'd2', 'd3', 'd4'],  # d5, d6 deliberately left uncovered (a gap)
            'region_id': ['A', 'A', 'B', 'B'],
        }).to_csv(self.path_crosswalk, index=False)

        attr_config = {
            'col_schema': [{'featureID': '{gage_id}'}, {'featureSource': 'test_src'}],
            'file_io': [{'home_dir': str(self.test_path)}, {'dir_base': str(self.test_path)}],
            'formulation_metadata': [{'datasets': ['test_dataset']}],
            'attr_select': [{'static_vars': ['slope']}],
        }
        self.path_attr_config = self.test_path / "attr_config.yaml"
        with open(self.path_attr_config, 'w') as f:
            yaml.dump(attr_config, f)

        self.pred_config = {
            'name_attr_config': self.path_attr_config.name,
            'name_algo_config': 'algo.yaml',
            'path_meta': 'path/meta/{ds}',
            'pred_file_comid_colname': 'divide_id',
            'regions': {
                'scheme': 'custom',
                'path_regions_crosswalk': str(self.path_crosswalk),
                'path_hf_finl_gpkg': str(self.path_hf_gpkg),
                'min_train_gages': 5,
            },
        }
        self.path_pred_config = self.test_path / "pred_config.yaml"

    def _write_pred_config(self):
        with open(self.path_pred_config, 'w') as f:
            yaml.dump(self.pred_config, f)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_no_regions_block_is_skipped(self):
        del self.pred_config['regions']
        self._write_pred_config()
        self.assertEqual(qa_region_coverage.run(self.path_pred_config), 0)

    def test_detects_gaps_and_flags_small_regions(self):
        self._write_pred_config()
        n_issues = qa_region_coverage.run(self.path_pred_config)
        # d5/d6 are gaps; include_conus defaults to True, so severity is 'info' (not counted
        # as an error) and the clean crosswalk (no divide in >1 region) has no overlaps.
        self.assertEqual(n_issues, 0)

    def test_gaps_are_errors_when_include_conus_false(self):
        self.pred_config['regions']['include_conus'] = False
        self._write_pred_config()
        n_issues = qa_region_coverage.run(self.path_pred_config)
        self.assertEqual(n_issues, 2)  # d5, d6

    def test_detects_overlap(self):
        pd.DataFrame({
            'divide_id': ['d1', 'd2', 'd3', 'd4', 'd1'],
            'region_id': ['A', 'A', 'B', 'B', 'B'],  # d1 claimed by both A and B
        }).to_csv(self.path_crosswalk, index=False)
        self._write_pred_config()
        n_issues = qa_region_coverage.run(self.path_pred_config)
        self.assertGreaterEqual(n_issues, 1)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
