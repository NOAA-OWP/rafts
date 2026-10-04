"""Unit tests for qa_check_huc12_divide_reassignment.py.

Follows the project convention (see README.claude "Unit tests should avoid mocking" and
test_qa_check_missing_huc12_divide_existence.py): a real, minimal attr_config/pred_config
chain plus a HUC12 GPKG and a divides GPKG, varying only the crosswalk's divide->huc12
assignment across tests to exercise the majority/minority/no-crosswalk-entry-at-all
classification branches (see the module docstring in
rafts_algo/flow/qa/qa_check_huc12_divide_reassignment.py for what each means).

Shared geometry across every test in this file:
    H1 = box(-82.0, 27.0, -81.9, 27.1)   # divide-sized receiver, always crosswalked
    H2 = box(-81.9, 27.0, -81.8, 27.1)
    H3 = box(-81.8, 27.0, -81.7, 27.1)
    D_small = box(-81.98, 27.02, -81.92, 27.08)  # entirely inside H1
    D_big   = box(-81.95, 27.0, -81.75, 27.1)    # spans H1 (25%), H2 (50%), H3 (25%)
                                                  # of D_big's OWN footprint, by construction
"""
import tempfile
import unittest
from pathlib import Path

import geopandas as gpd
import pandas as pd
import yaml
from shapely.geometry import box

import rafts_algo.flow.qa.qa_check_huc12_divide_reassignment as qa_reassign


class TestRunEndToEnd(unittest.TestCase):

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
        self.path_out_csv = self.test_path / "out.csv"

    def _write_crosswalk(self, divide_to_huc: dict):
        pd.DataFrame({
            'divide_id': list(divide_to_huc.keys()),
            'huc12': list(divide_to_huc.values()),
        }).to_parquet(self.path_crosswalk, index=False)
        with open(self.path_pred_config, 'w') as f:
            yaml.dump(self.pred_config, f)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_majority_share_assignment_is_not_flagged(self):
        # D_big crosswalked to H2 (holds 50% of D_big's footprint); H3 is "missing"
        # and only holds 25% of D_big's footprint -- the assigned HUC12 (H2) holds
        # the larger share, consistent with a majority-share assignment rule.
        self._write_crosswalk({'D_small': 'H1', 'D_big': 'H2'})
        n_true_gap = qa_reassign.run(self.path_pred_config, states=['FL'],
                                      path_out_csv=self.path_out_csv, top_n_plot=0, id_zfill_width=0)
        self.assertEqual(n_true_gap, 0)
        result = pd.read_csv(self.path_out_csv)
        row_h3 = result[result['huc12'] == 'H3'].iloc[0]
        self.assertEqual(row_h3['classification'], 'assigned_elsewhere_majority_share_of_divide')
        self.assertEqual(row_h3['assigned_huc12'], 'H2')

    def test_minority_share_assignment_is_flagged_as_true_gap(self):
        # D_big crosswalked to H3 instead (holds only 25% of D_big's footprint); H2 is
        # now "missing" and holds 50% of D_big's footprint -- a strictly larger share
        # than the HUC12 it was actually assigned to, the strongest evidence of a real
        # crosswalk-build defect.
        self._write_crosswalk({'D_small': 'H1', 'D_big': 'H3'})
        n_true_gap = qa_reassign.run(self.path_pred_config, states=['FL'],
                                      path_out_csv=self.path_out_csv, top_n_plot=0, id_zfill_width=0)
        self.assertEqual(n_true_gap, 1)
        result = pd.read_csv(self.path_out_csv)
        row_h2 = result[result['huc12'] == 'H2'].iloc[0]
        self.assertEqual(row_h2['classification'], 'assigned_elsewhere_minority_share_of_divide')
        self.assertEqual(row_h2['assigned_huc12'], 'H3')

    def test_divide_with_no_crosswalk_entry_at_all(self):
        # D_big is crosswalked nowhere -- a deeper gap than "assigned elsewhere."
        # H2 and H3 are both "missing" here; both rows resolve to the same
        # classification since neither divide has any assigned_huc to compare against.
        self._write_crosswalk({'D_small': 'H1'})
        n_true_gap = qa_reassign.run(self.path_pred_config, states=['FL'],
                                      path_out_csv=self.path_out_csv, top_n_plot=0, id_zfill_width=0)
        self.assertEqual(n_true_gap, 0)
        result = pd.read_csv(self.path_out_csv)
        self.assertTrue((result['classification'] == 'divide_itself_not_in_crosswalk').all())
        self.assertTrue(result['assigned_huc12'].isna().all())


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
