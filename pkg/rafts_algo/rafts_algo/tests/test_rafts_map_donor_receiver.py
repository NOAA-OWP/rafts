"""Unit tests for rafts_map_donor_receiver.build_divide_proxy_receivers.

rafts_map_donor_receiver.py has no run()/main() function to call end-to-end (its body is
all inline under __main__), so this targets its one pure, independently testable function
-- real, small GeoDataFrames, no mocking (see README.claude "Unit tests should avoid
mocking").
"""
import unittest

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from rafts_algo.flow.rafts_map_donor_receiver import build_divide_proxy_receivers


class TestBuildDivideProxyReceivers(unittest.TestCase):

    def setUp(self):
        # H_gap has no donor-pairs entry (the "no pairing" gap); a divide bigger than
        # H_gap covers most of it but was actually crosswalked to H_assigned instead.
        self.gdf_no_pairing = gpd.GeoDataFrame({
            'huc12': ['H_gap'],
            'areasqkm': [50.0],
            'geometry': [box(-82.0, 27.0, -81.9, 27.1)],
        }, crs='EPSG:4326')
        self.gdf_divides = gpd.GeoDataFrame({
            'divide_id': ['D_dominant'],
            'geometry': [box(-82.05, 26.95, -81.85, 27.15)],  # bigger than H_gap, covers it fully
        }, crs='EPSG:4326')
        self.divide_to_assigned_huc = pd.Series({'D_dominant': 'H_assigned'})
        self.divide_area_sqkm = pd.Series({'D_dominant': 200.0})  # > H_gap's areasqkm
        self.dp = pd.DataFrame({
            'receiver_id': ['H_assigned'],
            'cluster_id': [2],
            'donor_id': ['01234567'],
        })

    def test_resolves_gap_into_divide_shaped_proxy_with_borrowed_pairing(self):
        gdf_proxy, gdf_remaining = build_divide_proxy_receivers(
            self.gdf_no_pairing, self.gdf_divides, self.divide_to_assigned_huc,
            self.divide_area_sqkm, self.dp, pred_gpkg_id_col='huc12', divide_id_col='divide_id')

        self.assertEqual(len(gdf_proxy), 1)
        self.assertEqual(gdf_proxy.iloc[0]['huc12'], 'H_gap')
        self.assertEqual(gdf_proxy.iloc[0]['divide_id'], 'D_dominant')
        self.assertEqual(gdf_proxy.iloc[0]['cluster_id'], 2)
        self.assertEqual(gdf_proxy.iloc[0]['donor_id'], '01234567')
        self.assertTrue(gdf_proxy.iloc[0]['is_divide_proxy'])
        self.assertTrue(gdf_remaining.empty)

    def test_no_proxy_when_divide_is_smaller_than_the_gap_huc(self):
        # The dominant divide isn't actually bigger than H_gap -- no substitution,
        # since the physical-correctness justification for a proxy doesn't hold.
        small_divide_area = pd.Series({'D_dominant': 10.0})
        gdf_proxy, gdf_remaining = build_divide_proxy_receivers(
            self.gdf_no_pairing, self.gdf_divides, self.divide_to_assigned_huc,
            small_divide_area, self.dp, pred_gpkg_id_col='huc12', divide_id_col='divide_id')

        self.assertTrue(gdf_proxy.empty)
        self.assertEqual(len(gdf_remaining), 1)

    def test_no_proxy_when_assigned_huc_not_in_donor_pairs(self):
        # The divide's assigned HUC12 has no entry in dp (e.g. that algo/resp_var
        # combination never paired it) -- nothing to borrow, falls back to the gap.
        dp_empty = pd.DataFrame({'receiver_id': [], 'cluster_id': [], 'donor_id': []})
        gdf_proxy, gdf_remaining = build_divide_proxy_receivers(
            self.gdf_no_pairing, self.gdf_divides, self.divide_to_assigned_huc,
            self.divide_area_sqkm, dp_empty, pred_gpkg_id_col='huc12', divide_id_col='divide_id')

        self.assertTrue(gdf_proxy.empty)
        self.assertEqual(len(gdf_remaining), 1)

    def test_no_overlap_returns_empty_proxy(self):
        gdf_divides_far = gpd.GeoDataFrame({
            'divide_id': ['D_far'],
            'geometry': [box(10.0, 10.0, 10.1, 10.1)],  # nowhere near H_gap
        }, crs='EPSG:4326')
        gdf_proxy, gdf_remaining = build_divide_proxy_receivers(
            self.gdf_no_pairing, gdf_divides_far, self.divide_to_assigned_huc,
            self.divide_area_sqkm, self.dp, pred_gpkg_id_col='huc12', divide_id_col='divide_id')

        self.assertTrue(gdf_proxy.empty)
        self.assertEqual(len(gdf_remaining), 1)

    def test_empty_gdf_no_pairing_short_circuits(self):
        gdf_empty = self.gdf_no_pairing.iloc[0:0]
        gdf_proxy, gdf_remaining = build_divide_proxy_receivers(
            gdf_empty, self.gdf_divides, self.divide_to_assigned_huc,
            self.divide_area_sqkm, self.dp, pred_gpkg_id_col='huc12', divide_id_col='divide_id')

        self.assertTrue(gdf_proxy.empty)
        self.assertTrue(gdf_remaining.empty)

    def test_missing_areasqkm_column_short_circuits(self):
        gdf_no_area = self.gdf_no_pairing.drop(columns=['areasqkm'])
        gdf_proxy, gdf_remaining = build_divide_proxy_receivers(
            gdf_no_area, self.gdf_divides, self.divide_to_assigned_huc,
            self.divide_area_sqkm, self.dp, pred_gpkg_id_col='huc12', divide_id_col='divide_id')

        self.assertTrue(gdf_proxy.empty)
        self.assertTrue(gdf_remaining.equals(gdf_no_area))


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
