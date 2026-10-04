"""Unit tests for rafts_algo.regions.

Follows the project convention (see README.claude "Unit tests should avoid mocking" and
test_qa_utils.py): builds real, small GeoDataFrames, a real crosswalk CSV, and a real
hydrofabric GPKG divides layer on disk, rather than mocking geopandas/file I/O.
"""
import tempfile
import unittest
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, box

import rafts_algo.regions as raftsregions
from rafts_algo.schemas.pydantic_schemas import RegionsConfig


class RegionsTestBase(unittest.TestCase):
    """Shared fixture: two 3-divide crosswalk-defined regions (Florida-ish and Pacific
    NW-ish, placed at realistic CONUS longitude/latitude so EPSG:5070 buffering behaves
    sanely), written to a real crosswalk CSV and a real hydrofabric GPKG divides layer.
    """

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_path = Path(self.temp_dir.name)

        self.region_a_ids = ['d1', 'd2', 'd3']
        geoms_a = [box(-82.0 + 0.1 * i, 27.0, -82.0 + 0.1 * (i + 1), 27.1) for i in range(3)]
        self.region_b_ids = ['d4', 'd5', 'd6']
        geoms_b = [box(-120.0 + 0.1 * i, 45.0, -120.0 + 0.1 * (i + 1), 45.1) for i in range(3)]

        divide_ids = self.region_a_ids + self.region_b_ids
        geoms = geoms_a + geoms_b
        gdf_divides = gpd.GeoDataFrame({'divide_id': divide_ids, 'geometry': geoms}, crs='EPSG:4326')
        self.path_hf_gpkg = self.test_path / "hydrofabric.gpkg"
        gdf_divides.to_file(self.path_hf_gpkg, layer='divides', driver='GPKG')

        df_crosswalk = pd.DataFrame({
            'divide_id': divide_ids,
            'region_id': ['A'] * 3 + ['B'] * 3,
        })
        self.path_crosswalk = self.test_path / "regions_crosswalk.csv"
        df_crosswalk.to_csv(self.path_crosswalk, index=False)

        self.regions_cfg = RegionsConfig(
            scheme='custom',
            path_regions_crosswalk=str(self.path_crosswalk),
            path_hf_finl_gpkg=str(self.path_hf_gpkg),
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def _donors_with_geometry(self):
        """Donors at each divide's centroid, plus a point ~5km east of region A's edge
        ('nearby') and a point at region B's location ('far_away')."""
        return gpd.GeoDataFrame({
            'divide_id': ['d1', 'd2', 'd3', 'd4', 'd5', 'd6', 'nearby', 'far_away'],
            'geometry': [
                box(-82.0, 27.0, -81.9, 27.1).centroid,
                box(-81.9, 27.0, -81.8, 27.1).centroid,
                box(-81.8, 27.0, -81.7, 27.1).centroid,
                box(-120.0, 45.0, -119.9, 45.1).centroid,
                box(-119.9, 45.0, -119.8, 45.1).centroid,
                box(-119.8, 45.0, -119.7, 45.1).centroid,
                Point(-81.65, 27.05),
                Point(-120.0, 45.0),
            ],
        }, crs='EPSG:4326')


class TestLoadRegions(RegionsTestBase):

    def test_degenerate_case_returns_empty(self):
        self.assertEqual(raftsregions.load_regions(None), [])
        self.assertEqual(raftsregions.load_regions(RegionsConfig()), [])

    def test_custom_scheme_resolves_two_regions(self):
        regions = raftsregions.load_regions(self.regions_cfg)
        self.assertEqual({r.region_id for r in regions}, {'A', 'B'})
        region_a = next(r for r in regions if r.region_id == 'A')
        self.assertEqual(region_a.divide_ids, frozenset(self.region_a_ids))
        self.assertEqual(region_a.scheme, 'custom')
        # No buffer configured -> buffered_geom's area matches core_geom's (within
        # floating-point tolerance of a same-CRS reprojection round trip).
        self.assertAlmostEqual(region_a.core_geom.area, region_a.buffered_geom.area, delta=1.0)

    def test_accepts_raw_dict(self):
        raw = {
            'scheme': 'custom',
            'path_regions_crosswalk': str(self.path_crosswalk),
            'path_hf_finl_gpkg': str(self.path_hf_gpkg),
        }
        regions = raftsregions.load_regions(raw)
        self.assertEqual({r.region_id for r in regions}, {'A', 'B'})

    def test_ids_filter_restricts_to_subset(self):
        regions_cfg = self.regions_cfg.model_copy(update={'ids': ['A']})
        regions = raftsregions.load_regions(regions_cfg)
        self.assertEqual({r.region_id for r in regions}, {'A'})

    def test_crosswalk_divide_id_missing_from_divides_layer_warns_not_crashes(self):
        df_crosswalk = pd.read_csv(self.path_crosswalk)
        df_crosswalk = pd.concat(
            [df_crosswalk, pd.DataFrame({'divide_id': ['ghost'], 'region_id': ['A']})], ignore_index=True)
        df_crosswalk.to_csv(self.path_crosswalk, index=False)

        regions = raftsregions.load_regions(self.regions_cfg)  # must not raise
        region_a = next(r for r in regions if r.region_id == 'A')
        # 'ghost' is retained in divide_ids (the crosswalk's exact membership) even though it
        # has no match in the divides layer and so contributes nothing to the geometry.
        self.assertIn('ghost', region_a.divide_ids)


class TestBufferRegion(unittest.TestCase):

    def test_zero_buffer_is_noop_on_area(self):
        geom = box(-82.0, 27.0, -81.9, 27.1)
        buffered = raftsregions.buffer_region(geom, 0.0, src_crs='EPSG:4326')
        expected = gpd.GeoSeries([geom], crs='EPSG:4326').to_crs('EPSG:5070').iloc[0]
        self.assertAlmostEqual(buffered.area, expected.area, delta=0.001)

    def test_positive_buffer_expands_area(self):
        geom = box(-82.0, 27.0, -81.9, 27.1)
        unbuffered = raftsregions.buffer_region(geom, 0.0, src_crs='EPSG:4326')
        buffered = raftsregions.buffer_region(geom, 10.0, src_crs='EPSG:4326')
        self.assertGreater(buffered.area, unbuffered.area * 1.5)


class TestAssignToRegion(RegionsTestBase):

    def test_assigns_correct_region(self):
        regions = raftsregions.load_regions(self.regions_cfg)
        result = raftsregions.assign_to_region(['d1', 'd4', 'unknown'], regions)
        self.assertEqual(result['d1'], 'A')
        self.assertEqual(result['d4'], 'B')
        self.assertTrue(pd.isna(result['unknown']))

    def test_overlap_raises(self):
        shared = frozenset({'x'})
        region_a = raftsregions.RegionSpec(region_id='A', scheme='custom', divide_ids=shared,
                                            core_geom=box(0, 0, 1, 1), buffered_geom=box(0, 0, 1, 1))
        region_b = raftsregions.RegionSpec(region_id='B', scheme='custom', divide_ids=shared,
                                            core_geom=box(0, 0, 1, 1), buffered_geom=box(0, 0, 1, 1))
        with self.assertRaises(ValueError):
            raftsregions.assign_to_region(['x'], [region_a, region_b])


class TestDonorMask(RegionsTestBase):

    def test_core_only_when_no_buffer(self):
        regions = raftsregions.load_regions(self.regions_cfg)
        region_a = next(r for r in regions if r.region_id == 'A')
        df_donors = pd.DataFrame({'divide_id': ['d1', 'd2', 'd4']})
        result = raftsregions.donor_mask(df_donors, region_a, id_col='divide_id')
        self.assertEqual(set(result['divide_id']), {'d1', 'd2'})

    def test_buffer_zone_extension_includes_nearby_point_only(self):
        regions_cfg = self.regions_cfg.model_copy(update={'donor_buffer_km': 20.0})
        regions = raftsregions.load_regions(regions_cfg)
        region_a = next(r for r in regions if r.region_id == 'A')

        result = raftsregions.donor_mask(self._donors_with_geometry(), region_a, id_col='divide_id')
        ids = set(result['divide_id'])
        self.assertTrue({'d1', 'd2', 'd3'}.issubset(ids))
        self.assertIn('nearby', ids)
        self.assertNotIn('far_away', ids)
        self.assertNotIn('d4', ids)


class GageBasinTestBase(RegionsTestBase):
    """A gage whose basin spans only region A (g1), one spanning both A and B with a
    majority in A (g2), one entirely in B (g3), and one matching no region at all (g4) --
    the one-row-per-(gage_id, divide_id) shape `gdf_comid` has before its own
    `drop_duplicates(subset=['gage_id'])` collapses it.
    """

    def setUp(self):
        super().setUp()
        self.df_basin_rows = pd.DataFrame({
            'gage_id': ['g1', 'g1', 'g2', 'g2', 'g2', 'g3', 'g3', 'g3', 'g4', 'g4'],
            'divide_id': ['d1', 'd2', 'd2', 'd3', 'd4', 'd4', 'd5', 'd6', 'ghost1', 'ghost2'],
        })


class TestAssignGagesByBasinDivides(GageBasinTestBase):

    def test_majority_vote_resolves_basin_spanning_gage(self):
        regions = raftsregions.load_regions(self.regions_cfg)
        result = raftsregions.assign_gages_by_basin_divides(
            self.df_basin_rows, regions, gage_id_col='gage_id', divide_id_col='divide_id')
        self.assertEqual(result['g1'], 'A')
        self.assertEqual(result['g2'], 'A')  # 2 of 3 basin divides in A
        self.assertEqual(result['g3'], 'B')
        self.assertNotIn('g4', result.index)  # no basin divide matched any region

    def test_missing_divide_id_col_returns_empty(self):
        regions = raftsregions.load_regions(self.regions_cfg)
        result = raftsregions.assign_gages_by_basin_divides(
            pd.DataFrame({'gage_id': ['g1']}), regions, gage_id_col='gage_id', divide_id_col='divide_id')
        self.assertTrue(result.empty)


class TestDonorGageIdsForRegion(GageBasinTestBase):

    def test_inclusive_membership_allows_basin_spanning_gage_in_both_regions(self):
        regions = raftsregions.load_regions(self.regions_cfg)
        region_a = next(r for r in regions if r.region_id == 'A')
        region_b = next(r for r in regions if r.region_id == 'B')

        donors_a = raftsregions.donor_gage_ids_for_region(
            self.df_basin_rows, region_a, gage_id_col='gage_id', divide_id_col='divide_id')
        donors_b = raftsregions.donor_gage_ids_for_region(
            self.df_basin_rows, region_b, gage_id_col='gage_id', divide_id_col='divide_id')

        self.assertEqual(donors_a, {'g1', 'g2'})
        # g2 is eligible for BOTH -- donor eligibility is inclusive, not exclusive.
        self.assertEqual(donors_b, {'g2', 'g3'})

    def test_core_only_excludes_buffer_zone_but_includes_basin_spanning_gage(self):
        regions_cfg = self.regions_cfg.model_copy(update={'donor_buffer_km': 20.0})
        regions = raftsregions.load_regions(regions_cfg)
        region_a = next(r for r in regions if r.region_id == 'A')

        core_gages = raftsregions.donor_gage_ids_for_region_core(
            self.df_basin_rows, region_a, gage_id_col='gage_id', divide_id_col='divide_id')
        # g2 touches region A's core via d2/d3 even though its basin also reaches into B.
        self.assertEqual(core_gages, {'g1', 'g2'})

    def test_apply_min_train_gages_fallback_with_gage_keyed_eligibility_fn(self):
        regions = raftsregions.load_regions(self.regions_cfg)
        region_a = next(r for r in regions if r.region_id == 'A')
        df_donors_global = pd.DataFrame({'gage_id': ['g1', 'g2', 'g3']})

        eligible_fn = lambda r: raftsregions.donor_gage_ids_for_region(
            self.df_basin_rows, r, gage_id_col='gage_id', divide_id_col='divide_id')
        core_buffer = df_donors_global[df_donors_global['gage_id'].isin(eligible_fn(region_a))]
        self.assertEqual(set(core_buffer['gage_id']), {'g1', 'g2'})

        # min_train_gages=3 can't be met by region A alone (only g1/g2 ever match it, even
        # at the widest buffer) -- 'skip' with a gage-keyed eligibility_fn must behave
        # identically to the divide-keyed default (it never calls eligible_ids_fn at all).
        result, _, scope = raftsregions.apply_min_train_gages_fallback(
            region_a, core_buffer, df_donors_global, min_train_gages=3, fallback='skip',
            id_col='gage_id', eligible_ids_fn=eligible_fn)
        self.assertIsNone(result)
        self.assertEqual(scope, 'skip')


class TestApplyMinTrainGagesFallback(RegionsTestBase):

    def _region_a(self, donor_buffer_km=0.0):
        regions_cfg = self.regions_cfg.model_copy(update={'donor_buffer_km': donor_buffer_km})
        regions = raftsregions.load_regions(regions_cfg)
        return next(r for r in regions if r.region_id == 'A')

    def test_skip_returns_none(self):
        region_a = self._region_a()
        donors = self._donors_with_geometry()
        core_buffer = raftsregions.donor_mask(donors, region_a, id_col='divide_id')
        result, region_out, scope = raftsregions.apply_min_train_gages_fallback(
            region_a, core_buffer, donors, min_train_gages=5, fallback='skip', id_col='divide_id')
        self.assertIsNone(result)
        self.assertEqual(scope, 'skip')

    def test_parent_returns_global_set(self):
        region_a = self._region_a()
        donors = self._donors_with_geometry()
        core_buffer = raftsregions.donor_mask(donors, region_a, id_col='divide_id')
        result, region_out, scope = raftsregions.apply_min_train_gages_fallback(
            region_a, core_buffer, donors, min_train_gages=5, fallback='parent', id_col='divide_id')
        self.assertEqual(len(result), len(donors))
        self.assertEqual(scope, 'region_parent_fallback')

    def test_expand_buffer_succeeds_within_cap(self):
        region_a = self._region_a(donor_buffer_km=0.0)
        donors = self._donors_with_geometry()
        core_buffer = raftsregions.donor_mask(donors, region_a, id_col='divide_id')
        self.assertEqual(len(core_buffer), 3)  # only d1,d2,d3 at buffer=0

        result, region_out, scope = raftsregions.apply_min_train_gages_fallback(
            region_a, core_buffer, donors, min_train_gages=4, fallback='expand_buffer',
            id_col='divide_id', max_buffer_km=100.0, buffer_step_km=10.0)
        self.assertEqual(scope, 'region_expanded_buffer')
        self.assertGreaterEqual(len(result), 4)
        self.assertGreater(region_out.donor_buffer_km, 0.0)

    def test_expand_buffer_exhausts_to_parent(self):
        region_a = self._region_a(donor_buffer_km=0.0)
        donors = self._donors_with_geometry()
        core_buffer = raftsregions.donor_mask(donors, region_a, id_col='divide_id')

        # min_train_gages=8 can never be reached by buffering region A alone (region B's
        # divides are thousands of km away) -- must fall through to 'parent'.
        result, region_out, scope = raftsregions.apply_min_train_gages_fallback(
            region_a, core_buffer, donors, min_train_gages=8, fallback='expand_buffer',
            id_col='divide_id', max_buffer_km=50.0, buffer_step_km=10.0)
        self.assertEqual(scope, 'region_parent_fallback')
        self.assertEqual(len(result), len(donors))


class TestFindRegionOverlaps(RegionsTestBase):

    def test_no_overlap_in_clean_crosswalk(self):
        overlaps = raftsregions.find_region_overlaps(self.regions_cfg)
        self.assertTrue(overlaps.empty)

    def test_detects_duplicated_divide_id(self):
        df_crosswalk = pd.read_csv(self.path_crosswalk)
        df_crosswalk = pd.concat(
            [df_crosswalk, pd.DataFrame({'divide_id': ['d1'], 'region_id': ['B']})], ignore_index=True)
        df_crosswalk.to_csv(self.path_crosswalk, index=False)

        overlaps = raftsregions.find_region_overlaps(self.regions_cfg)
        self.assertEqual(len(overlaps), 1)
        self.assertEqual(overlaps.iloc[0]['divide_id'], 'd1')
        self.assertEqual(set(overlaps.iloc[0]['region_ids']), {'A', 'B'})

    def test_degenerate_case_returns_empty(self):
        self.assertTrue(raftsregions.find_region_overlaps(None).empty)


class TestFindRegionGaps(RegionsTestBase):

    def test_detects_uncovered_id(self):
        regions = raftsregions.load_regions(self.regions_cfg)
        gdf_candidates = pd.DataFrame({'divide_id': ['d1', 'd4', 'd99']})
        gaps = raftsregions.find_region_gaps(gdf_candidates, regions, id_col='divide_id')
        self.assertEqual(list(gaps['divide_id']), ['d99'])
        self.assertEqual(gaps.iloc[0]['severity'], 'info')

    def test_severity_is_error_when_include_conus_false(self):
        regions = raftsregions.load_regions(self.regions_cfg)
        gdf_candidates = pd.DataFrame({'divide_id': ['d99']})
        gaps = raftsregions.find_region_gaps(gdf_candidates, regions, id_col='divide_id', include_conus=False)
        self.assertEqual(gaps.iloc[0]['severity'], 'error')


class TestNormalizeNamedStateGroups(unittest.TestCase):

    def test_single_string(self):
        self.assertEqual(raftsregions.normalize_named_state_groups('fl'), {'FL': ['FL']})

    def test_flat_list(self):
        self.assertEqual(raftsregions.normalize_named_state_groups(['wa', 'or']), {'WA-OR': ['WA', 'OR']})

    def test_dict_form(self):
        result = raftsregions.normalize_named_state_groups({'PNW': ['wa', 'or'], 'FL': 'fl'})
        self.assertEqual(result, {'PNW': ['WA', 'OR'], 'FL': ['FL']})


class TestStatesMask(unittest.TestCase):

    def test_word_boundary_matching(self):
        gdf = pd.DataFrame({'states': ['FL,GA', 'WA', 'XFLX', 'OR']})
        mask = raftsregions.states_mask(gdf, 'states', ['FL'])
        self.assertEqual(list(mask), [True, False, False, False])


class TestResolveRegionLoop(RegionsTestBase):

    def test_degenerate_case(self):
        self.assertEqual(raftsregions.resolve_region_loop(None), [(None, None)])

    def test_loops_over_all_regions(self):
        loop = raftsregions.resolve_region_loop(self.regions_cfg)
        self.assertEqual({rid for rid, _ in loop}, {'A', 'B'})

    def test_cli_arg_restricts_to_one_region(self):
        loop = raftsregions.resolve_region_loop(self.regions_cfg, cli_region_arg='A')
        self.assertEqual(len(loop), 1)
        self.assertEqual(loop[0][0], 'A')

    def test_cli_arg_no_match_raises(self):
        with self.assertRaises(ValueError):
            raftsregions.resolve_region_loop(self.regions_cfg, cli_region_arg='nonexistent')


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
