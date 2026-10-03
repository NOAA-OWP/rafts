"""regions.py

Sub-regional partitioning for RaFTS training, clustering, pairing, and prediction.

CONUS is treated as the degenerate case of "one region with no buffer": when no `regions:`
config block is set (or its `scheme` is None), :func:`load_regions` returns ``[]`` and
:func:`resolve_region_loop` returns ``[(None, None)]`` -- every caller must treat that as "run
once, unregioned," byte-identical to behavior before this module existed, never as an error.

For `scheme` in ``{'vpu', 'huc2', 'custom'}``, region membership comes from a
divide_id -> region_id crosswalk file (``RegionsConfig.path_regions_crosswalk``), a file
dedicated to this purpose and never shared with ``PredConfig.path_crosswalk_ids`` (which maps
divide_id to an aggregation unit like huc12, for the unrelated purpose of attribute
aggregation -- see ``RegionsConfig``'s docstring in :mod:`rafts_algo.schemas.pydantic_schemas`).
Hydrofabric divide geometry for dissolving/buffering comes from
``RegionsConfig.path_hf_finl_gpkg``'s divides layer, read via
:func:`rafts_algo.qa_utils.resolve_divides_layer` (reused, not reimplemented).

For `scheme='states'`, region membership instead comes from a `states` text-column match
against a Census state-boundary basemap -- the same, simpler mechanism
``rafts_map_donor_receiver.py`` already used before this module existed (there, as
`donor_map_states`/`_normalize_donor_map_regions`), generalized here as
:func:`normalize_named_state_groups`/:func:`states_mask` so the three call sites that
duplicated it can share one implementation. A `states`-scheme :class:`RegionSpec` has an empty
`divide_ids` (no crosswalk applies to it) -- :func:`donor_mask` is designed for the
crosswalk-based schemes; callers filtering a `states`-columned layer should call
:func:`states_mask` directly, exactly as the pre-existing code already did.
"""
import dataclasses
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, FrozenSet, Iterable, List, Optional, Tuple, Union

import geopandas as gpd
import pandas as pd
from shapely.geometry.base import BaseGeometry

import rafts_algo.utils as raftsutil
import rafts_algo.plots as raftsplot
import rafts_algo.qa_utils as qa_utils
from rafts_algo.schemas.pydantic_schemas import RegionsConfig


@dataclass
class RegionSpec:
    """A single resolved sub-region: its exact crosswalk membership plus geometry for
    buffering/QA/display.

    :param region_id: The sub-region's identifier, as given in the crosswalk/config.
    :type region_id: str
    :param scheme: The region scheme this spec came from ('vpu'|'huc2'|'states'|'custom').
    :type scheme: str
    :param divide_ids: Exact divide_id membership (the region's "core"), from the crosswalk.
        Empty for scheme='states', which has no crosswalk (see module docstring).
    :type divide_ids: FrozenSet[str]
    :param core_geom: Dissolved geometry of `divide_ids` (or of the matched state polygons,
        for scheme='states'), in `crs`.
    :type core_geom: shapely.geometry.base.BaseGeometry
    :param buffered_geom: `core_geom` buffered outward by `donor_buffer_km`; identical to
        `core_geom` when `donor_buffer_km` is 0.
    :type buffered_geom: shapely.geometry.base.BaseGeometry
    :param crs: The CRS `core_geom`/`buffered_geom` are expressed in. Defaults to 'EPSG:5070'.
    :type crs: str
    :param donor_buffer_km: The buffer distance actually used to build `buffered_geom`.
    :type donor_buffer_km: float
    :param model_scope: How this region's training set was resolved: 'region' (normal),
        'region_expanded_buffer', or 'region_parent_fallback'. Set by
        :func:`apply_min_train_gages_fallback`, left at its default here.
    :type model_scope: str
    """
    region_id: str
    scheme: str
    divide_ids: FrozenSet[str]
    core_geom: BaseGeometry
    buffered_geom: BaseGeometry
    crs: str = 'EPSG:5070'
    donor_buffer_km: float = 0.0
    model_scope: str = 'region'


def _as_regions_config(regions_cfg: Union[RegionsConfig, dict, None]) -> Optional[RegionsConfig]:
    """Coerce a raw ``regions:`` dict (from an untyped ``*_cfg_dict`` parser) into a validated
    :class:`RegionsConfig`, so flow scripts not yet migrated to full Pydantic validation can
    still call :func:`load_regions` directly.

    :param regions_cfg: A validated RegionsConfig, a raw ``regions:`` dict, or None.
    :type regions_cfg: RegionsConfig | dict | None
    :return: The validated RegionsConfig, or None if `regions_cfg` was None.
    :rtype: Optional[RegionsConfig]
    """
    if regions_cfg is None:
        return None
    if isinstance(regions_cfg, RegionsConfig):
        return regions_cfg
    return RegionsConfig(**regions_cfg)


def buffer_region(core_geom: BaseGeometry, donor_buffer_km: float, *, src_crs: str,
                   buffer_crs: str = 'EPSG:5070') -> BaseGeometry:
    """Buffer a region's core geometry outward by `donor_buffer_km`, always in an
    equal-distance projected CRS -- never a geographic CRS, where "kilometers" would actually
    vary with latitude (see ``RegionsConfig.buffer_crs``'s docstring for why this matters).

    :param core_geom: The geometry to buffer, in `src_crs`.
    :type core_geom: shapely.geometry.base.BaseGeometry
    :param donor_buffer_km: Buffer distance in kilometers. 0 is a no-op (returns `core_geom`
        reprojected to `buffer_crs`, unbuffered).
    :type donor_buffer_km: float
    :param src_crs: The CRS `core_geom` is currently expressed in.
    :type src_crs: str
    :param buffer_crs: The equal-distance projected CRS to buffer in, defaults to 'EPSG:5070'.
    :type buffer_crs: str
    :return: The (possibly buffered) geometry, in `buffer_crs`.
    :rtype: shapely.geometry.base.BaseGeometry
    """
    geom_proj = gpd.GeoSeries([core_geom], crs=src_crs).to_crs(buffer_crs).iloc[0]
    if donor_buffer_km <= 0:
        return geom_proj
    return geom_proj.buffer(donor_buffer_km * 1000.0)


def _load_crosswalk_regions(regions_cfg: RegionsConfig, *, context: Optional[dict] = None
                             ) -> List[RegionSpec]:
    """Resolve `scheme` in ``{'vpu', 'huc2', 'custom'}``: a divide_id -> region_id crosswalk
    file plus the hydrofabric divides layer for geometry. The three schemes differ only in
    who/what produces the crosswalk file, not in mechanics.

    :param regions_cfg: The validated regions config (`scheme` in {'vpu','huc2','custom'}).
    :type regions_cfg: RegionsConfig
    :param context: f-string resolution context for `path_regions_crosswalk`/
        `path_hf_finl_gpkg` (e.g. ``{'home_dir': ...}``), defaults to None.
    :type context: Optional[dict]
    :return: One RegionSpec per distinct region_id in the crosswalk (restricted to
        `regions_cfg.ids` when given).
    :rtype: List[RegionSpec]
    :raises FileNotFoundError: If `path_regions_crosswalk` doesn't resolve to a real file.
    :raises ValueError: If the crosswalk file is missing `divide_id_col`/`region_id_col`.
    """
    context = context or {}
    path_crosswalk = Path(raftsutil.resolve_fstrings(regions_cfg.path_regions_crosswalk, context))
    if not path_crosswalk.exists():
        raise FileNotFoundError(f"regions.path_regions_crosswalk not found: {path_crosswalk}")

    divide_id_col, region_id_col = regions_cfg.divide_id_col, regions_cfg.region_id_col
    df_cw = (pd.read_csv(path_crosswalk) if path_crosswalk.suffix == '.csv'
             else pd.read_parquet(path_crosswalk, columns=[divide_id_col, region_id_col]))
    missing_cols = {divide_id_col, region_id_col} - set(df_cw.columns)
    if missing_cols:
        raise ValueError(f"path_regions_crosswalk {path_crosswalk} is missing column(s): {missing_cols}")
    df_cw[divide_id_col] = df_cw[divide_id_col].astype(str)
    df_cw[region_id_col] = df_cw[region_id_col].astype(str)

    if regions_cfg.ids:
        df_cw = df_cw[df_cw[region_id_col].isin(regions_cfg.ids)]

    # resolve_divides_layer only reads these two keys from its `pred_cfg_dict` argument, so a
    # small shim dict satisfies it without requiring regions_cfg to be nested inside a real
    # PredConfigParser dict.
    gdf_divides = qa_utils.resolve_divides_layer(
        {'path_hf_finl_gpkg': regions_cfg.path_hf_finl_gpkg, 'layr_hf_finl_gpkg': regions_cfg.layr_hf_finl_gpkg},
        context, divide_id_col)
    if gdf_divides.crs is None:
        logging.warning("Divides layer read for regions has no CRS set; assuming EPSG:4326.")
        gdf_divides = gdf_divides.set_crs(epsg=4326)

    regions = []
    for region_id, df_grp in df_cw.groupby(region_id_col):
        divide_ids = frozenset(df_grp[divide_id_col])
        gdf_member = gdf_divides[gdf_divides[divide_id_col].isin(divide_ids)]
        n_found = len(gdf_member)
        if n_found < len(divide_ids):
            logging.warning(
                f"Region '{region_id}': {len(divide_ids) - n_found} of {len(divide_ids)} crosswalk "
                f"divide_ids were not found in the divides layer (hydrofabric version drift?); "
                f"dissolving only the {n_found} found.")
        if gdf_member.empty:
            logging.warning(f"Region '{region_id}': none of its crosswalk divide_ids were found in "
                             f"the divides layer; skipping (no geometry to build).")
            continue
        core_geom = gdf_member.to_crs(regions_cfg.buffer_crs).union_all()
        buffered_geom = buffer_region(core_geom, regions_cfg.donor_buffer_km,
                                       src_crs=regions_cfg.buffer_crs, buffer_crs=regions_cfg.buffer_crs)
        regions.append(RegionSpec(
            region_id=region_id, scheme=regions_cfg.scheme, divide_ids=divide_ids,
            core_geom=core_geom, buffered_geom=buffered_geom, crs=regions_cfg.buffer_crs,
            donor_buffer_km=regions_cfg.donor_buffer_km))
    return regions


def _load_states_regions(regions_cfg: RegionsConfig, *, dir_out_viz_base: Optional[Union[str, Path]] = None
                          ) -> List[RegionSpec]:
    """Resolve `scheme='states'`: dissolve the Census state basemap's polygons per
    `regions_cfg.states` group. Unrelated mechanism to the crosswalk schemes -- see module
    docstring for why `divide_ids` is left empty here.

    :param regions_cfg: The validated regions config (`scheme='states'`).
    :type regions_cfg: RegionsConfig
    :param dir_out_viz_base: Directory :func:`rafts_algo.plots.gen_conus_basemap` caches its
        downloaded state-boundary shapefile into. Required (the basemap cache has no other
        default here); callers should pass the same path every time to avoid re-downloading.
    :type dir_out_viz_base: Optional[str | os.PathLike]
    :return: One RegionSpec per key of `regions_cfg.states` (restricted to `regions_cfg.ids`
        when given), with an empty `divide_ids`.
    :rtype: List[RegionSpec]
    :raises ValueError: If `dir_out_viz_base` is not given, or the basemap lacks a state
        abbreviation column.
    """
    if not dir_out_viz_base:
        raise ValueError("scheme='states' requires dir_out_viz_base (gen_conus_basemap's cache dir).")
    gdf_states = raftsplot.gen_conus_basemap(dir_out_viz_base)
    state_col = next((c for c in ('STUSPS', 'stusps') if c in gdf_states.columns), None)
    if state_col is None:
        raise ValueError(f"CONUS basemap has no state-abbreviation column (expected 'STUSPS'); "
                          f"found columns: {list(gdf_states.columns)}")

    regions = []
    for region_id, state_codes in regions_cfg.states.items():
        if regions_cfg.ids and region_id not in regions_cfg.ids:
            continue
        gdf_member = gdf_states[gdf_states[state_col].astype(str).str.upper().isin(
            [s.upper() for s in state_codes])]
        if gdf_member.empty:
            logging.warning(f"Region '{region_id}': no basemap states matched {state_codes}; skipping.")
            continue
        core_geom = gdf_member.to_crs(regions_cfg.buffer_crs).union_all()
        buffered_geom = buffer_region(core_geom, regions_cfg.donor_buffer_km,
                                       src_crs=regions_cfg.buffer_crs, buffer_crs=regions_cfg.buffer_crs)
        regions.append(RegionSpec(
            region_id=region_id, scheme='states', divide_ids=frozenset(), core_geom=core_geom,
            buffered_geom=buffered_geom, crs=regions_cfg.buffer_crs,
            donor_buffer_km=regions_cfg.donor_buffer_km))
    return regions


def load_regions(regions_cfg: Union[RegionsConfig, dict, None], *, context: Optional[dict] = None,
                  dir_out_viz_base: Optional[Union[str, Path]] = None) -> List[RegionSpec]:
    """Resolve a `regions:` config into concrete :class:`RegionSpec` objects.

    :param regions_cfg: A validated RegionsConfig, a raw ``regions:`` dict (coerced via
        :func:`_as_regions_config`), or None.
    :type regions_cfg: RegionsConfig | dict | None
    :param context: f-string resolution context for any templated path fields (e.g.
        ``{'home_dir': ...}``), defaults to None.
    :type context: Optional[dict]
    :param dir_out_viz_base: Required only when `scheme='states'`; see
        :func:`_load_states_regions`. Defaults to None.
    :type dir_out_viz_base: Optional[str | os.PathLike]
    :return: ``[]`` when `regions_cfg` is None or has no `scheme` -- callers must treat ``[]``
        as "run once, unregioned," never as an error. Otherwise one RegionSpec per resolved
        region.
    :rtype: List[RegionSpec]
    """
    regions_cfg = _as_regions_config(regions_cfg)
    if regions_cfg is None or regions_cfg.scheme is None:
        return []
    if regions_cfg.scheme == 'states':
        return _load_states_regions(regions_cfg, dir_out_viz_base=dir_out_viz_base)
    return _load_crosswalk_regions(regions_cfg, context=context)


def _divide_id_to_region_map(regions: List[RegionSpec]) -> dict:
    """Build a flat ``{divide_id: region_id}`` lookup from every region's `divide_ids`.

    :param regions: The resolved regions (only those with non-empty `divide_ids` -- i.e.
        crosswalk-based schemes -- contribute anything).
    :type regions: List[RegionSpec]
    :return: The flat lookup dict.
    :rtype: dict
    :raises ValueError: If any divide_id appears in more than one region's `divide_ids`.
    """
    id_to_region: dict = {}
    duplicates = set()
    for region in regions:
        for divide_id in region.divide_ids:
            prior = id_to_region.get(divide_id)
            if prior is not None and prior != region.region_id:
                duplicates.add(divide_id)
            id_to_region[divide_id] = region.region_id
    if duplicates:
        raise ValueError(f"{len(duplicates)} id(s) matched more than one region's divide_ids: "
                          f"{sorted(duplicates)[:10]}{'...' if len(duplicates) > 10 else ''}")
    return id_to_region


def assign_to_region(ids: Iterable[str], regions: List[RegionSpec]) -> pd.Series:
    """Map each id to the region_id of the region whose `divide_ids` it's a member of.

    Implemented as a dict lookup (:func:`_divide_id_to_region_map`) rather than per-region
    membership testing, so repeated ids in `ids` (e.g. the same divide appearing in more than
    one gage's basin in a nested-catchment hierarchy -- see
    :func:`assign_gages_by_basin_divides`) are handled correctly and efficiently, rather than
    relying on `.loc[]` selection against a non-unique index.

    :param ids: The ids to look up (e.g. a `col_locid` column's values). May contain repeats.
    :type ids: Iterable[str]
    :param regions: The resolved regions to match against (only those with non-empty
        `divide_ids` -- i.e. crosswalk-based schemes -- can match anything).
    :type regions: List[RegionSpec]
    :return: Series indexed by id -> region_id (NaN if the id is in no region's `divide_ids`).
    :rtype: pd.Series
    :raises ValueError: If any id appears in more than one region's `divide_ids`.
    """
    id_to_region = _divide_id_to_region_map(regions)
    ids = pd.Series(list(ids))
    result = ids.map(id_to_region)
    result.index = ids.values
    return result


def assign_gages_by_basin_divides(gdf_divide_rows: Union[gpd.GeoDataFrame, pd.DataFrame],
                                   regions: List[RegionSpec], *, gage_id_col: str,
                                   divide_id_col: str) -> pd.Series:
    """Resolve each gage's region membership from its full basin divide set, before any
    per-gage dedup collapses that basin down to a single representative divide.

    A gage's calibration basin generally corresponds to many divides (its upstream
    contributing area), not one -- `rafts_agg_hfatl_basin.py` builds exactly this
    gage_id<->divide_id basin mapping to area-aggregate attributes up to each gage, and its
    companion `{ds}_loc.gpkg` 'outlet' layer is the only place downstream of prep where that
    one-row-per-(gage,divide) structure survives. `rafts_proc_algo_pool.py`'s own
    `gdf_comid.drop_duplicates(subset=['gage_id'])` collapses it to one arbitrary divide per
    gage for an unrelated reason (picking one representative geometry) -- call this function
    on `gdf_comid` *before* that dedup runs, not after, or every gage will appear to have
    exactly one divide.

    A gage whose basin spans more than one region is assigned to whichever region contains
    the most of that basin's divides (ties broken by whichever region_id sorts first). This
    is a donor/training-location classification, not a receiver-coverage guarantee -- unlike
    a region's own `divide_ids` (which must be an exact, non-overlapping crosswalk lookup),
    an imprecise split for a basin spanning a region boundary is an acceptable approximation
    for "is this gage eligible as training data for region R."

    :param gdf_divide_rows: One row per (gage_id, divide_id) pair, e.g. `gdf_comid` as
        returned by `combine_resp_gdf_comid_wrap`, read *before* any
        `drop_duplicates(subset=[gage_id_col])`. Must have `gage_id_col` and `divide_id_col`.
    :type gdf_divide_rows: gpd.GeoDataFrame | pd.DataFrame
    :param regions: The resolved regions to match against.
    :type regions: List[RegionSpec]
    :param gage_id_col: Column holding the gage identifier.
    :type gage_id_col: str
    :param divide_id_col: Column holding each row's divide identifier.
    :type divide_id_col: str
    :return: Series indexed by gage_id -> region_id. Empty if `divide_id_col` isn't present
        (e.g. a legacy NLDI dataset with no divide-level basin mapping at all -- see module
        docstring) or no gage's basin matched any region.
    :rtype: pd.Series
    """
    if divide_id_col not in gdf_divide_rows.columns:
        return pd.Series(dtype=object)
    df = pd.DataFrame({
        gage_id_col: gdf_divide_rows[gage_id_col].astype(str).values,
        divide_id_col: gdf_divide_rows[divide_id_col].astype(str).values,
    })
    df['region_id'] = assign_to_region(df[divide_id_col], regions).values
    df = df.dropna(subset=['region_id'])
    if df.empty:
        return pd.Series(dtype=object)
    counts = df.groupby([gage_id_col, 'region_id']).size().rename('n').reset_index()
    counts = counts.sort_values(['n', 'region_id'], ascending=[False, True])
    winners = counts.drop_duplicates(subset=[gage_id_col], keep='first')
    return winners.set_index(gage_id_col)['region_id']


def donor_mask(gdf_or_ids: Union[gpd.GeoDataFrame, pd.DataFrame], region: RegionSpec, *,
               id_col: str, geom_col: str = 'geometry') -> Union[gpd.GeoDataFrame, pd.DataFrame]:
    """Select the rows of `gdf_or_ids` eligible as donors/training data for `region`: its
    crosswalk core members, plus (when `region.donor_buffer_km` > 0 and geometry is available)
    rows whose geometry falls within `region.buffered_geom`.

    Designed for crosswalk-based regions (`region.divide_ids` non-empty). For a
    `scheme='states'` region (`divide_ids` empty), this reduces to the spatial-only buffer
    test -- callers working with a `states`-columned layer should use :func:`states_mask`
    directly instead, matching the pre-existing mechanism that mask generalizes.

    :param gdf_or_ids: Candidate rows with an `id_col` column; a GeoDataFrame with `geom_col`
        if spatial buffer-zone extension is needed (i.e. `region.donor_buffer_km` > 0).
    :type gdf_or_ids: gpd.GeoDataFrame | pd.DataFrame
    :param region: The region whose core+buffer membership to test against.
    :type region: RegionSpec
    :param id_col: Column in `gdf_or_ids` holding the id to match against `region.divide_ids`.
    :type id_col: str
    :param geom_col: Geometry column name, used only for the buffer-zone extension, defaults
        to 'geometry'.
    :type geom_col: str, optional
    :return: The subset of `gdf_or_ids` eligible as donors/training data for `region`.
    :rtype: gpd.GeoDataFrame | pd.DataFrame
    """
    ids_str = gdf_or_ids[id_col].astype(str)
    core_mask = ids_str.isin(region.divide_ids)
    if region.donor_buffer_km <= 0 or not isinstance(gdf_or_ids, gpd.GeoDataFrame) or geom_col not in gdf_or_ids.columns:
        return gdf_or_ids[core_mask]

    gdf = gdf_or_ids
    if gdf.crs is None:
        logging.warning("donor_mask: candidate GeoDataFrame has no CRS set; assuming EPSG:4326.")
        gdf = gdf.set_crs(epsg=4326)
    gdf_proj = gdf.to_crs(region.crs)
    buffer_mask = gdf_proj.geometry.within(region.buffered_geom)
    return gdf_or_ids[core_mask | buffer_mask.values]


def donor_gage_ids_for_region(gdf_divide_rows: Union[gpd.GeoDataFrame, pd.DataFrame], region: RegionSpec, *,
                               gage_id_col: str, divide_id_col: str, geom_col: str = 'geometry') -> set:
    """Which gage_ids are eligible donors/training locations for `region`: gages with at
    least one basin divide in the region's core+buffer (:func:`donor_mask`, applied at the
    divide level, then rolled up to gage_id). A gage whose basin spans this region and
    another is eligible for both -- unlike a region's own `divide_ids` membership, donor
    eligibility is intentionally inclusive, not exclusive (more training data from a
    boundary-spanning basin isn't harmful; see :func:`assign_gages_by_basin_divides` for the
    single-winner alternative used for reporting/QA instead of eligibility).

    Call with `gdf_divide_rows` = `gdf_comid` read *before* its own
    `drop_duplicates(subset=[gage_id_col])` -- see :func:`assign_gages_by_basin_divides`'s
    docstring for why that order matters.

    :param gdf_divide_rows: One row per (gage_id, divide_id) pair. Must have `gage_id_col`
        and `divide_id_col`; needs `geom_col` only when `region.donor_buffer_km` > 0.
    :type gdf_divide_rows: gpd.GeoDataFrame | pd.DataFrame
    :param region: The region whose core+buffer eligibility to test.
    :type region: RegionSpec
    :param gage_id_col: Column holding the gage identifier.
    :type gage_id_col: str
    :param divide_id_col: Column holding each row's divide identifier.
    :type divide_id_col: str
    :param geom_col: Geometry column name, defaults to 'geometry'.
    :type geom_col: str, optional
    :return: The set of eligible gage_ids. Empty if `divide_id_col` isn't present (e.g. a
        legacy NLDI dataset with no divide-level basin mapping -- see module docstring).
    :rtype: set
    """
    if divide_id_col not in gdf_divide_rows.columns:
        return set()
    eligible_divides = donor_mask(gdf_divide_rows, region, id_col=divide_id_col, geom_col=geom_col)
    return set(eligible_divides[gage_id_col].astype(str))


def donor_gage_ids_for_region_core(gdf_divide_rows: Union[gpd.GeoDataFrame, pd.DataFrame], region: RegionSpec, *,
                                    gage_id_col: str, divide_id_col: str) -> set:
    """Gages whose basin has at least one divide in `region`'s CORE, ignoring any buffer --
    the set a held-out test split should be scoped to (per-region test scoring must use the
    core only, never the buffer, so no model is ever scored on a gage used to train another
    region). Contrast with :func:`donor_gage_ids_for_region`, which includes the buffer and
    is for training/donor eligibility, not test-set scoping.

    :param gdf_divide_rows: One row per (gage_id, divide_id) pair.
    :type gdf_divide_rows: gpd.GeoDataFrame | pd.DataFrame
    :param region: The region whose core-only eligibility to test.
    :type region: RegionSpec
    :param gage_id_col: Column holding the gage identifier.
    :type gage_id_col: str
    :param divide_id_col: Column holding each row's divide identifier.
    :type divide_id_col: str
    :return: The set of core-eligible gage_ids.
    :rtype: set
    """
    core_only_region = dataclasses.replace(region, buffered_geom=region.core_geom, donor_buffer_km=0.0)
    return donor_gage_ids_for_region(gdf_divide_rows, core_only_region, gage_id_col=gage_id_col,
                                      divide_id_col=divide_id_col)


def apply_min_train_gages_fallback(region: RegionSpec, gdf_donors_core_buffer, gdf_donors_global,
                                    min_train_gages: int, fallback: str, *, id_col: str,
                                    eligible_ids_fn: Optional[Callable[[RegionSpec], set]] = None,
                                    max_buffer_km: Optional[float] = None,
                                    buffer_step_km: float = 50.0
                                    ) -> Tuple[Optional[Union[gpd.GeoDataFrame, pd.DataFrame]], RegionSpec, str]:
    """Apply the configured small-region `fallback` policy when a region's core+buffer donor
    set falls below `min_train_gages`.

    :param region: The region whose donor set was too small.
    :type region: RegionSpec
    :param gdf_donors_core_buffer: The region's (too-small) core+buffer donor set.
    :type gdf_donors_core_buffer: gpd.GeoDataFrame | pd.DataFrame
    :param gdf_donors_global: The full, unfiltered CONUS-wide donor set, used by the 'parent'
        policy (and as the retry pool for 'expand_buffer').
    :type gdf_donors_global: gpd.GeoDataFrame | pd.DataFrame
    :param min_train_gages: The configured minimum donor/training-gage count.
    :type min_train_gages: int
    :param fallback: One of 'parent', 'expand_buffer', 'skip' (RegionsConfig.fallback).
    :type fallback: str
    :param id_col: Column identifying each donor row in `gdf_donors_global`.
    :type id_col: str
    :param eligible_ids_fn: Callable(region) -> set of eligible ids at that region's current
        `buffered_geom`, used only by the 'expand_buffer' policy to re-test eligibility after
        widening the buffer. Defaults to None, which tests `id_col` directly against
        :func:`donor_mask` -- correct when `gdf_donors_global` is divide-keyed (e.g. a
        receiver/prediction use). Pass a closure over
        :func:`donor_gage_ids_for_region` instead when `gdf_donors_global` is gage-keyed
        (e.g. training/donor use, where a gage's basin can span many divides -- see that
        function's docstring for why a direct `donor_mask` test would be wrong there).
    :type eligible_ids_fn: Optional[Callable[[RegionSpec], set]]
    :param max_buffer_km: Cap on how far 'expand_buffer' may widen the buffer before falling
        through to the 'parent' policy, defaults to None (no cap -- widens until it succeeds
        or `gdf_donors_global` itself is exhausted).
    :type max_buffer_km: Optional[float]
    :param buffer_step_km: How much to widen the buffer by on each 'expand_buffer' retry,
        defaults to 50.0.
    :type buffer_step_km: float
    :return: A 3-tuple: (the resolved donor set, or None for 'skip'; the region, possibly with
        a widened `buffered_geom`/`donor_buffer_km`; the `model_scope` tag to record).
    :rtype: tuple
    """
    if fallback == 'skip':
        return None, region, 'skip'
    if fallback == 'parent':
        return gdf_donors_global, region, 'region_parent_fallback'

    # fallback == 'expand_buffer'
    if eligible_ids_fn is None:
        eligible_ids_fn = lambda r: set(donor_mask(gdf_donors_global, r, id_col=id_col)[id_col].astype(str))

    new_buffer_km = region.donor_buffer_km
    while max_buffer_km is None or new_buffer_km < max_buffer_km:
        new_buffer_km += buffer_step_km
        if max_buffer_km is not None:
            new_buffer_km = min(new_buffer_km, max_buffer_km)
        widened_geom = buffer_region(region.core_geom, new_buffer_km, src_crs=region.crs, buffer_crs=region.crs)
        widened_region = RegionSpec(region_id=region.region_id, scheme=region.scheme,
                                     divide_ids=region.divide_ids, core_geom=region.core_geom,
                                     buffered_geom=widened_geom, crs=region.crs,
                                     donor_buffer_km=new_buffer_km, model_scope='region_expanded_buffer')
        eligible_ids = eligible_ids_fn(widened_region)
        gdf_widened = gdf_donors_global[gdf_donors_global[id_col].astype(str).isin(eligible_ids)]
        if len(gdf_widened) >= min_train_gages:
            return gdf_widened, widened_region, 'region_expanded_buffer'
        if max_buffer_km is not None and new_buffer_km >= max_buffer_km:
            break

    logging.warning(f"Region '{region.region_id}': 'expand_buffer' exhausted without reaching "
                     f"{min_train_gages} training gages; falling through to 'parent'.")
    return gdf_donors_global, region, 'region_parent_fallback'


def find_region_overlaps(regions_cfg: Union[RegionsConfig, dict, None], *, context: Optional[dict] = None
                          ) -> pd.DataFrame:
    """Find every divide_id assigned to more than one region_id in the crosswalk -- a pure
    tabular check, no geometry needed, since crosswalk-based region membership is an exact id
    assignment, not a spatial test.

    :param regions_cfg: A validated RegionsConfig, a raw ``regions:`` dict, or None.
    :type regions_cfg: RegionsConfig | dict | None
    :param context: f-string resolution context for `path_regions_crosswalk`, defaults to None.
    :type context: Optional[dict]
    :return: One row per (divide_id, [region_ids]) with more than one assigned region_id.
        Empty if `regions_cfg` has no crosswalk scheme, or none overlap.
    :rtype: pd.DataFrame
    """
    regions_cfg = _as_regions_config(regions_cfg)
    if regions_cfg is None or regions_cfg.scheme not in ('vpu', 'huc2', 'custom'):
        return pd.DataFrame(columns=['divide_id', 'region_ids'])

    context = context or {}
    path_crosswalk = Path(raftsutil.resolve_fstrings(regions_cfg.path_regions_crosswalk, context))
    divide_id_col, region_id_col = regions_cfg.divide_id_col, regions_cfg.region_id_col
    df_cw = (pd.read_csv(path_crosswalk) if path_crosswalk.suffix == '.csv'
             else pd.read_parquet(path_crosswalk, columns=[divide_id_col, region_id_col]))
    df_cw[divide_id_col] = df_cw[divide_id_col].astype(str)

    grouped = df_cw.groupby(divide_id_col)[region_id_col].agg(lambda s: sorted(set(s.astype(str))))
    overlaps = grouped[grouped.map(len) > 1]
    return pd.DataFrame({'divide_id': overlaps.index, 'region_ids': overlaps.values}).reset_index(drop=True)


def find_region_gaps(gdf_candidates: Union[gpd.GeoDataFrame, pd.DataFrame], regions: List[RegionSpec],
                      id_col: str, *, include_conus: bool = True) -> pd.DataFrame:
    """Find rows of `gdf_candidates` matched to no region's core.

    :param gdf_candidates: Candidate rows with an `id_col` column (e.g. all receivers/divides
        a workflow expects to be covered by some region).
    :type gdf_candidates: gpd.GeoDataFrame | pd.DataFrame
    :param regions: The resolved regions to match against.
    :type regions: List[RegionSpec]
    :param id_col: Column in `gdf_candidates` holding the id to match.
    :type id_col: str
    :param include_conus: Whether unmatched rows are expected (an implicit CONUS catch-all),
        reflected only in the returned 'severity' column -- this function always returns the
        gap rows themselves; it's the caller's job to decide whether 'info'-severity gaps are
        worth failing a build over. Defaults to True.
    :type include_conus: bool
    :return: One row per unmatched id, with a 'severity' column ('info' if `include_conus`
        else 'error'). Empty if every id matched a region.
    :rtype: pd.DataFrame
    """
    assigned = assign_to_region(gdf_candidates[id_col].astype(str), regions)
    gaps = assigned[assigned.isna()]
    severity = 'info' if include_conus else 'error'
    return pd.DataFrame({id_col: gaps.index, 'severity': severity})


def normalize_named_state_groups(states_cfg) -> dict:
    """Normalize a `donor_map_states`-shaped config value into ``{region_name: [state, ...]}``.

    Relocated verbatim from ``rafts_map_donor_receiver._normalize_donor_map_regions`` --
    identical behavior, generalized location so the three call sites that need it (that
    script, and the two QA scripts with their own `--states` filter) share one implementation
    instead of each keeping their own copy.

    :param states_cfg: Raw value: ``None`` (caller should apply its own default), a falsy
        value (explicit opt-out), a single state string, a flat list of state codes (one
        region, named by joining the codes), or a dict of ``{region_name: state_or_states}``.
    :type states_cfg: None | str | list | dict
    :return: ``{region_name: [2-letter state code, ...]}``, empty if opted out.
    :rtype: dict
    """
    if isinstance(states_cfg, dict):
        regions = {}
        for name, states in states_cfg.items():
            states_list = [states] if isinstance(states, str) else list(states)
            regions[str(name)] = [str(s).strip().upper() for s in states_list]
        return regions

    states_list = [states_cfg] if isinstance(states_cfg, str) else list(states_cfg)
    states_norm = [str(s).strip().upper() for s in states_list]
    return {"-".join(states_norm): states_norm}


def states_mask(gdf: Union[gpd.GeoDataFrame, pd.DataFrame], states_col: str,
                 region_states: List[str]) -> pd.Series:
    """Boolean mask of rows whose `states_col` value contains any of `region_states`, matched
    as whole state codes (word-boundary, not substring).

    Generalizes the identical regex block duplicated at (pre-generalization)
    ``rafts_map_donor_receiver.py``, and the two QA scripts' own `--states` filters -- all
    three should import this instead of keeping their own copy.

    :param gdf: Rows with a `states_col` column (e.g. a multi-state string like 'FL' or
        'FL,GA').
    :type gdf: gpd.GeoDataFrame | pd.DataFrame
    :param states_col: Column holding the state-code string to match against.
    :type states_col: str
    :param region_states: The 2-letter state codes defining this region.
    :type region_states: List[str]
    :return: Boolean mask, same index as `gdf`.
    :rtype: pd.Series
    """
    state_pattern = '|'.join(rf'\b{re.escape(s)}\b' for s in region_states)
    return gdf[states_col].astype(str).str.contains(state_pattern, regex=True, na=False)


def resolve_region_loop(regions_cfg: Union[RegionsConfig, dict, None], cli_region_arg: Optional[str] = None,
                         **load_regions_kwargs) -> List[Tuple[Optional[str], Optional[RegionSpec]]]:
    """Resolve the (region_id, RegionSpec) pairs a flow script's outermost loop should iterate
    over -- the one shared loop helper every region-aware flow script calls, instead of each
    duplicating this dispatch.

    :param regions_cfg: A validated RegionsConfig, a raw ``regions:`` dict, or None.
    :type regions_cfg: RegionsConfig | dict | None
    :param cli_region_arg: An optional ``--region`` CLI value restricting the loop to just
        that one region_id, defaults to None (loop over every resolved region).
    :type cli_region_arg: Optional[str]
    :param load_regions_kwargs: Forwarded to :func:`load_regions` (e.g. `context`,
        `dir_out_viz_base`).
    :return: ``[(None, None)]`` if `regions_cfg` has no scheme (run once, unregioned).
        Otherwise one ``(region_id, RegionSpec)`` per resolved region, or just the one
        matching `cli_region_arg` if given.
    :rtype: List[Tuple[Optional[str], Optional[RegionSpec]]]
    :raises ValueError: If `cli_region_arg` matches no resolved region_id.
    """
    regions = load_regions(regions_cfg, **load_regions_kwargs)
    if not regions:
        return [(None, None)]
    if cli_region_arg is not None:
        matched = [r for r in regions if r.region_id == cli_region_arg]
        if not matched:
            raise ValueError(f"--region '{cli_region_arg}' matched no resolved region_id "
                              f"(available: {sorted(r.region_id for r in regions)}).")
        return [(matched[0].region_id, matched[0])]
    return [(r.region_id, r) for r in regions]
