from typing import Any, Dict, Optional, Tuple, List, Union, Literal
from pydantic import BaseModel, Field, field_validator, model_validator
import numpy as np
import re

# %% Region Configuration Validation

class RegionsConfig(BaseModel):
    """Validates the optional `regions:` config block. Absent/None (the default on both
    AlgoConfig.regions and PredConfig.regions) reproduces today's CONUS-wide, unregioned
    behavior byte-for-byte -- every downstream consumer treats `scheme is None` as "run
    exactly once, unregioned," never as an error.

    :param scheme: Region-definition source. 'vpu'/'huc2'/'custom' all resolve via a
        divide_id -> region_id crosswalk file plus the hydrofabric divides layer for
        geometry (they differ only in who/what produces the crosswalk, not in mechanics).
        'states' resolves a `states` column match, independent of the crosswalk mechanism
        (see rafts_algo.regions.states_mask). None means no regions configured (degenerate
        CONUS case). Defaults to None.
    :type scheme: Optional[Literal['vpu', 'huc2', 'states', 'custom']]
    :param ids: Subset of region identifiers to actually run, keyed against `region_id_col`
        (vpu/huc2/custom) or the keys of `states` (states). None means every region the
        source defines. Defaults to None.
    :type ids: Optional[List[str]]
    :param path_regions_crosswalk: Path (may use the same f-string placeholders as other
        *_config path fields, e.g. '{home_dir}') to a CSV/Parquet crosswalk with at least
        [divide_id_col, region_id_col] columns, one row per divide assigned to a sub-region.
        This is a dedicated file for region partitioning ONLY -- deliberately separate from
        PredConfig.path_crosswalk_ids, which maps divide_id to an aggregation unit (e.g.
        huc12) for a wholly different purpose (attribute aggregation). Never read
        divide_id<->region_id pairs out of path_crosswalk_ids, and never add a region_id
        column to that file -- a divide's aggregation huc12 and its training sub-region are
        independent facts that must be editable independently. Required when scheme is
        'vpu', 'huc2', or 'custom'. Existence is checked in rafts_algo.regions.load_regions()
        after f-string resolution, not here -- same idiom as PredConfig.path_hf_finl_gpkg.
        Defaults to None.
    :type path_regions_crosswalk: Optional[str]
    :param divide_id_col: Column name holding the hydrofabric divide identifier, in both
        `path_regions_crosswalk` and the `path_hf_finl_gpkg` divides layer. Defaults to
        'divide_id' (matches PredConfig.map_divide_id_col's default).
    :type divide_id_col: str
    :param region_id_col: Column in `path_regions_crosswalk` holding the sub-region
        identifier. Defaults to 'region_id'.
    :type region_id_col: str
    :param path_hf_finl_gpkg: Path to the hydrofabric GPKG providing divide geometry for the
        crosswalk's entries (dissolved into each region's core polygon, then buffered).
        Required whenever scheme is 'vpu', 'huc2', or 'custom'. Same field name/convention as
        PredConfig.path_hf_finl_gpkg. Defaults to None.
    :type path_hf_finl_gpkg: Optional[str]
    :param layr_hf_finl_gpkg: Layer to read from `path_hf_finl_gpkg`. Defaults to 'divides'.
    :type layr_hf_finl_gpkg: str
    :param states: For scheme='states' only: {region_id: [2-letter state codes]}, same shape
        as PredConfig.donor_map_states' dict form. Required when scheme='states'. Defaults
        to None.
    :type states: Optional[Dict[str, List[str]]]
    :param donor_buffer_km: Kilometers to buffer each region's core geometry outward when
        selecting eligible donors/training data beyond the crosswalk's explicit membership;
        0 means donors are exactly the crosswalk's core set, no spatial extension. Buffering
        always happens in `buffer_crs`, never a geographic CRS. Defaults to 0.0.
    :type donor_buffer_km: float
    :param buffer_crs: Equal-distance projected CRS to buffer in. Defaults to 'EPSG:5070'
        (CONUS Albers Equal Area) -- deliberately not EPSG:3857, whose distance distortion
        grows with latitude.
    :type buffer_crs: str
    :param min_train_gages: Minimum donor/training-gage count a region's buffered set must
        reach before training proceeds normally; below this, `fallback` applies. Defaults to
        1 (effectively no gate) -- set explicitly for real small-region protection.
    :type min_train_gages: int
    :param fallback: Policy when a region falls below `min_train_gages`: 'parent' trains
        that region on the full global/CONUS donor pool instead (predictions stay scoped to
        the region's own core); 'expand_buffer' widens `donor_buffer_km` and retries, falling
        through to 'parent' if still short; 'skip' trains nothing for that region (its core
        receivers surface as gaps in the region-coverage QA check). Defaults to 'skip'.
    :type fallback: Literal['parent', 'expand_buffer', 'skip']
    :param include_conus: Whether locations outside every configured region's core are
        expected (an implicit catch-all) rather than a QA error. Defaults to True.
    :type include_conus: bool
    :param max_spa_dist_km: Optional maximum spatial distance (km) between a receiver and
        its assigned donor in rafts_algo_train.assign_donors_to_receivers' attribute-space
        pairing. None (default) preserves today's behavior exactly (no spatial term at all).
    :type max_spa_dist_km: Optional[float]
    """
    scheme: Optional[Literal['vpu', 'huc2', 'states', 'custom']] = None
    ids: Optional[List[str]] = None
    path_regions_crosswalk: Optional[str] = None
    divide_id_col: str = 'divide_id'
    region_id_col: str = 'region_id'
    path_hf_finl_gpkg: Optional[str] = None
    layr_hf_finl_gpkg: str = 'divides'
    states: Optional[Dict[str, List[str]]] = None
    donor_buffer_km: float = 0.0
    buffer_crs: str = 'EPSG:5070'
    min_train_gages: int = 1
    fallback: Literal['parent', 'expand_buffer', 'skip'] = 'skip'
    include_conus: bool = True
    max_spa_dist_km: Optional[float] = None

    @model_validator(mode="after")
    def validate_scheme_requirements(self) -> "RegionsConfig":
        """Cross-field validation mirroring the scheme-specific requirements documented above.

        :return: The validated model.
        :rtype: RegionsConfig
        """
        if self.scheme in ('vpu', 'huc2', 'custom'):
            if not self.path_regions_crosswalk or not self.path_hf_finl_gpkg:
                raise ValueError(
                    f"scheme='{self.scheme}' requires both path_regions_crosswalk and "
                    f"path_hf_finl_gpkg.")
        if self.scheme == 'states' and not self.states:
            raise ValueError("scheme='states' requires a non-empty 'states' mapping.")
        if self.donor_buffer_km < 0:
            raise ValueError("donor_buffer_km must be >= 0.")
        if self.min_train_gages < 1:
            raise ValueError("min_train_gages must be >= 1.")
        return self

# %% ML Configuration Validation

class AlgoConfig(BaseModel):
    """Validates the algorithm training configuration YAML (algo_config).

    :param task_type: Either 'regression' or 'clustering'; selects which
        algorithms/evaluation columns are applicable. Defaults to 'regression'.
    :type task_type: str
    :param algorithms: The algorithms to train, keyed by name (e.g. 'rf',
        'mlp', 'kmeans'), with each value being that algorithm's own
        parameter dict as read from the YAML.
    :type algorithms: Dict[str, Any]
    :param test_size: Proportion of the dataset held out for testing, passed
        to :func:`sklearn.model_selection.train_test_split`. Defaults to 0.3.
    :type test_size: float
    :param seed: Random seed for reproducibility. Defaults to 32.
    :type seed: int
    :param name_attr_config: Name of the linked attribute config file,
        expected in the same directory as this algo config.
    :type name_attr_config: str
    :param name_attr_csv: Optional name of a .csv file defining the
        attributes to use for training, in lieu of the attribute config's
        own selection. Defaults to None.
    :type name_attr_csv: Optional[str]
    :param colname_attr_csv: Column name inside `name_attr_csv` holding the
        attribute names; required if `name_attr_csv` is set. Defaults to None.
    :type colname_attr_csv: Optional[str]
    :param verbose: Whether the train/test/eval steps should print progress.
        Defaults to True.
    :type verbose: bool
    :param read_type: 'all' or 'filename'; controls how attribute Parquet
        files are read (see CLAUDE.md's hfATLAS vs. legacy NLDI workflows).
        Defaults to 'all'.
    :type read_type: str
    :param make_plots: Whether to create and save diagnostic plots. Defaults
        to False.
    :type make_plots: bool
    :param same_test_ids: Whether all datasets being compared should share
        the same held-out test IDs. Defaults to True.
    :type same_test_ids: bool
    :param metrics: Response variable/metric names to process; if None, all
        metrics in the input dataset are processed. Defaults to None.
    :type metrics: Optional[List[str]]
    :param uncertainty: Uncertainty quantification configuration (forestci,
        bagging, and/or MAPIE blocks). Defaults to None.
    :type uncertainty: Optional[Dict[str, Any]]
    :param n_jobs: Number of parallel jobs for GridSearchCV. Defaults to 1.
    :type n_jobs: Optional[int]
    :param save_all_clusters: For clustering algorithms, whether to save
        every candidate cluster count rather than only the best. Defaults
        to False.
    :type save_all_clusters: bool
    :param regions: Optional sub-region configuration. None (default) means training runs
        CONUS-wide exactly as before this field existed. See RegionsConfig.
    :type regions: Optional[RegionsConfig]
    """
    task_type: str = 'regression'
    algorithms: Dict[str, Any]
    test_size: float = 0.3
    seed: int = 32
    name_attr_config: str
    name_attr_csv: Optional[str] = None
    colname_attr_csv: Optional[str] = None
    verbose: bool = True
    read_type: str = 'all'
    make_plots: bool = False
    same_test_ids: bool = True
    metrics: Optional[List[str]] = None
    uncertainty: Optional[Dict[str, Any]] = None
    n_jobs: Optional[int] = 1
    save_all_clusters: bool = False
    regions: Optional[RegionsConfig] = None

class PredConfig(BaseModel):
    """Validates the out-of-sample prediction configuration YAML (pred_config).

    :param name_attr_config: Name of the linked attribute config file.
    :type name_attr_config: str
    :param name_algo_config: Name of the linked algorithm config file.
    :type name_algo_config: str
    :param ds_type: Dataset type label used in output filenames. Defaults to
        'prediction'.
    :type ds_type: str
    :param write_type: Output file format ('parquet' or 'csv'). Defaults to
        'parquet'.
    :type write_type: str
    :param path_meta: Path to the metadata/predictor file (or directory)
        defining the prediction locations.
    :type path_meta: str
    :param pred_file_comid_colname: Column name identifying locations inside
        `path_meta`.
    :type pred_file_comid_colname: str
    :param path_gpkg_pred: Optional path to a GeoPackage of prediction
        locations. Defaults to None.
    :type path_gpkg_pred: Optional[str]
    :param pred_gpkg_lyr: Layer name inside `path_gpkg_pred`. Defaults to None.
    :type pred_gpkg_lyr: Optional[str]
    :param pred_gpkg_id_col: Identifier column inside `path_gpkg_pred`.
        Defaults to None.
    :type pred_gpkg_id_col: Optional[str]
    :param path_crosswalk_ids: Optional path to a Parquet file crosswalking
        aggregated identifiers (e.g. huc12) to the standard identifier (e.g.
        divide_id). Defaults to None.
    :type path_crosswalk_ids: Optional[str]
    :param crosswalk_target_col: The standardized identifier column name in
        `path_crosswalk_ids` (e.g. 'divide_id'). Only needed when performing
        a crosswalk/aggregation. Defaults to None.
    :type crosswalk_target_col: Optional[str]
    :param featureSource: The prediction-time featureSource provenance tag
        (see raftsutil.build_hfatl_feature_source). Training and prediction
        may legitimately run at different spatial scales (e.g. trained on
        gage-basin-aggregated attributes, predicting at HUC10/HUC14/divide
        scale), so this is independent of -- and takes priority over -- the
        training-time featureSource from the linked prep config. Defaults to
        None, in which case rafts_pred_algo.py falls back to the prep
        config's value.
    :type featureSource: Optional[str]
    :param donor_map_states: Region(s) rafts_map_donor_receiver.py maps donor-receiver
        pairings for -- only used when the algo config's task_type is 'clustering'.
        A flat list of 2-letter state codes for a single region (e.g. ['FL']), a dict
        of {region_name: [state codes]} for multiple named regions, or [] to opt out
        of the map entirely. Defaults to None, in which case that script falls back to
        its own built-in default regions (Florida and the Pacific Northwest) --
        distinct from an explicit [] opt-out. Declared here (previously validated only
        via the untyped PredConfigParser dict, unlike featureSource) so a malformed
        value (e.g. a plain string instead of a list/dict) is rejected here rather than
        failing confusingly wherever rafts_map_donor_receiver.py first iterates it. Note
        this does not catch a *mistyped key name* (e.g. donor_map_state) -- PredConfig
        has no model_config restricting extra keys, so an unrecognized key is silently
        ignored rather than rejected, same as any other config key in this model.
    :type donor_map_states: Optional[Union[List[str], Dict[str, List[str]]]]
    :param path_hf_finl_gpkg: Path to the hydrofabric GPKG containing the
        divides referenced by `path_crosswalk_ids`; required when running
        rafts_regn_params_gpkg.py. Defaults to None.
    :type path_hf_finl_gpkg: Optional[str]
    :param layr_hf_finl_gpkg: Layer to read from `path_hf_finl_gpkg`.
        Defaults to 'divides'.
    :type layr_hf_finl_gpkg: str
    :param overwrite_sql: Whether to overwrite an existing table when writing
        to the regionalization SQLite/GeoPackage. Defaults to False.
    :type overwrite_sql: bool
    :param algo_response_vars: Response variable/metric names to predict.
        Defaults to None.
    :type algo_response_vars: Optional[List[str]]
    :param algo_type: Base algorithm names to predict with (e.g. 'rf',
        'kmeans'). Defaults to None.
    :type algo_type: Optional[List[str]]
    :param algo_select: The specific trained algorithm name (e.g.
        'gower_agglomerative_k4') used for the final regionalization GPKG
        layer. Defaults to None.
    :type algo_select: Optional[str]
    :param mapie_alpha: Alpha values for MAPIE prediction interval
        estimation, read from the YAML's 'MAPIE_alpha' key. Defaults to None.
    :type mapie_alpha: Optional[List[float]]
    :param uncn_bnd_pred: Whether to apply min/max physical bounds to
        predictions. Defaults to False.
    :type uncn_bnd_pred: bool
    :param hf_fp_layer: Flowpath layer name, used by nexus mapping scripts.
        Defaults to 'flowpaths'.
    :type hf_fp_layer: str
    :param hf_fp_id_col: Flowpath identifier column name. Defaults to 'id'.
    :type hf_fp_id_col: str
    :param fp_toid_col: Flowpath downstream-id column name. Defaults to
        'toid'.
    :type fp_toid_col: str
    :param map_divide_id_col: Divide identifier column name. Defaults to
        'divide_id'.
    :type map_divide_id_col: str
    :param regions: Optional sub-region configuration. None (default) means prediction,
        pairing, and the final regionalization GPKG all run CONUS-wide exactly as before
        this field existed. See RegionsConfig. Declared independently from AlgoConfig.regions
        (same precedent as donor_map_states above) -- the two configs should agree for a
        coherent workflow, but that isn't enforced across separate YAML files.
    :type regions: Optional[RegionsConfig]
    """
    name_attr_config: str
    name_algo_config: str
    ds_type: str = 'prediction'
    write_type: str = 'parquet'
    path_meta: str
    pred_file_comid_colname: str

    # Optional prediction and routing parameters
    path_gpkg_pred: Optional[str] = None
    pred_gpkg_lyr: Optional[str] = None
    pred_gpkg_id_col: Optional[str] = None
    path_crosswalk_ids: Optional[str] = None
    crosswalk_target_col: Optional[str] = None
    featureSource: Optional[str] = None
    donor_map_states: Optional[Union[List[str], Dict[str, List[str]]]] = None
    path_hf_finl_gpkg: Optional[str] = None
    layr_hf_finl_gpkg: str = 'divides'
    overwrite_sql: bool = False
    algo_response_vars: Optional[List[str]] = None
    algo_type: Optional[List[str]] = None
    algo_select: Optional[str] = None
    # Field name is snake_case per convention; 'MAPIE_alpha' is kept as the
    # validation alias since that's the YAML key every existing config uses.
    mapie_alpha: Optional[List[float]] = Field(default=None, alias='MAPIE_alpha')
    uncn_bnd_pred: bool = False

    # Topology and Flowpath Mappings (for nexus mapping scripts)
    hf_fp_layer: str = 'flowpaths'
    hf_fp_id_col: str = 'id'
    fp_toid_col: str = 'toid'
    map_divide_id_col: str = 'divide_id'
    regions: Optional[RegionsConfig] = None

# %% Pydantic model pipeline validation 

class UncertaintyConfig(BaseModel):
    """
    Validates the dictionary structure of the 'Uncertainty' key 
    saved within the model joblib file.
    """
    # ForestCI keys usually follow pattern ci_95, ci_90 etc.
    forestci: Optional[Dict[str, Dict[str, Any]]] = None
    
    # Bagging keys
    bagging_confidence_interval: Optional[Dict[str, Any]] = None
    
    # MAPIE specific configurations if saved in config dict
    mapie: Optional[List[Dict[str, Any]]] = None
    
class ModelMetadata(BaseModel):
    """
    Validates the top-level dictionary loaded from the .joblib file
    in rafts_pred_algo_new.py.
    """
    pipeline: Any #BaseEstimator # Should be a sklearn object (e.g. pipeline, model_selection )
    X_train_shape: Optional[Tuple[int, int]] # Required for ForestCI
    mapie: Optional[Any] = None #Optional[MapieRegressor] = None # The MAPIE regressor object (not just config)
    Uncertainty: Optional[Dict[str, Any]] = None # The uncertainty configuration dict

    @field_validator("X_train_shape")
    def validate_shape(cls, v):
        if v is None:
            return v
        if not (isinstance(v, tuple) and len(v) == 2 and all(isinstance(i, int) for i in v)):
            raise ValueError("X_train_shape must be a tuple of two integers (n_samples, n_features)")
        return v

    @model_validator(mode="after")
    def validate_uncertainty(self) -> "ModelMetadata":
        """
        Deep validation of the Uncertainty dictionary if it exists.
        """
        unc = self.Uncertainty 
        if unc is None:
            return self

        # forestci block
        if "forestci" in unc:
            forestci = unc["forestci"]
            pattern = r"^ci_(\d{1,2}|[1-9][0-9])$"
            matched = [k for k in forestci if re.match(pattern, k)]
            if not matched:
                raise ValueError("forestci must contain at least one 'ci_zz' with zz between 1 and 99")
            for k in matched:
                ci = forestci[k]
                for bound in ["upper_bound", "lower_bound"]:
                    arr = ci.get(bound)
                    if not (isinstance(arr, np.ndarray) and np.issubdtype(arr.dtype, np.floating)):
                        raise ValueError(f"forestci -> {k} -> {bound} must be a NumPy array of floats")

        # bagging_confidence_interval block
        if "bagging_confidence_interval" in unc:
            for key in ["bagging_std_pred", "bagging_mean_pred"]:
                arr = unc.get(key)
                if not (isinstance(arr, np.ndarray) and np.issubdtype(arr.dtype, np.floating)):
                    raise ValueError(f"{key} must be a NumPy array of floats")

            confs = unc.get("bagging_confidence_intervals", {})
            pattern = r"^confidence_level_(\d{1,2}|[1-9][0-9])$"
            matched = [k for k in confs if re.match(pattern, k)]
            if not matched:
                raise ValueError("bagging_confidence_intervals must contain keys like 'confidence_level_zz'")

            for k in matched:
                for bound in ["upper_bound", "lower_bound"]:
                    arr = confs[k].get(bound)
                    if not (isinstance(arr, np.ndarray) and np.issubdtype(arr.dtype, np.floating)):
                        raise ValueError(f"bagging_confidence_intervals -> {k} -> {bound} must be a NumPy array of floats")

        return self