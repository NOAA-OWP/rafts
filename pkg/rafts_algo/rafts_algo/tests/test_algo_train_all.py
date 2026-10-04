'''
Unit testing for the rafts_algo functions split across the following files:
 - rafts_algo_train.py
 - plots.py
 - tfrm_attr.py

example
> cd /path/to/rafts_algo/rafts_algo/tests/
> python -m unittest test_algo_train_eval

> coverage run -m unittest test_algo_train_eval.py  
> coverage report
> coverage html 

 # Useful for running in ipynb:
if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

Note on mocking (project convention is to avoid it -- see README.claude/CLAUDE.md
"Unit tests should avoid mocking"):
This file uses @patch/MagicMock in a couple dozen places, which is at odds with
that guidance on its face. In practice they fall into two different buckets, and
only one of them is a real problem:
  - Legitimate: isolating a genuinely external or destructive side effect that
    would otherwise slow down or pollute the test -- e.g. @patch('joblib.dump')
    to avoid an actual disk write, @patch('pathlib.Path.mkdir')/.exists() to
    check a helper's directory-creation logic without touching the filesystem,
    or @patch('pynhd.NLDI.navigate_byid') to avoid a real network call. These
    don't replace the logic under test; they replace an I/O boundary around it.
  - Logic-hiding: patching away the actual computation a test claims to verify,
    leaving only a call-count assertion.
'''
import os
import matplotlib
# MUST set headless backend before pyplot or rafts_algo imports to prevent Tkinter Tcl_AsyncDelete threading aborts
matplotlib.use('Agg')

import unittest
from unittest.mock import patch, MagicMock, mock_open
import pandas as pd
import dask.dataframe as dd
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import tempfile
from pathlib import Path
import rafts_algo.rafts_algo_train as raftsalgo
import rafts_algo.utils as raftsutil
import rafts_algo.plots as raftsplots
import warnings
from sklearn.exceptions import ConvergenceWarning
import xarray as xr
import numpy as np
import forestci as fci
from scipy import stats as st
from sklearn.utils import resample
from sklearn.pipeline import Pipeline
from mapie.regression import MapieRegressor
import yaml
import shutil
import logging
import io
import sys
from contextlib import redirect_stdout
from types import SimpleNamespace
# Ignore some warnings that arise during testing:
import pytest
import geopandas as gpd
from shapely import Point
import joblib 
import sqlite3
from pydantic import ValidationError
from rafts_algo.schemas.pydantic_schemas import AlgoConfig, PredConfig

# Tell pytest natively to ignore these specific warnings for this entire file
pytestmark = pytest.mark.filterwarnings(
    "ignore:.*disp.*iprint.*:DeprecationWarning",
    "ignore:.*lbfgs failed to converge.*:sklearn.exceptions.ConvergenceWarning",
    "ignore:.*Some inputs do not have OOB scores.*:UserWarning",
    "ignore:.*Removed duplicate entries from 'train_sizes'.*:RuntimeWarning"
)

# Silence warnings when running via `python -m unittest` instead of pytest
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
warnings.filterwarnings("ignore", message=".*Unknown solver options: iprint.*")

# %% UNIT TESTING FOR AttrConfigAndVars
parent_dir_test = Path(__file__).parent
dir_test_data = Path(parent_dir_test,"test_data")

# Set up logging, do not write to file!
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TestAttrConfigAndVars(unittest.TestCase):
    print("Testing AttrConfigAndVars")
    @patch('builtins.open', new_callable=mock_open, read_data='''
            attr_select:
            - attr_vars: [attr1, attr2, attr3]
            file_io:
            - dir_base: "{home_dir}/base_dir"
            - dir_db_attrs: "{dir_base}/db_attrs"
            - dir_std_base: "{dir_base}/std_base"
            - home_dir: "{home_dir}"
            formulation_metadata:
            - datasets: ["dataset1", "dataset2"]
                ''')
    @patch('pathlib.Path.home', return_value='/mocked/home')
    def test_read_attr_config(self, mock_home, mock_file):
        print('    Testing _read_attr_config')
        path = Path(dir_test_data / 'attr_config.yaml')
        attr_obj = raftsutil.AttrConfigAndVars(path)
        attr_obj._read_attr_config()

        # Test if the file is opened with the correct path
        mock_file.assert_called_once_with(path, 'r')

        # Test if Path.home() was called
        mock_home.assert_called()
        print(attr_obj.attrs_cfg_dict)
        # Test the parsed data from the config
        expected_home_dir = Path('/mocked/home')
        expected_attrs_cfg_dict = {
            'attrs_sel': ['attr1', 'attr2', 'attr3'],
            'dir_db_attrs': expected_home_dir / 'base_dir' / 'db_attrs',
            'dir_std_base': expected_home_dir / 'base_dir' / 'std_base',
            'dir_base': expected_home_dir / 'base_dir',
            'home_dir': expected_home_dir,
            'datasets': ['dataset1', 'dataset2']
        }
           
        self.assertEqual(attr_obj.attrs_cfg_dict, expected_attrs_cfg_dict)
        print("✅ test_read_attr_config test passed.")

class TestAttrConfigAndVarsBranches(unittest.TestCase):
    """
    Covers AttrConfigAndVars._read_attr_config branches TestAttrConfigAndVars
    doesn't reach: a nonexistent config path, an attr_select with no '_vars'
    keys (falls back to 'all'), an unparseable dir_db_attrs template, the
    name_prep_config success path, and the legacy-fallback failure path.
    Uses real temp files rather than @patch('builtins.open', mock_open(...))
    like the existing test does, per the project's mocking-avoidance
    preference.

    One related branch is documented rather than tested: the
    `else: logging.warning(f"Prep config {path_prep_config.name} not
    found.")` alongside `if path_prep_config.exists():` can't be reached
    through build_cfig_path()'s actual contract -- like the analogous check
    in AlgoConfigParser, build_cfig_path() either raises FileNotFoundError
    when name_prep_config can't be found anywhere, or returns a Path it has
    already confirmed exists.
    """

    def _write_attr_cfg(self, tmpdir, extra=None):
        tmpdir = Path(tmpdir)
        cfg = {
            'attr_select': [{'attr_vars': ['attr1', 'attr2']}],
            'file_io': [
                {'dir_base': str(tmpdir / 'base_dir')},
                {'dir_db_attrs': '{dir_base}/db_attrs'},
                {'dir_std_base': '{dir_base}/std_base'},
            ],
            'formulation_metadata': [{'datasets': ['dataset1']}],
        }
        if extra:
            for key, value in extra.items():
                cfg[key] = value
        path_cfg = tmpdir / "attr_config.yaml"
        with open(path_cfg, "w") as f:
            yaml.safe_dump(cfg, f)
        return path_cfg

    def test_attr_config_path_does_not_exist(self):
        with self.assertRaises(ValueError):
            raftsutil.AttrConfigAndVars(Path("/nonexistent/attr_config.yaml"))._read_attr_config()

    def test_attrs_sel_defaults_to_all_when_no_vars_keys_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path_cfg = self._write_attr_cfg(tmpdir, extra={'attr_select': [{'other_key': ['x']}]})
            with self.assertLogs(level='WARNING') as cm:
                attr_obj = raftsutil.AttrConfigAndVars(path_cfg)
                attr_obj._read_attr_config()
            self.assertEqual(attr_obj.attrs_cfg_dict['attrs_sel'], 'all')
            self.assertTrue(any("Assuming all attributes desired" in log for log in cm.output))

    def test_dir_db_attrs_unresolvable_template_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path_cfg = self._write_attr_cfg(
                tmpdir, extra={'file_io': [
                    {'dir_base': str(Path(tmpdir) / 'base_dir')},
                    {'dir_db_attrs': '{undefined_placeholder}/db_attrs'},
                    {'dir_std_base': '{dir_base}/std_base'},
                ]}
            )
            with self.assertRaises(ValueError):
                raftsutil.AttrConfigAndVars(path_cfg)._read_attr_config()

    def test_name_prep_config_success_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            prep_cfg_path = Path(tmpdir) / "prep_config.yaml"
            with open(prep_cfg_path, "w") as f:
                yaml.safe_dump({'formulation_metadata': [{'dataset_name': 'from_prep_config'}]}, f)

            path_cfg = self._write_attr_cfg(tmpdir, extra={
                'file_io': [
                    {'dir_base': str(Path(tmpdir) / 'base_dir')},
                    {'dir_db_attrs': '{dir_base}/db_attrs'},
                    {'dir_std_base': '{dir_base}/std_base'},
                    {'name_prep_config': 'prep_config.yaml'},
                ]
            })
            attr_obj = raftsutil.AttrConfigAndVars(path_cfg)
            attr_obj._read_attr_config()
            self.assertEqual(attr_obj.attrs_cfg_dict['datasets'], ['from_prep_config'])

    def test_legacy_fallback_raises_when_datasets_unresolvable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # No name_prep_config, and formulation_metadata has no 'datasets' entry
            # -> the legacy fallback's list comprehension indexes into an empty
            # list, raising IndexError, caught and re-raised as ValueError.
            path_cfg = self._write_attr_cfg(tmpdir, extra={
                'formulation_metadata': [{'formulation_base': 'no_datasets_here'}]
            })
            with self.assertRaises(ValueError):
                raftsutil.AttrConfigAndVars(path_cfg)._read_attr_config()


class TestFsReadAttrComid(unittest.TestCase):
    @patch('rafts_algo.utils.dd.read_parquet')
    def test_rafts_read_attr_comid(self, mock_dd_read_parquet):
        print("    Testing rafts_read_attr_comid")

        # Create a Dask DataFrame from a Pandas DataFrame
        mock_pdf = pd.DataFrame({
            'data_source': ['hydroatlas__v1','hydroatlas__v1'],
            'dl_timestamp': ['2024-07-26 08:59:36','2024-07-26 08:59:36'],
            'attribute': ['pet_mm_s01', 'cly_pc_sav'],
            'value': [58, 21],
            'featureID': ['1520007','1520007'],
            'featureSource': ['COMID','COMID']
        })
        mock_ddf = dd.from_pandas(mock_pdf, npartitions=1)

        # Patch Dask read_parquet to return this mock DDF
        mock_dd_read_parquet.return_value = mock_ddf

        # Inputs
        dir_db_attrs = 'mock_dir'
        comids_resp = ['1520007']
        attrs_sel = ['pet_mm_s01', 'cly_pc_sav']

        # Call function
        result_df = raftsutil.rafts_read_attr_comid(
            dir_db_attrs=dir_db_attrs,
            comids_resp=comids_resp,
            attrs_sel=attrs_sel
        )

        # Assertions
        self.assertEqual(result_df.shape[0], 2)
        self.assertIn('1520007', result_df['featureID'].values)
        self.assertIn('pet_mm_s01', result_df['attribute'].values)
        self.assertIn('COMID', result_df['featureSource'].values)
        self.assertIn('value', result_df.columns)
        self.assertIn('data_source', result_df.columns)
        print("✅ rafts_read_attr_comid muliple-row test passed.")

        # When only one attribute requested
        single_result = raftsutil.rafts_read_attr_comid(dir_db_attrs=dir_db_attrs,
                                                            comids_resp= comids_resp,attrs_sel= ['pet_mm_s01'])
        self.assertIn('pet_mm_s01',single_result['attribute'].values)
        self.assertNotIn('cly_pc_sav',single_result['attribute'].values)

        # When COMID requested that doesn't exist
        with self.assertLogs(level="INFO") as cm:
            raftsutil.rafts_read_attr_comid(
                dir_db_attrs=dir_db_attrs,
                comids_resp=['010101010'],
                attrs_sel=['pet_mm_s01']
            )
        self.assertTrue(
            any("None of the provided featureIDs exist" in m for m in cm.output)
        )

        # When attribute requested that doesn't exist
        with self.assertLogs(level="INFO") as cm:
            raftsutil.rafts_read_attr_comid(dir_db_attrs=dir_db_attrs,
                                            comids_resp= comids_resp,
                                            attrs_sel= ['nonexistent'])
        self.assertTrue(any("Missing attributes include:" in m for m in cm.output))
        self.assertTrue(any("nonexistent" in m for m in cm.output))
        print("✅ rafts_read_attr_comid single-row test passed.")

    def _mock_ddf(self, feature_ids=('1520007', '1520007'), attributes=('pet_mm_s01', 'cly_pc_sav'), values=(58, 21)):
        mock_pdf = pd.DataFrame({
            'data_source': ['hydroatlas__v1'] * len(feature_ids),
            'dl_timestamp': ['2024-07-26 08:59:36'] * len(feature_ids),
            'attribute': list(attributes),
            'value': list(values),
            'featureID': list(feature_ids),
            'featureSource': ['COMID'] * len(feature_ids),
        })
        return dd.from_pandas(mock_pdf, npartitions=1)

    @patch('rafts_algo.utils.dd.read_parquet')
    def test_comids_resp_none_reads_all_locations(self, mock_dd_read_parquet):
        mock_dd_read_parquet.return_value = self._mock_ddf()
        result_df = raftsutil.rafts_read_attr_comid(
            dir_db_attrs='mock_dir', comids_resp=None, attrs_sel=['pet_mm_s01', 'cly_pc_sav']
        )
        self.assertEqual(result_df.shape[0], 2)
        mock_dd_read_parquet.assert_called_once()

    @patch('rafts_algo.utils.dd.read_parquet')
    def test_invalid_read_type_raises(self, mock_dd_read_parquet):
        mock_dd_read_parquet.return_value = self._mock_ddf()
        with self.assertRaises(ValueError):
            raftsutil.rafts_read_attr_comid(
                dir_db_attrs='mock_dir', comids_resp=['1520007'],
                attrs_sel=['pet_mm_s01'], read_type='bogus'
            )

    @patch('rafts_algo.utils.dd.read_parquet')
    def test_read_type_filename_with_matching_file(self, mock_dd_read_parquet):
        # read_type='filename' scans the real directory for files whose name
        # contains '_{comid}_' before ever calling dd.read_parquet -- needs a
        # real file on disk even though its content is still mocked.
        mock_dd_read_parquet.return_value = self._mock_ddf()
        with tempfile.TemporaryDirectory() as tmpdir:
            # Non-empty content: rafts_read_attr_comid deletes 0-byte .parquet
            # files before the filename-matching logic ever runs.
            (Path(tmpdir) / "comid_1520007_attrs.parquet").write_bytes(b"placeholder")
            result_df = raftsutil.rafts_read_attr_comid(
                dir_db_attrs=tmpdir, comids_resp=['1520007'],
                attrs_sel=['pet_mm_s01'], read_type='filename'
            )
        self.assertEqual(result_df.shape[0], 1)
        # Called with the list of matching Path objects, not the directory itself.
        called_arg = mock_dd_read_parquet.call_args.args[0]
        self.assertIsInstance(called_arg, list)

    @patch('rafts_algo.utils.dd.read_parquet')
    def test_read_type_filename_no_match_falls_back_to_all(self, mock_dd_read_parquet):
        mock_dd_read_parquet.return_value = self._mock_ddf()
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "unrelated_file.parquet").touch()
            with self.assertLogs(level="WARNING") as cm:
                raftsutil.rafts_read_attr_comid(
                    dir_db_attrs=tmpdir, comids_resp=['1520007'],
                    attrs_sel=['pet_mm_s01'], read_type='filename'
                )
        self.assertTrue(any("Falling back to reading 'all' partitions" in m for m in cm.output))

    @patch('rafts_algo.utils.dd.read_parquet')
    def test_removes_empty_and_na_parquet_files(self, mock_dd_read_parquet):
        mock_dd_read_parquet.return_value = self._mock_ddf()
        with tempfile.TemporaryDirectory() as tmpdir:
            path_empty = Path(tmpdir) / "comid_1520007_empty.parquet"
            path_empty.touch()  # 0 bytes
            path_na = Path(tmpdir) / "comid_1520007_NA_attrs.parquet"
            path_na.write_bytes(b"not actually empty")

            raftsutil.rafts_read_attr_comid(
                dir_db_attrs=tmpdir, comids_resp=['1520007'], attrs_sel=['pet_mm_s01']
            )
            self.assertFalse(path_empty.exists(), "0-byte parquet file should have been removed")
            self.assertFalse(path_na.exists(), "'_NA_' parquet file should have been removed")

    @patch('rafts_algo.utils.dd.read_parquet')
    def test_reindex_true(self, mock_dd_read_parquet):
        mock_dd_read_parquet.return_value = self._mock_ddf()
        result_df = raftsutil.rafts_read_attr_comid(
            dir_db_attrs='mock_dir', comids_resp=['1520007'],
            attrs_sel=['pet_mm_s01'], reindex=True
        )
        self.assertIsInstance(result_df, pd.DataFrame)

    @patch('rafts_algo.utils.dd.read_parquet')
    def test_s3_truthy_sets_storage_options(self, mock_dd_read_parquet):
        # _s3 is a future-feature placeholder; passing a truthy value exercises
        # the storage_options assignment branch (not yet otherwise wired up).
        mock_dd_read_parquet.return_value = self._mock_ddf()
        result_df = raftsutil.rafts_read_attr_comid(
            dir_db_attrs='mock_dir', comids_resp=['1520007'],
            attrs_sel=['pet_mm_s01'], _s3=True,
        )
        self.assertIsInstance(result_df, pd.DataFrame)

    @patch('rafts_algo.utils.dd.read_parquet')
    def test_comids_resp_none_drops_incomplete_locations(self, mock_dd_read_parquet):
        # Two locations: '1520007' has both requested attributes, '9999999'
        # only has one -- the intersection logic should drop the latter and
        # log how many locations were dropped.
        mock_dd_read_parquet.return_value = self._mock_ddf(
            feature_ids=('1520007', '1520007', '9999999'),
            attributes=('pet_mm_s01', 'cly_pc_sav', 'pet_mm_s01'),
            values=(58, 21, 40),
        )
        with self.assertLogs(level="INFO") as cm:
            result_df = raftsutil.rafts_read_attr_comid(
                dir_db_attrs='mock_dir', comids_resp=None,
                attrs_sel=['pet_mm_s01', 'cly_pc_sav'],
            )
        self.assertTrue(any("Dropped 1 locations" in m for m in cm.output))
        self.assertNotIn('9999999', result_df['featureID'].values)
        self.assertIn('1520007', result_df['featureID'].values)

    @patch('rafts_algo.utils.dd.read_parquet')
    def test_warns_on_unexpected_na_values(self, mock_dd_read_parquet):
        mock_dd_read_parquet.return_value = self._mock_ddf(values=(58, np.nan))
        with self.assertLogs(level="WARNING") as cm:
            raftsutil.rafts_read_attr_comid(
                dir_db_attrs='mock_dir', comids_resp=['1520007'],
                attrs_sel=['pet_mm_s01', 'cly_pc_sav'],
            )
        self.assertTrue(any("unexpected NA values" in m for m in cm.output))


class TestHomeDirUtilities(unittest.TestCase):
    def test_make_home_dir(self):
        # 1. Test None/Empty triggers Path.home()
        self.assertEqual(raftsutil._make_home_dir([]), Path.home())
        self.assertEqual(raftsutil._make_home_dir([None]), Path.home())

        # 2. Test tilde expansion
        self.assertEqual(raftsutil._make_home_dir(["~/some/path"]), Path.home() / "some/path")

        # 3. Test explicit path that doesn't exist (logs warning, uses default system home)
        with self.assertLogs(level='WARNING'):
            res = raftsutil._make_home_dir(["/fake/path/that/does/not/exist/12345"])
            self.assertEqual(res, Path.home())

# %% UNIT TESTING FOR AlgoConfigParser
class TestAlgoConfigParser(unittest.TestCase):
    print("Testing AlgoConfigParser")

    def setUp(self):
        self.test_data_dir = dir_test_data

    def load_yaml(self, filename):
        filepath = self.test_data_dir / filename
        with open(filepath, "r") as f:
            return yaml.safe_load(f), filepath

    def test_01_no_uncertainty_defaults(self):
        _, filepath = self.load_yaml("test_algo_config_01_nouncertainty.yaml")
        config = raftsutil.AlgoConfigParser(filepath)
        config._read_algo_config()
        self.assertIsInstance(config.algo_cfg_unc_dict, dict)
        self.assertIn("algo_cfg_dict", config.algo_cfg_unc_dict)
        print("Completed Test AlgoConfig parsing - YAML file #1")

    def test_02_uncertainty_defaults(self):
        _, filepath = self.load_yaml("test_algo_config_02_uncertainty.yaml")
        config = raftsutil.AlgoConfigParser(filepath)
        config._read_algo_config()
        self.assertIn("mapie", config.algo_cfg_unc_dict["algo_unc_dict"]["uncertainty_cfg"])
        print("Completed Test AlgoConfig parsing - YAML file #2")

    def test_03_uncertainty_params_custom(self):
        _, filepath = self.load_yaml("test_algo_config_03_uncertainty_nodefaults.yaml")
        config = raftsutil.AlgoConfigParser(filepath)
        config._read_algo_config()

        self.assertIn("algo_unc_dict", config.algo_cfg_unc_dict)
        self.assertIn("uncertainty_cfg", config.algo_cfg_unc_dict["algo_unc_dict"])
        self.assertIn("bagging", config.algo_cfg_unc_dict["algo_unc_dict"]["uncertainty_cfg"])

        bagging_cfg = config.algo_cfg_unc_dict["algo_unc_dict"]["uncertainty_cfg"]["bagging"]
        self.assertIsInstance(bagging_cfg, list)
        self.assertIsInstance(bagging_cfg[0], dict)
        self.assertIn("n_algos", bagging_cfg[0])
        n_algos_test = 20
        self.assertEqual(bagging_cfg[0]["n_algos"], n_algos_test)
        print("Completed Test AlgoConfig parsing - YAML file #3")

    def test_04_missing_required_parameter(self):
        _, filepath = self.load_yaml("test_algo_config_04_errortest_parammissing.yaml")
        config = raftsutil.AlgoConfigParser(filepath)
        with self.assertRaises(KeyError) as context:
            config._read_algo_config()
        self.assertIn("'algorithms'", str(context.exception))
        print("Completed Test AlgoConfig parsing - YAML file #4")

    def test_04_pydantic_missing_required_parameter(self):
        raw, _ = self.load_yaml("test_algo_config_04_errortest_parammissing.yaml")
        with self.assertRaises(ValidationError) as context:
            AlgoConfig(**raw)
        self.assertIn("algorithms", str(context.exception))
        print("Completed Pydantic AlgoConfig validation - missing parameter")

    def test_05_wrong_datatype(self):
        _, filepath = self.load_yaml("test_algo_config_05_errortest_paramdatatype.yaml")
        config = raftsutil.AlgoConfigParser(filepath)
        with self.assertRaises(TypeError) as context:
            config._read_algo_config()
        self.assertIn("'seed' must be an integer", str(context.exception))
        print("Completed Test AlgoConfig parsing - YAML file #5")

    def test_05_pydantic_wrong_datatype(self):
        raw, _ = self.load_yaml("test_algo_config_05_errortest_paramdatatype.yaml")
        with self.assertRaises(ValidationError) as context:
            AlgoConfig(**raw)
        self.assertIn("seed", str(context.exception))
        print("Completed Pydantic AlgoConfig validation - wrong datatype")

    def test_06_invalid_uncertainty_structure(self):
        _, filepath = self.load_yaml("test_algo_config_06_errortest_unc_param_struct.yaml")
        config = raftsutil.AlgoConfigParser(filepath)
        with self.assertRaises(TypeError) as context:
            config._read_algo_config()
        self.assertIn("The 'uncertainty' block must be a dictionary", str(context.exception))
        print("Completed Test AlgoConfig parsing - YAML file #6")


    def test_Print(self):
        print("✅ TestAlgoConfigParser test passed.")


class TestAlgoConfigParserValidationBranches(unittest.TestCase):
    """
    Covers AlgoConfigParser._read_algo_config's per-field type/range validation
    branches, none of which were previously exercised (existing tests only
    cover the happy path, a missing 'algorithms' key, one wrong-datatype case
    for 'seed', and a malformed 'uncertainty' block).

    Two branches documented rather than tested here are genuinely unreachable
    given the surrounding code, not just untested:
      - The `if not algo_cfg_dict['path_attr_config'].exists(): raise
        ValueError(...)` check (right after build_cfig_path() is called):
        build_cfig_path() itself either raises FileNotFoundError when the
        configured name_attr_config can't be found anywhere, or returns a
        Path it has already confirmed exists -- it never returns a
        non-existent Path. So this check's condition can never evaluate True
        through build_cfig_path's actual contract.
      - The `else: confidence_levels = [95]` branch paired with
        `if not algo_unc_dict == {}:`: algo_unc_dict is always constructed as
        {'uncertainty_cfg': ...}, so it can never equal {} -- the intended
        check (falling back to a default when no uncertainty config was
        given) was written against the wrong variable and its else branch
        can never run.
    """

    def setUp(self):
        self.test_data_dir = dir_test_data
        with open(self.test_data_dir / "test_algo_config_01_nouncertainty.yaml", "r") as f:
            self.base_cfg = yaml.safe_load(f)

    def _write_cfg(self, tmpdir, overrides=None, pop_keys=None):
        cfg = dict(self.base_cfg)
        if pop_keys:
            for k in pop_keys:
                cfg.pop(k, None)
        if overrides:
            cfg.update(overrides)
        tmpdir = Path(tmpdir)
        shutil.copy(self.test_data_dir / "attr_config.yaml", tmpdir / "attr_config.yaml")
        path_cfg = tmpdir / "algo_config.yaml"
        with open(path_cfg, "w") as f:
            yaml.safe_dump(cfg, f)
        return path_cfg

    def test_algo_config_path_does_not_exist(self):
        with self.assertRaises(ValueError):
            raftsutil.AlgoConfigParser(Path("/nonexistent/algo_config.yaml"))._read_algo_config()

    def test_missing_seed_and_test_size_falls_back_to_algotraineval_defaults(self):
        # Neither 'seed' nor 'test_size' is truthy -> the try block introspects
        # AlgoTrainEval.__init__'s defaults via inspect.signature.
        with tempfile.TemporaryDirectory() as tmpdir:
            path_cfg = self._write_cfg(tmpdir, pop_keys=["seed", "test_size"])
            config = raftsutil.AlgoConfigParser(path_cfg)
            config._read_algo_config()
            algo_cfg_dict = config.algo_cfg_unc_dict["algo_cfg_dict"]
            self.assertIsInstance(algo_cfg_dict["seed"], int)
            self.assertIsInstance(algo_cfg_dict["test_size"], float)

    def test_test_size_wrong_type_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path_cfg = self._write_cfg(tmpdir, overrides={"test_size": "not_a_float"})
            with self.assertRaises(TypeError):
                raftsutil.AlgoConfigParser(path_cfg)._read_algo_config()

    def test_test_size_out_of_range_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path_cfg = self._write_cfg(tmpdir, overrides={"test_size": 1.5})
            with self.assertRaises(ValueError):
                raftsutil.AlgoConfigParser(path_cfg)._read_algo_config()

    def test_read_type_wrong_type_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path_cfg = self._write_cfg(tmpdir, overrides={"read_type": 5})
            with self.assertRaises(TypeError):
                raftsutil.AlgoConfigParser(path_cfg)._read_algo_config()

    def test_read_type_invalid_value_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path_cfg = self._write_cfg(tmpdir, overrides={"read_type": "bogus"})
            with self.assertRaises(ValueError):
                raftsutil.AlgoConfigParser(path_cfg)._read_algo_config()

    def test_make_plots_wrong_type_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path_cfg = self._write_cfg(tmpdir, overrides={"make_plots": "yes"})
            with self.assertRaises(TypeError):
                raftsutil.AlgoConfigParser(path_cfg)._read_algo_config()

    def test_same_test_ids_wrong_type_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path_cfg = self._write_cfg(tmpdir, overrides={"same_test_ids": "yes"})
            with self.assertRaises(TypeError):
                raftsutil.AlgoConfigParser(path_cfg)._read_algo_config()

    def test_verbose_wrong_type_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path_cfg = self._write_cfg(tmpdir, overrides={"verbose": "yes"})
            with self.assertRaises(TypeError):
                raftsutil.AlgoConfigParser(path_cfg)._read_algo_config()

    def test_confidence_levels_wrong_type_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            overrides = {"uncertainty": {"confidence_levels": 95}}  # not a list
            path_cfg = self._write_cfg(tmpdir, overrides=overrides)
            with self.assertRaises(TypeError):
                raftsutil.AlgoConfigParser(path_cfg)._read_algo_config()

    def test_confidence_levels_out_of_range_entry_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            overrides = {"uncertainty": {"confidence_levels": [95, 150]}}
            path_cfg = self._write_cfg(tmpdir, overrides=overrides)
            with self.assertRaises(ValueError):
                raftsutil.AlgoConfigParser(path_cfg)._read_algo_config()

    def test_mapie_alpha_invalid_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            overrides = {"uncertainty": {"mapie": [{"alpha": [1.5]}]}}  # not between 0 and 1
            path_cfg = self._write_cfg(tmpdir, overrides=overrides)
            with self.assertRaises(ValueError):
                raftsutil.AlgoConfigParser(path_cfg)._read_algo_config()

    def test_mapie_method_invalid_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            overrides = {"uncertainty": {"mapie": [{"alpha": [0.1], "method": "bogus"}]}}
            path_cfg = self._write_cfg(tmpdir, overrides=overrides)
            with self.assertRaises(ValueError):
                raftsutil.AlgoConfigParser(path_cfg)._read_algo_config()

    def test_mapie_cv_not_int_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            overrides = {"uncertainty": {"mapie": [
                {"alpha": [0.1], "method": "plus", "cv": "ten"}
            ]}}
            path_cfg = self._write_cfg(tmpdir, overrides=overrides)
            with self.assertRaises(TypeError):
                raftsutil.AlgoConfigParser(path_cfg)._read_algo_config()

    def test_mapie_agg_function_invalid_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            overrides = {"uncertainty": {"mapie": [
                {"alpha": [0.1], "method": "plus", "cv": 10, "agg_function": "bogus"}
            ]}}
            path_cfg = self._write_cfg(tmpdir, overrides=overrides)
            with self.assertRaises(ValueError):
                raftsutil.AlgoConfigParser(path_cfg)._read_algo_config()

class TestCheckAttributesExist(unittest.TestCase):
    print('Testing _check_attributes_exist')
    # NOTE: _check_attributes_exist has two more defensive branches that look
    # reachable but, like the dead branches already documented in utils.py's
    # AlgoConfigParser/AttrConfigAndVars tests, aren't:
    #   - `else: bad_comids = counts.index.tolist() if vec_missing else []`
    #     (the isinstance(vec_missing, pd.Series) check's else branch): counts
    #     is always a pandas Series (a groupby().count() result), so
    #     `counts != len(attrs_sel)` is always a Series too, even when counts
    #     has a single row -- vec_missing can never be a bare bool.
    #   - `if isinstance(attrs_sel, list):`: attrs_sel is unconditionally
    #     coerced to a pd.Series a few lines earlier in this same function
    #     (both branches of the preceding if/else reassign it), so it can
    #     never still be a plain list by the time this check runs.
    def test_check_attributes_exist(self):
        mock_pdf = pd.DataFrame({
            'data_source': 'hydroatlas__v1',
            'dl_timestamp': '2024-07-26 08:59:36',
            'attribute': ['pet_mm_s01', 'cly_pc_sav','pet_mm_s01', 'cly_pc_sav'],
            'value': [58, 21, 65,32],
            'featureID': ['1520007','1520007','1623207','1623207'],
            'featureSource': 'COMID'
            })
        
        with warnings.catch_warnings(record = True) as w:
            warnings.simplefilter("always")
            raftsutil._check_attributes_exist(mock_pdf,pd.Series(['pet_mm_s01','cly_pc_sav']))
            self.assertEqual(len(w),0)

        mock_pdf_bad = mock_pdf.copy()
        mock_pdf_bad.drop(index=0, inplace=True)

        with self.assertLogs(level="INFO") as cm:
            raftsutil._check_attributes_exist(
                mock_pdf_bad, pd.Series(['pet_mm_s01', 'cly_pc_sav'])
            )

        self.assertTrue(any("Not all featureID groupings" in m for m in cm.output))
        self.assertTrue(any("TOTAL unique locations with missing attributes" in m for m in cm.output))
        self.assertTrue(any("TOTAL MISSING ATTRS" in m for m in cm.output))
        
        print("✅ _check_attributes_exist test passed.")

class TestRaftsRetrNhdpComids(unittest.TestCase):

    def test_rafts_retr_nhdp_comids(self):

        # Define test inputs
        featureSource = 'nwissite'
        featureID = 'USGS-{gage_id}'
        gage_ids = ["01031500", "08070000"]

        result = raftsutil.rafts_retr_nhdp_comids_geom(featureSource, featureID, gage_ids)
        result_comids = [str(x) for x in result['comid'].tolist()]

        # Assertions
        self.assertListEqual(result_comids, ['1722317', '1520007'])
        self.assertEqual(result.columns.tolist(), ['comid','gage_id', 'geometry'])
        print("✅ test_rafts_retr_nhdp_comids test passed.")

    def test_rafts_retr_nhdp_comids_geom_wrap_cached(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gpkg_path = Path(tmpdir) / "cache.gpkg"
            
            # Create a dummy cached file with our target gage_id
            gdf = gpd.GeoDataFrame({
                'gage_id': ['gage1'],
                'comid': ['111']
            }, geometry=[Point(0,0)], crs="EPSG:4326")
            gdf.to_file(gpkg_path, driver="GPKG", layer="outlet")
            
            # Request the exact same gage_id to hit the fully cached branch
            result = raftsutil.rafts_retr_nhdp_comids_geom_wrap(
                path_save_gpkg=gpkg_path,
                gage_ids=['gage1']
            )
            self.assertEqual(result['comid'].iloc[0], '111')

class TestFindFeatSrceId(unittest.TestCase):

    def test_find_feat_srce_id(self):
        attr_config = {'col_schema': [{'featureID': 'USGS-{gage_id}'},
                        {'featureSource': 'nwissite'}],
                        'loc_id_read': [{'gage_id': 'gage_id'},
                        {'loc_id_filepath': '{dir_std_base}/juliemai-xSSA/eval/metrics/juliemai-xSSA_Raven_blended.csv'},
                        {'featureID_loc': 'USGS-{gage_id}'},
                        {'featureSource_loc': 'nwissite'}],
                        }
        rslt = raftsutil._find_feat_srce_id(attr_config = attr_config)
        self.assertEqual(rslt,['nwissite','USGS-{gage_id}'])
        print("✅ _find_feat_srce_id standard test passed.")
    # Raise error when featureSource not provided:
    def test_missing_feat_srce(self):
        attr_config_miss = {'col_schema': [{'featureID': 'USGS-{gage_id}'},
                                            {'fe0a0ur0eS0ou0r0ce': 'nwissite'}],
                            'loc_id_read': {'gage_id': 'gage_id'}}
        with self.assertRaises(ValueError):
            raftsutil._find_feat_srce_id(attr_config = attr_config_miss, dat_resp=None)
        print("✅ _find_feat_srce_id missing test passed.")
    def test_netcdf_attributes(self):
         # Create a mock xarray.Dataset object w/ attributes
        mock_xr = MagicMock(spec=xr.Dataset)
        mock_xr.attrs = {'featureSource': 'nwissite',
                         'featureID': 'USGS-{gage_id}'}

        rslt = raftsutil._find_feat_srce_id(mock_xr)
        self.assertEqual(rslt,['nwissite','USGS-{gage_id}'])
        print("✅ _find_feat_srce_id mock xarray test passed.")

    # Raise error when featureID not provided:
    def test_missing_feat_id(self):
         # Create a mock xarray.Dataset object
        mock_xr = MagicMock(spec=xr.Dataset)
        mock_xr.attrs = {'featureSource': 'nwissite',
                         'f0e1a0tu1reID': 'USGS-{gage_id}'}

        with self.assertRaises(ValueError):
            raftsutil._find_feat_srce_id(mock_xr)
        print("✅ _find_feat_srce_id mock xarray missing featureID test passed.")
    # Test when dataset does not have any attributes but does have config:
    def test_missing_attrs(self):
        mock_xr = MagicMock(spec=xr.Dataset)
        mock_xr.attrs = {'notit': 'blah',
                         'alsonotit': 'bleh'}
        attr_config = {'col_schema': [{'featureID': 'USGS-{gage_id}'},
                {'featureSource': 'nwissite'}],
                'loc_id_read': [{'gage_id': 'gage_id'},
                {'loc_id_filepath': '{dir_std_base}/juliemai-xSSA/eval/metrics/juliemai-xSSA_Raven_blended.csv'},
                {'featureID_loc': 'USGS-{gage_id}'},
                {'featureSource_loc': 'nwissite'}],
                }
        rslt = raftsutil._find_feat_srce_id(dat_resp = mock_xr, attr_config = attr_config)
        self.assertEqual(rslt,['nwissite','USGS-{gage_id}'])
        print("✅ _find_feat_srce_id missing attributes test passed.")

class build_cfig_path(unittest.TestCase):
    def test_build_cfig_path(self):
        dir_base = tempfile.gettempdir()
        dir_new = Path(dir_base)/Path('testingitout')
        dir_new.mkdir(exist_ok=True)
        with self.assertRaises(FileNotFoundError):
            path_cfig = raftsutil.build_cfig_path(dir_new,'test.yaml')

        with self.assertRaises(FileNotFoundError):
            raftsutil.build_cfig_path('this_dir/doesnt/exist','test.yaml')

        with self.assertLogs(level="INFO") as cm:
            raftsutil.build_cfig_path(dir_new,'')
        self.assertTrue(any("configuration file may not have specified the path or file name." in m for m in cm.output))
        print("✅ test_build_cfig_path build config paths test passed.")

    @patch('pathlib.Path.exists')
    def test_file_exists(self, mock_exists):
        dir_base = tempfile.gettempdir()
        dir_new = Path(dir_base)/Path('testingitout')
        path_known_config = Path(dir_new)/Path('test.yaml')
        path_or_name_cfig = Path(dir_new)/Path('a_nother_config.yaml')

        # Mock the existence of the directories and files, side_effect attr of mock object allws specifying a function or iterable called e/ time mock is called
        mock_exists.side_effect = lambda: True # Tells the mock object to return True everytime the path.exists method called
        rslt = raftsutil.build_cfig_path(path_known_config, path_or_name_cfig)
        # Assert
        self.assertEqual(rslt, path_or_name_cfig)
        self.assertEqual(mock_exists.call_count, 2)
        print("✅ build_cfig_path build config paths test with mock paths passed.")

class TestRaftsSaveAlgoDirStruct(unittest.TestCase):
    def test_rafts_save_algo_dir_struct(self):
        dir_base = tempfile.gettempdir()
        rslt = raftsutil.rafts_save_algo_dir_struct(dir_base)
        self.assertIn('dir_out', rslt.keys())
        self.assertIn('dir_out_alg_base', rslt.keys())
        self.assertTrue(Path(rslt['dir_out_alg_base']).exists)

        with self.assertRaises(ValueError):
            raftsutil.rafts_save_algo_dir_struct(dir_base + '/not_a_dir/')
        print("✅ rafts_save_algo_dir_struct creating directory structure for outputs passed.")

    def test_rafts_save_algo_dir_struct_region_aware(self):
        dir_base = tempfile.gettempdir()
        rslt_unregioned = raftsutil.rafts_save_algo_dir_struct(dir_base)

        # A region/scheme is only honored when BOTH are given -- a region with no scheme (or
        # vice versa) must stay byte-identical to the unregioned call, never partially nest.
        rslt_region_only = raftsutil.rafts_save_algo_dir_struct(dir_base, region='FL')
        rslt_scheme_only = raftsutil.rafts_save_algo_dir_struct(dir_base, scheme='custom')
        self.assertEqual(rslt_region_only['dir_out'], rslt_unregioned['dir_out'])
        self.assertEqual(rslt_scheme_only['dir_out'], rslt_unregioned['dir_out'])

        rslt_regioned = raftsutil.rafts_save_algo_dir_struct(dir_base, region='FL', scheme='custom')
        expected_dir_out = Path(dir_base) / 'output' / 'regions' / 'custom' / 'FL'
        self.assertEqual(rslt_regioned['dir_out'], expected_dir_out)
        self.assertEqual(rslt_regioned['dir_out_alg_base'], expected_dir_out / 'trained_algorithms')
        self.assertTrue(expected_dir_out.exists())
        print("✅ rafts_save_algo_dir_struct region-aware nesting test passed.")

class TestOpenResponseDataRafts(unittest.TestCase):
    dir_std_base = tempfile.gettempdir()

    def test_open_response_data_rafts(self):

        with self.assertRaisesRegex(ValueError, 'Could not identify an approach to read in dataset'):
            raftsutil._open_response_data_rafts(self.dir_std_base,ds='not_a_ds')
        print("✅ _open_response_data_rafts unable to read dataset test passed.")
#%% ALGO TRAIN & EVAL
class TestStdAlgoPath(unittest.TestCase):

    @patch('pathlib.Path.mkdir')
    @patch('pathlib.Path.exists')
    def test_std_algo_path(self, mock_exists, mock_mkdir):
        dir_out_alg_ds = tempfile.gettempdir()
        algo = 'test_algo'
        metric = 'test_metric'
        dataset_id = 'test_dataset'
        expected_path = Path(dir_out_alg_ds) / 'algo_test_algo_test_metric__test_dataset.joblib'

        # Mock the existence of the directory
        mock_exists.return_value = True

        result = raftsutil.std_algo_path(dir_out_alg_ds, algo, metric, dataset_id)
        mock_mkdir.assert_called_once_with(exist_ok=True, parents=True)
        self.assertEqual(result, expected_path)

class TestStdPredPath(unittest.TestCase):
    @patch('pathlib.Path.mkdir')
    @patch('pathlib.Path.exists')
    def test_std_pred_path(self, mock_exists, mock_mkdir):
        dir_out = '/some/directory'
        algo = 'test_algo'
        metric = 'test_metric'
        dataset_id = 'test_dataset'
        expected_path = Path(dir_out) / 'algorithm_predictions' / dataset_id / 'pred_test_algo_test_metric__test_dataset.parquet'

        # Mock the existence of the directory
        mock_exists.return_value = True
        result = raftsutil.std_pred_path(dir_out, algo, metric, dataset_id)

        mock_mkdir.assert_called_once_with(exist_ok=True, parents=True)
        self.assertEqual(result, expected_path)

class TestReadPredComid(unittest.TestCase):
    @patch('pathlib.Path.exists')
    @patch('pandas.read_csv')
    def test_read_pred_comid(self, mock_read_csv, mock_exists):
        # Arrange
        path_pred_locs = '/some/directory/predictions.csv'
        comid_pred_col = 'comid'
        mock_exists.return_value = True
        mock_read_csv.return_value = pd.DataFrame({comid_pred_col: [1, 2, 3]})

        result = raftsutil._read_pred_comid(path_pred_locs, comid_pred_col)
        mock_exists.assert_called_once_with()
        mock_read_csv.assert_called_once_with(path_pred_locs)
        self.assertEqual(result, ['1', '2', '3'])

    @patch('pathlib.Path.exists')
    @patch('pandas.read_csv')
    def test_read_csv_error(self, mock_read_csv, mock_exists):
        path_pred_locs = '/some/directory/predictions.csv'
        comid_pred_col = 'comid'
        mock_exists.return_value = True
        mock_read_csv.side_effect = Exception("Read CSV error")
        with self.assertRaises(ValueError):
            raftsutil._read_pred_comid(path_pred_locs, comid_pred_col)

    @patch('pathlib.Path.exists')
    def test_unsupported_file_extension(self, mock_exists):
        path_pred_locs = '/some/directory/predictions.txt'
        comid_pred_col = 'comid'
        mock_exists.return_value = True
        with self.assertRaises(ValueError):
            raftsutil._read_pred_comid(path_pred_locs, comid_pred_col)

    def test_read_pred_comid_raises_file_not_found_for_real_missing_path(self):
        """The .exists() check itself (previously always mocked True above) --
        a genuinely missing path should raise FileNotFoundError, not ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path_pred_locs = Path(tmpdir) / "does_not_exist.csv"
            with self.assertRaises(FileNotFoundError):
                raftsutil._read_pred_comid(path_pred_locs, 'comid')

    def test_read_pred_comid_parquet_file(self):
        """.parquet branch, previously untested in either direction."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path_pred_locs = Path(tmpdir) / "predictions.parquet"
            pd.DataFrame({'comid': [1, 2, 2, 3]}).to_parquet(path_pred_locs)

            result = raftsutil._read_pred_comid(path_pred_locs, 'comid')
            self.assertEqual(result, ['1', '2', '3'])  # drop_duplicates preserves order

    def test_read_pred_comid_directory_of_parquet(self):
        """Directory-of-parquet branch, previously untested in either direction."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dir_pred_locs = Path(tmpdir)
            pd.DataFrame({'comid': [10, 20]}).to_parquet(dir_pred_locs / "part.parquet")

            result = raftsutil._read_pred_comid(dir_pred_locs, 'comid')
            self.assertEqual(sorted(result), ['10', '20'])


# %% UNIT TEST FOR AlgoTrainEval class
class TestAlgoTrainEval(unittest.TestCase):
    print("Testing AlgoTrainEval")
    def setUp(self):
        # Create a simple DataFrame for testing
        data = {
            'comid': [f'id_{i}' for i in range(20)],
            'attr1': list(range(1, 21)),
            'attr2': list(range(21, 1, -1)),
            'metric1': [0.1, 0.9, 0.3, 0.1, 0.8, 0.2, 0.7, 0.5, 0.3, 0.6,
                        0.2, 0.2, 0.3, 0.6, 0.5, 0.2, 0.7, 0.9, 0.3, 0.1]
        }
        self.df = pd.DataFrame(data)

        # Variables and configurations for algorithms
        self.attrs = ['attr1', 'attr2']
        self.algo_config = {
            'rf': {'n_estimators': 10},
            'mlp': {'hidden_layer_sizes': (10,), 'max_iter': 2000}
        }
        # self.bagging_ci_params = {'n_algos': 5}  # Example parameters
        self.dataset_id = 'test_dataset'
        self.metric = 'metric1'
        self.verbose = False
        # Output directory
        self.dir_out_alg_ds = tempfile.gettempdir()
        self.confidence_levels = [90, 95]  # Example parameters
        # self.mapie_alpha = [0.1, 0.2]
        uncertainty_cfg = {
            'forestci': [{'fci_flag': True}],
            'bagging': [{'n_algos': 10}],
            'mapie': [{
                'alpha': [0.05, 0.32],
                'method': 'plus',
                'cv': 10,
                'agg_function': 'median'
            }]
        }
        
        # Instantiate AlgoTrainEval class
        self.train_eval = raftsalgo.AlgoTrainEval(df=self.df, attrs=self.attrs, 
                                        algo_config=self.algo_config,
                                        uncertainty=uncertainty_cfg,
                                 dir_out_alg_ds=self.dir_out_alg_ds, dataset_id=self.dataset_id,
                                 metr=self.metric, test_size=0.4, rs=42,
                                 test_id_col='comid',
                                 confidence_levels = self.confidence_levels,
                                 )

    def test_split_data(self):
        # Test data splitting
        self.train_eval.split_data()
        self.assertEqual(len(self.train_eval.X_train), 12)
        self.assertEqual(len(self.train_eval.X_test), 8)
        self.assertEqual(len(self.train_eval.y_train), 12)
        self.assertEqual(len(self.train_eval.y_test), 8)

    def test_train_algos(self):
        # Test algorithm training
        self.train_eval.split_data()
        self.train_eval.train_algos()

        self.assertIn('rf', self.train_eval.algs_dict)
        self.assertIsInstance(self.train_eval.algs_dict['rf']['algo'], RandomForestRegressor)

        self.assertIn('mlp', self.train_eval.algs_dict)
        self.assertIsInstance(self.train_eval.algs_dict['mlp']['algo'], MLPRegressor)

    def test_predict_algos(self):
        # Test algorithm predictions
        self.train_eval.split_data()
        self.train_eval.train_algos()

        preds = self.train_eval.predict_algos()

        self.assertIn('rf', preds)
        self.assertIn('mlp', preds)
        self.assertEqual(len(preds['rf']['y_pred']), len(self.train_eval.X_test))  # Number of test samples
        self.assertEqual(len(preds['mlp']['y_pred']), len(self.train_eval.X_test))

    def test_evaluate_algos(self):
        # Test evaluation of algorithms
        self.train_eval.split_data()
        self.train_eval.train_algos()
        self.train_eval.predict_algos()

        eval_dict = self.train_eval.evaluate_algos()

        self.assertIn('rf', eval_dict)
        self.assertIn('mlp', eval_dict)
        self.assertIn('mse', eval_dict['rf'])
        self.assertIn('r2', eval_dict['mlp'])

    @patch('joblib.dump')
    def test_save_algos(self, mock_dump):
        # Test saving algorithms to disk
        self.train_eval.split_data()
        self.train_eval.train_algos()

        # Mock joblib.dump to avoid file operations
        self.train_eval.save_algos()
        self.assertTrue(mock_dump.called)

        for algo in self.train_eval.algs_dict.keys():
            self.assertIn('file_pipe', self.train_eval.algs_dict[algo])

    def test_org_metadata_alg(self):
        # Test organizing metadata
        self.train_eval.split_data()
        self.train_eval.train_algos()
        self.train_eval.predict_algos()
        self.train_eval.evaluate_algos()

        # Mock saving algorithms and call organization
        with patch('joblib.dump'):
            self.train_eval.save_algos()

        self.train_eval.org_metadata_alg()

        # Check eval_df is correctly populated
        self.assertFalse(self.train_eval.eval_df.empty)
        self.assertIn('dataset', self.train_eval.eval_df.columns)
        self.assertIn('file_pipe', self.train_eval.eval_df.columns)
        self.assertIn('algo', self.train_eval.eval_df.columns)
        self.assertEqual(self.train_eval.eval_df['dataset'].iloc[0], self.dataset_id)

    def test_calculate_forestci_uncertainty(self):
        # Test the calculate_forestci_uncertainty method
        self.train_eval.split_data()
        self.train_eval.train_algos()

        rf = self.train_eval.algs_dict['rf']['algo']
        ci_dict = self.train_eval.calculate_forestci_uncertainty(rf, self.train_eval.X_train, self.train_eval.X_test)

        self.assertIn('ci_95', ci_dict)  # Check for 95% confidence interval
        self.assertIn('lower_bound', ci_dict['ci_95'])
        self.assertIn('upper_bound', ci_dict['ci_95'])
        self.assertEqual(len(ci_dict['ci_95']['lower_bound']), len(self.train_eval.X_test))
        self.assertEqual(len(ci_dict['ci_95']['upper_bound']), len(self.train_eval.X_test))

    def test_calculate_bagging_ci(self):
        # Test the calculate_bagging_ci method
        self.train_eval.split_data()
        self.train_eval.train_algos()

        best_algo = self.train_eval.algs_dict['rf']['algo']  # Use the trained RF model
        self.train_eval.calculate_bagging_ci('rf', best_algo)

        # Check if uncertainty data is stored
        self.assertIn('Uncertainty', self.train_eval.algs_dict['rf'])
        self.assertIn('bagging_mean_pred', self.train_eval.algs_dict['rf']['Uncertainty'])
        self.assertIn('bagging_std_pred', self.train_eval.algs_dict['rf']['Uncertainty'])
        self.assertIn('bagging_confidence_intervals', self.train_eval.algs_dict['rf']['Uncertainty'])
        
        # Check confidence intervals
        ci = self.train_eval.algs_dict['rf']['Uncertainty']['bagging_confidence_intervals']
        self.assertIn('confidence_level_90', ci)
        self.assertIn('confidence_level_95', ci)
        
        self.assertEqual(len(ci['confidence_level_90']['lower_bound']), len(self.train_eval.X_test))
        self.assertEqual(len(ci['confidence_level_90']['upper_bound']), len(self.train_eval.X_test))
        self.assertEqual(len(ci['confidence_level_95']['lower_bound']), len(self.train_eval.X_test))
        self.assertEqual(len(ci['confidence_level_95']['upper_bound']), len(self.train_eval.X_test))

    def test_calculate_mapie(self):
        """Test that calculate_mapie correctly fits MapieRegressor and stores it in algs_dict."""
        self.train_eval.split_data()
        self.train_eval.train_algos()

        self.train_eval.calculate_mapie()
        
        # Debugging: print algs_dict to check its structure
        print("algs_dict content:", self.train_eval.algs_dict)
    
        # Ensure 'rf' exists in algs_dict before accessing it
        self.assertIn('rf', self.train_eval.algs_dict, "Key 'rf' not found in algs_dict")
    
        # Ensure 'mapie' is stored correctly under 'rf'
        self.assertIn('mapie', self.train_eval.algs_dict['rf'], "Key 'mapie' not found under 'rf' in algs_dict")
        self.assertIsInstance(self.train_eval.algs_dict['rf']['mapie'], MapieRegressor)
        
        # Check that MapieRegressor is fitted
        mapie = self.train_eval.algs_dict['rf']['mapie']

class TestAlgoTrainEvalMlti(unittest.TestCase):

    def setUp(self):
        # Sample data for testing
        data = {
            'comid':[f'id_{i}' for i in range(15)],
            'attr1': [1, 2, 3, 4, 5,1, 2, 3, 4, 5,1, 2, 3, 4, 5],
            'attr2': [5, 4, 3, 2, 1,5, 4, 3, 2, 1,5, 4, 3, 2, 1],
            'metric': [0.1, 0.9, 0.3, 0.1, 0.8,0.1, 0.9, 0.3, 0.1, 0.8,0.1, 0.9, 0.3, 0.1, 0.8]
        }
        self.df = pd.DataFrame(data)
        self.attrs = ['attr1', 'attr2']
        self.algo_config = {
            'rf': [{'n_estimators': [10, 50]}],
            'mlp': [{'hidden_layer_sizes': [(10,), (5, 5)], 'max_iter': [2000]}]
        }
        self.dir_out_alg_ds = './'
        self.dataset_id = 'test_dataset'
        self.metric = 'metric'
        self.test_size = 0.3
        self.test_id_col = 'comid'
        self.rs = 32
        self.verbose = False

        self.confidence_levels = [90, 95]  # Example parameters
        uncertainty_cfg = {
            'forestci': [{'fci_flag': True}],
            'bagging': [{'n_algos': 10}],
            'mapie': [{
                'alpha': [0.05, 0.32],
                'method': 'plus',
                'cv': 10,
                'agg_function': 'median'
            }]
        }

        self.algo_train_eval = raftsalgo.AlgoTrainEval(df=self.df, attrs=self.attrs, algo_config=self.algo_config,
                                             uncertainty=uncertainty_cfg,
                                              dir_out_alg_ds=self.dir_out_alg_ds,dataset_id=self.dataset_id,
                                              metr=self.metric, test_size=self.test_size, rs=self.rs,
                                              test_id_col=self.test_id_col,
                                              verbose=self.verbose,
                                              confidence_levels = self.confidence_levels,
                                              )
    def test_initialization(self):
        self.assertEqual(self.algo_train_eval.df.shape, self.df.shape)
        self.assertEqual(self.algo_train_eval.attrs, self.attrs)
        self.assertEqual(self.algo_train_eval.algo_config, self.algo_config)
        self.assertEqual(self.algo_train_eval.dir_out_alg_ds, self.dir_out_alg_ds)
        self.assertEqual(self.algo_train_eval.metric, self.metric)
        self.assertEqual(self.algo_train_eval.test_size, self.test_size)
        self.assertEqual(self.algo_train_eval.rs, self.rs)
        self.assertEqual(self.algo_train_eval.dataset_id, self.dataset_id)
        self.assertEqual(self.algo_train_eval.verbose, self.verbose)

    def test_split_data(self):
        self.algo_train_eval.split_data()
        self.assertFalse(self.algo_train_eval.X_train.empty)
        self.assertFalse(self.algo_train_eval.X_test.empty)
        self.assertFalse(self.algo_train_eval.y_train.empty)
        self.assertFalse(self.algo_train_eval.y_test.empty)
        self.assertEqual(len(self.algo_train_eval.X_train) + len(self.algo_train_eval.X_test), len(self.df.dropna()))

    def test_select_algs_grid_search(self):
        self.algo_train_eval.select_algs_grid_search()
        self.assertIn('mlp', self.algo_train_eval.grid_search_algs)
        self.assertNotIn('mlp', self.algo_train_eval.algo_config)
        self.assertIn('mlp', self.algo_train_eval.algo_config_grid)


    def test_train_algos(self):
        self.algo_train_eval.split_data()
        self.algo_train_eval.select_algs_grid_search()
        self.algo_train_eval.train_algos_grid_search()
        self.assertTrue('rf' in  self.algo_train_eval.algo_config_grid)
        self.assertIn('mlp', self.algo_train_eval.algs_dict)

    def test_empty_dict(self):
        d = {}
        self.algo_train_eval.convert_to_list(d)
        self.assertEqual(d, {})

    def test_single_level_dict(self):
        d = {'a': 1, 'b': 2}
        self.algo_train_eval.convert_to_list(d)
        self.assertEqual(d, {'a': [1], 'b': [2]})

    def test_nested_dict(self):
        d = {'a': {'sub1': 1, 'sub2': 2}, 'b': {'sub1': 3, 'sub2': {'subsub1': 4}}, 'c': 5}
        self.algo_train_eval.convert_to_list(d)
        self.assertEqual(d, {'a': {'sub1': [1], 'sub2': [2]}, 'b': {'sub1': [3], 'sub2': {'subsub1': [4]}}, 'c': [5]})

    def test_already_list(self):
        d = {'a': [1, 2], 'b': {'sub1': [3, 4]}}
        self.algo_train_eval.convert_to_list(d)
        self.assertEqual(d, {'a': [1, 2], 'b': {'sub1': [3, 4]}})

        
class TestAlgoTrainEvalSngl(unittest.TestCase):
    # An algo_config with singular hyperparameter value
    def setUp(self):
        # Sample data for testing
        self.df = pd.DataFrame({
            'comid': [f'id_{i}' for i in range(5)],
            'attr1': [1, 2, 3, 4, 5],
            'attr2': [5, 4, 3, 2, 1],
            'metric': [1, 0, 1, 0, 1]
        })
        self.attrs = ['attr1', 'attr2']
        self.algo_config = {'mlp': {'max_iter': [100]}}
        uncertainty_cfg = {
            'forestci': [{'fci_flag': True}],
            'bagging': [{'n_algos': 10}],
            'mapie': [{
                'alpha': [0.32],
                'method': 'plus',
                'cv': 10,
                'agg_function': 'median'
            }]
        }
        self.dir_out_alg_ds = tempfile.gettempdir()
        self.dataset_id = 'dataset_1'
        self.metr = 'metric'
        self.test_size = 0.3
        self.rs = 32
        self.verbose = False
        self.algo_config_grid = dict()
        self.grid_search_algs=list()

        self.algo_train_eval = raftsalgo.AlgoTrainEval(
            df=self.df, attrs=self.attrs, algo_config=self.algo_config, 
            uncertainty=uncertainty_cfg,
            dir_out_alg_ds=self.dir_out_alg_ds,
            dataset_id=self.dataset_id, metr=self.metr, test_size=self.test_size, 
            rs=self.rs, verbose=self.verbose,
            test_id_col='comid'
        )

    @patch('rafts_algo.rafts_algo_train.joblib.dump')
    @patch.object(raftsalgo.AlgoTrainEval, 'calculate_bagging_ci')
    @patch.object(raftsalgo.AlgoTrainEval, 'calculate_mapie')
    @patch.object(raftsalgo.AlgoTrainEval, 'split_data')
    @patch.object(raftsalgo.AlgoTrainEval, 'select_algs_grid_search')
    @patch.object(raftsalgo.AlgoTrainEval, 'train_algos_grid_search')
    @patch.object(raftsalgo.AlgoTrainEval, 'train_algos')
    def test_train_eval(self, mock_train_algos, mock_train_algos_grid_search,
                        mock_select_algs_grid_search, mock_split_data,
                        mock_calc_bagging_ci,mock_calc_mapie, mock_dump):
        # Mock the methods to avoid actual execution
        mock_split_data.return_value = None
        mock_select_algs_grid_search.return_value = None
        mock_train_algos_grid_search.return_value = None
        mock_train_algos.return_value = None
        mock_calc_bagging_ci.return_value = None
        mock_calc_mapie.return_value = None

        # Mock pipeline with a .predict() method
        mock_pipeline = MagicMock()
        test_indices = [1, 4]  # Example indices from the original df
    
        # Create a realistic X_test DataFrame with a valid index
        self.algo_train_eval.X_test = self.algo_train_eval.df.loc[test_indices, self.algo_train_eval.attrs]
        self.algo_train_eval.y_test = self.algo_train_eval.df.loc[test_indices, self.algo_train_eval.metric]
    
        # Update mock prediction length to match the new X_test size
        mock_pipeline.predict.return_value = np.random.rand(len(test_indices))
        self.algo_train_eval.preds_dict = {}
        self.algo_train_eval.uncertainty = {
            'forestci': [{'fci_flag': True}],
            'bagging': [{'n_algos': 10}],
            'mapie': [{
                'alpha': [0.32],
                'method': 'plus',
                'cv': 10,
                'agg_function': 'median'
            }]
        }   
        
        # Setup algs_dict with required keys
        self.algo_train_eval.algs_dict = {
            'mlp': {
                'algo': MLPRegressor(),
                'pipeline': mock_pipeline,
                'type': 'MLP',
                'metric': 'metric',
                'Uncertainty': {}
            }
        }

        # Call the method
        self.algo_train_eval.train_eval()

        # Assert that the methods were or were not called
        mock_split_data.assert_called_once()
        mock_select_algs_grid_search.assert_called_once()
        mock_train_algos_grid_search.assert_not_called()
        mock_train_algos.assert_called_once()
        mock_calc_bagging_ci.assert_called_once()
        mock_calc_mapie.assert_called_once()

class TestAlgoTrainEvalBasic(unittest.TestCase):

    def setUp(self):
        # Set up a small test dataframe
        self.df = pd.DataFrame({
            'comid': [f'id_{i}' for i in range(15)],
            'attr1': [1, 2, 3, 4, 5,1, 2, 3, 4, 5,1, 2, 3, 4, 5],
            'attr2': [5, 4, 3, 2, 1,5, 4, 3, 2, 1,5, 4, 3, 2, 1],
            'target': [10, 15, 20, 25, 30,10, 15, 20, 25, 30,10, 15, 20, 25, 30]
        })
        self.attrs = ['attr1', 'attr2']
        self.algo_config = {
            'rf': [{'n_estimators': [10,30,40]}],
            'mlp': [{'hidden_layer_sizes': (50,)}]
        }

        # mkdtemp (not TemporaryDirectory's `with`) so the directory survives past
        # setUp() for the duration of each test method; a `with` block here exits
        # (and deletes the directory) before self.dir_out_alg_ds is ever used,
        # which previously left every test in this class pointed at a directory
        # that no longer existed on disk -- tests only passed because
        # utils.std_algo_path() happens to recreate it via mkdir(parents=True)
        # as an incidental side effect before writing.
        temp_dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, temp_dir, ignore_errors=True)

        self.dir_out_alg_ds = temp_dir
        self.dataset_id = 'test_ds'
        self.metric = 'target'
        self.test_size = 0.2
        self.rs = 42
        self.verbose = False
        self.algo_config_grid = dict()

        self.confidence_levels = [90, 95]  # Example parameters
        uncertainty_cfg = {
            'forestci': [{'fci_flag': True}],
            'bagging': [{'n_algos': 10}],
            'mapie': [{
                'alpha': [0.2, 0.32],
                'method': 'plus',
                'cv': 10,
                'agg_function': 'median'
            }]
        }
        
        self.algo = raftsalgo.AlgoTrainEval(df=self.df, attrs=self.attrs, algo_config=self.algo_config,
                                  uncertainty=uncertainty_cfg,
                                  dir_out_alg_ds=self.dir_out_alg_ds, dataset_id=self.dataset_id, 
                                  metr=self.metric, test_size=self.test_size, rs=self.rs, 
                                  test_id_col='comid',
                                  verbose=self.verbose,
                                  confidence_levels = self.confidence_levels,
                                  )

    @patch('joblib.dump')  # Avoid writing the trained model to disk; joblib is
                           # imported as a module in rafts_algo_train.py (`import
                           # joblib`), so patching it here does take effect --
                           # unlike the sklearn mocks previously attempted below.
    def test_train_eval(self, mock_joblib_dump):
        self.algo.train_eval()

        # Check predictions and evaluations were made
        self.assertIn('rf', self.algo.preds_dict)
        self.assertIn('mlp', self.algo.preds_dict)

        self.assertIn('rf', self.algo.eval_dict)
        self.assertIn('mlp', self.algo.eval_dict)

        # Check if models were saved
        mock_joblib_dump.assert_called()

        # Check eval dataframe was created
        self.assertIsInstance(self.algo.eval_df, pd.DataFrame)
        self.assertFalse(self.algo.eval_df.empty)
        
    def test_learning_curve_plotting(self):
        """Test the learning curve generation and plotting logic without mocking I/O."""
        self.algo.train_eval()
        
        # 1. Grab the trained Random Forest pipeline and data
        pipe_rf = self.algo.algs_dict['rf']['pipeline']
        df_X, y_all = self.algo.all_X_all_y()
        
        # 2. Instantiate the Plotting Object
        plot_obj = raftsalgo.AlgoEvalPlotLC(df_X, y_all)
        
        # 3. Generate the data (cv=2 to make it run fast on our tiny dummy dataset)
        plot_obj.gen_learning_curve(model=pipe_rf, cv=2, n_jobs=1)
        self.assertTrue(hasattr(plot_obj, 'train_mean_lc')) # Verifies calculation succeeded
        
        # 4. Test the wrapper which physically saves the PNG
        raftsalgo.plot_learning_curve_save_wrap(
            algo_plot=plot_obj,
            train_eval=self.algo,
            dir_out_viz_base=self.dir_out_alg_ds, # Use temp dir
            ds=self.dataset_id,
            cv=2,
            n_jobs=1
        )
        
        # Verify the .png file was generated and saved to the temp directory
        expected_png = self.dir_out_alg_ds / self.dataset_id / f"learning_curve_{self.dataset_id}_{self.metric}_rf.png"
        self.assertTrue(expected_png.exists(), "Learning curve PNG was not saved!")

    def test_learning_curve_plotting_with_training_uncertainty_band(self):
        """training_uncn=True adds a shaded training-uncertainty band -- previously untested."""
        self.algo.train_eval()
        pipe_rf = self.algo.algs_dict['rf']['pipeline']
        df_X, y_all = self.algo.all_X_all_y()
        plot_obj = raftsalgo.AlgoEvalPlotLC(df_X, y_all)
        plot_obj.gen_learning_curve(model=pipe_rf, cv=2, n_jobs=1)
        fig = plot_obj.plot_learning_curve(training_uncn=True)
        self.assertIsNotNone(fig)
# %%

class TestReadMetadata(unittest.TestCase):
    def test_read_metadata_returns_none_when_file_missing(self):
        """_read_metadata logs a warning and returns None (rather than raising)
        when the resolved metadata path doesn't exist.

        Update 2026-04-24: changed to return None to accommodate param
        regionalization ignoring attr_config.yaml -- this test previously
        asserted the same result but was still named/documented for the old
        FileNotFoundError-raising behavior, and carried an unused
        @patch('pandas.read_parquet') left over from before that change
        (the missing-file branch returns before pd.read_parquet is ever
        called, so the patch had nothing to intercept).
        """
        path_attr_config = Path(dir_test_data, "attr_config.yaml")
        result = raftsutil._read_metadata(path_attr_config, ds='dataset_name')
        self.assertIsNone(result)

    @patch('pandas.read_parquet')
    def test_read_metadata_reads_parquet(self, mock_read_parquet):
        """Test reading metadata when parquet file is present and correct."""
        # Arrange
        mock_df = pd.DataFrame({'gage_id': [12345, 25432], 'featureID': ['prvi-cat-135', 'ak-cat-123']})
        mock_read_parquet.return_value = mock_df

        # Use real attr_config.yaml
        path_attr_config = Path(dir_test_data, "attr_config.yaml")
        attr_cfig = raftsutil.AttrConfigAndVars(path_attr_config)
        
        # Patch AttrConfigAndVars and mock its instance
        with patch("rafts_algo.utils.AttrConfigAndVars") as MockAttrClass:
            mock_instance = MockAttrClass.return_value
            mock_instance._read_attr_config.return_value = None

            mock_instance.attr_config = {
                'file_io': [
                    {'ds_type': 'training'},
                    {'write_type': 'parquet'},
                    {'path_meta': "{dir_std_base}/{ds}/{ds}_{ds_type}.parquet"}
                ]
            }
            # Only override dir_std_base key
            mock_instance.attrs_cfg_dict = {'dir_std_base': dir_test_data}

            # Act
            result = raftsutil._read_metadata(path_attr_config, ds="dataset_name")

            # Assert
            mock_read_parquet.assert_called_once()
            pd.testing.assert_frame_equal(result, mock_df)


# %% Creating Unit tests for PredConfigParser
class TestPredConfigParser(unittest.TestCase):

    def setUp(self):
        # Create temporary directory for config files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_path = Path(self.temp_dir.name)

        # Create dummy dir_base and dir_std_base
        self.dir_base = self.test_path / "base"
        self.dir_base.mkdir()
        self.dir_std_base = self.dir_base / "std"
        self.dir_std_base.mkdir()

        # Also create dir_db_attrs if referenced
        self.dir_db_attrs = self.dir_base / "db_attrs"
        self.dir_db_attrs.mkdir()

        # Attribute config file
        self.attr_config = {
            'file_io': [
                {'home_dir': str(self.test_path)},
                {'dir_base': str(self.dir_base)},
                {'dir_std_base': str(self.dir_std_base)},
                {'dir_db_attrs': str(self.dir_db_attrs)}
            ],
            'formulation_metadata': [
                {'datasets': ['test_dataset']}
            ],
            'attr_select': [
                {'static_vars': ['slope', 'elevation']}
            ]
        }
        
        self.path_attr_config = self.test_path / "attr_config.yaml"
        with open(self.path_attr_config, 'w') as f:
            yaml.dump(self.attr_config, f)

        # Prediction config file
        self.pred_config = {
            "name_attr_config": self.path_attr_config.name,
            "name_algo_config": "algo.yaml",
            "name_tfrm_config": "tfrm.yaml",
            "path_tfrm_script": "/some/script.py",
            "conda_env": "test_env",
            "ds_type": "eval",
            "write_type": "parquet",
            "path_meta": "path/meta/{ds}",
            "pred_file_comid_colname": "feature_id",
            "algo_response_vars": ["runoff"],
            "algo_type": ["rf"],
            "MAPIE_alpha": [0.1]
        }
        self.path_pred_config = self.test_path / "pred_config.yaml"
        with open(self.path_pred_config, 'w') as f:
            yaml.dump(self.pred_config, f)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_read_pred_config_success(self):
        parser = raftsutil.PredConfigParser(str(self.path_pred_config))
        parser._read_pred_config()

        self.assertEqual(parser.pred_cfg_dict['ds_type'], 'eval')
        self.assertEqual(parser.pred_cfg_dict['write_type'], 'parquet')
        self.assertEqual(parser.pred_cfg_dict['datasets'], ['test_dataset'])
        self.assertEqual(parser.pred_cfg_dict['algo_type'], ['rf'])
        self.assertEqual(parser.pred_cfg_dict['path_pred_config'], str(self.path_pred_config))
        self.assertEqual(parser.pred_cfg_dict['dir_base'], self.dir_base)
        self.assertEqual(parser.pred_cfg_dict['dir_std_base'], self.dir_std_base)
        # Not set in self.pred_config -- should default to None, not KeyError,
        # so rafts_pred_algo.py's `.get('featureSource')` fallback chain works.
        self.assertIsNone(parser.pred_cfg_dict['featureSource'])

    def test_read_pred_config_captures_featuresource(self):
        # Prediction-time featureSource is independent of (and may legitimately
        # differ from) the training-time value -- e.g. trained on gage-basin-
        # aggregated attributes, predicting at HUC10/HUC14/divide scale.
        self.pred_config['featureSource'] = 'hfv4_divides_raw'
        with open(self.path_pred_config, 'w') as f:
            yaml.dump(self.pred_config, f)

        parser = raftsutil.PredConfigParser(str(self.path_pred_config))
        parser._read_pred_config()
        self.assertEqual(parser.pred_cfg_dict['featureSource'], 'hfv4_divides_raw')

    def test_pydantic_pred_config_success(self):
        cfg = PredConfig(**self.pred_config)
        self.assertEqual(cfg.ds_type, 'eval')
        self.assertEqual(cfg.write_type, 'parquet')

    def test_read_pred_config_missing_required(self):
        # Remove a required field
        del self.pred_config["ds_type"]
        with open(self.path_pred_config, 'w') as f:
            yaml.dump(self.pred_config, f)

        parser = raftsutil.PredConfigParser(str(self.path_pred_config))
        with self.assertRaises(ValueError) as context:
            parser._read_pred_config()
        self.assertIn("Missing required keys", str(context.exception))

    def test_pydantic_pred_config_missing_required(self):
        del self.pred_config["path_meta"]
        with self.assertRaises(ValidationError) as context:
            PredConfig(**self.pred_config)
        self.assertIn("path_meta", str(context.exception))

    def test_nonexistent_pred_config_file(self):
        parser = raftsutil.PredConfigParser(str(self.test_path / "nonexistent.yaml"))
        with self.assertRaises(FileNotFoundError):
            parser._read_pred_config()

    def test_missing_dir_base_or_std(self):
        # Delete dir_base to simulate missing directory
        self.dir_std_base.rmdir()
        shutil.rmtree(self.dir_base)

        parser = raftsutil.PredConfigParser(str(self.path_pred_config))
        with self.assertRaises(FileNotFoundError) as context:
            parser._read_pred_config()
        self.assertIn("Resolved dir_base path does not exist", str(context.exception))

    def test_missing_dir_std_base_only(self):
        # dir_base exists, but dir_std_base (nested under it) does not --
        # isolates the separate dir_std_base check from test_missing_dir_base_or_std
        # above, which removes both at once via shutil.rmtree(self.dir_base).
        self.dir_std_base.rmdir()

        parser = raftsutil.PredConfigParser(str(self.path_pred_config))
        with self.assertRaises(FileNotFoundError) as context:
            parser._read_pred_config()
        self.assertIn("Resolved dir_base path does not exist", str(context.exception))
        self.assertIn(str(self.dir_std_base), str(context.exception))

    def test_relative_dir_base_and_std_base_resolved_against_attr_config(self):
        # dir_base/dir_std_base given as relative paths (as some configs do)
        # should resolve relative to the attribute config file's own
        # directory, not the process cwd.
        attr_config = {
            'file_io': [
                {'home_dir': str(self.test_path)},
                {'dir_base': 'base'},        # relative, not str(self.dir_base)
                {'dir_std_base': 'base/std'},  # relative
                {'dir_db_attrs': str(self.dir_db_attrs)},
            ],
            'formulation_metadata': [{'datasets': ['test_dataset']}],
            'attr_select': [{'static_vars': ['slope', 'elevation']}],
        }
        with open(self.path_attr_config, 'w') as f:
            yaml.dump(attr_config, f)

        parser = raftsutil.PredConfigParser(str(self.path_pred_config))
        parser._read_pred_config()
        self.assertEqual(Path(parser.pred_cfg_dict['dir_base']), self.dir_base.resolve())
        self.assertEqual(Path(parser.pred_cfg_dict['dir_std_base']), self.dir_std_base.resolve())

def test_build_pred_locs_path():
    # Given
    template = "{dir_std_base}/{ds}/pred_{ds}_{ds_type}.parquet"
    dir_std_base = "/test/standardized"
    ds = "camels"
    ds_type = "prediction"
    write_type = "parquet"

    # When
    result_path = raftsutil.build_pred_locs_path(template, dir_std_base, ds, ds_type, write_type)

    # Then
    expected = Path("/test/standardized/camels/pred_camels_prediction.parquet")
    assert result_path == expected

# %% UNIT TESTING FOR WARNING AND CLIPPING FUNCTIONS
class TestWarningAndClippingFunctions(unittest.TestCase):
    print("Testing warning and clipping helper functions")

    def setUp(self):
        """Set up common data for all test cases."""
        self.feature_ids = pd.Index(['ID_01', 'ID_02', 'ID_03', 'ID_04'])
        self.resp_var = 'test_metric'
        
        # Data for 1D prediction values
        self.y_pred = np.array([0.5, -0.2, 1.3, 0.8])
        
        # Data for 3D prediction intervals
        self.y_pis = np.array([
            [[0.1], [0.9]],  # In bounds
            [[-0.5], [0.8]], # Out of bounds (low)
            [[0.2], [1.5]],  # Out of bounds (high)
            [[-0.7], [1.7]]  # Out of bounds (both)
        ])

    # --- Tests for warn_if_out_of_bounds ---

    def test_warn_values_correction_active(self):
        """Test warning for 1D values when correction is ON."""
        with self.assertLogs(level='WARNING') as cm:  # root logger
            raftsutil._warn_if_out_of_bounds(
                self.y_pred, self.feature_ids, 0.0, 1.0, self.resp_var,
                correction_is_active=True, prediction_type="values"
            )
            self.assertEqual(len(cm.output), 1)
            self.assertIn("Post-hoc correction will be applied", cm.output[0])
            self.assertIn("['ID_02', 'ID_03']", cm.output[0])
        print("✅ test_warn_values_correction_active passed.")

    def test_warn_intervals_correction_inactive(self):
        """Test warning for 3D intervals when correction is OFF."""
        with self.assertLogs(level='WARNING') as cm:  # root logger
            raftsutil._warn_if_out_of_bounds(
                self.y_pis, self.feature_ids, 0.0, 1.0, self.resp_var,
                correction_is_active=False, prediction_type="intervals"
            )
            self.assertEqual(len(cm.output), 1)
            self.assertIn("Correction was NOT applied", cm.output[0])
            self.assertIn("['ID_02', 'ID_03', 'ID_04']", cm.output[0])
        print("✅ test_warn_intervals_correction_inactive passed.")

    def test_no_warning_when_in_bounds(self):
        """Test that no warning is logged when all values are within bounds.

        NOTE: this previously patched logging.getLogger and asserted the mock
        logger's .warning() was never called -- but _warn_if_out_of_bounds calls
        the module-level logging.warning(...), which uses the `logging` module's
        own root Logger instance directly rather than going through
        logging.getLogger(). Patching logging.getLogger never intercepted that
        call, so the assertion was checking a mock that was never wired into the
        code path at all: it would have passed even if the function warned on
        every input. assertNoLogs actually listens on the real logger.
        """
        with self.assertNoLogs(level='WARNING'):
            raftsutil._warn_if_out_of_bounds(
                np.array([0.1, 0.5, 0.9]), self.feature_ids, 0.0, 1.0, self.resp_var,
                correction_is_active=True, prediction_type="values"
            )
        print("✅ test_no_warning_when_in_bounds passed.")

    def test_warn_accepts_string_none_bounds(self):
        """min_lim/max_lim of the literal string 'none' (as YAML configs may pass
        through unparsed) should be treated the same as Python None, i.e. skip
        the bounds check entirely rather than comparing against the string."""
        with self.assertNoLogs(level='WARNING'):
            raftsutil._warn_if_out_of_bounds(
                self.y_pred, self.feature_ids, 'None', 'none', self.resp_var,
                correction_is_active=True, prediction_type="values"
            )
        print("✅ test_warn_accepts_string_none_bounds passed.")

    # --- Tests for clip_predictions (1D) ---

    def test_clip_predictions_correctly(self):
        """Test that 1D prediction values are clipped correctly."""
        expected = np.array([0.5, 0.0, 1.0, 0.8])
        result = raftsutil.clip_predictions(self.y_pred, 0.0, 1.0)
        np.testing.assert_array_equal(result, expected)
        print("✅ test_clip_predictions_correctly passed.")

    def test_clip_predictions_no_bounds(self):
        """Test that 1D predictions are unchanged if no bounds are provided."""
        result = raftsutil.clip_predictions(self.y_pred, None, None)
        np.testing.assert_array_equal(result, self.y_pred)
        print("✅ test_clip_predictions_no_bounds passed.")

    # --- Tests for clip_pis (3D) ---

    def test_clip_pis_correctly(self):
        """Test that 3D prediction intervals are clipped correctly."""
        expected = np.array([
            [[0.1], [0.9]],
            [[0.0], [0.8]],
            [[0.2], [1.0]],
            [[0.0], [1.0]]
        ])
        result = raftsutil.clip_pis(self.y_pis, 0.0, 1.0)
        np.testing.assert_array_equal(result, expected)
        print("✅ test_clip_pis_correctly passed.")

    def test_clip_pis_min_only(self):
        """Test that 3D PIs are clipped correctly with only a min bound."""
        expected = np.array([
            [[0.1], [0.9]],
            [[0.0], [0.8]],
            [[0.2], [1.5]],
            [[0.0], [1.7]]
        ])
        result = raftsutil.clip_pis(self.y_pis, 0.0, None)
        np.testing.assert_array_equal(result, expected)
        print("✅ test_clip_pis_min_only passed.")

    def test_clip_pis_no_bounds(self):
        """Test that 3D PIs are returned unchanged if no bounds are provided."""
        result = raftsutil.clip_pis(self.y_pis, None, None)
        np.testing.assert_array_equal(result, self.y_pis)
        print("✅ test_clip_pis_no_bounds passed.")

class TestAlgoTrainEvalVerboseRegressionSingleRun(unittest.TestCase):
    """
    Covers AlgoTrainEval branches that need verbose=True and/or a non-grid-search
    (single hyperparameter value) config, none of which TestAlgoTrainEvalBasic's
    verbose=False / grid-search-for-rf setup reaches: the verbose logging
    branches in split_data, train_algos (single-run rf/mlp), predict_algos,
    evaluate_algos, and save_algos; the uncn_bnd_algo prediction/interval
    clipping branches; and -- now that the 'forestfci' typo bug is fixed --
    the forestci uncertainty computation itself, via the non-grid-search
    'best_rf_algo = self.algs_dict['rf']['algo']' path.
    """

    def setUp(self):
        self.df = pd.DataFrame({
            'comid': [f'id_{i}' for i in range(15)],
            'attr1': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            'attr2': [5, 4, 3, 2, 1, 5, 4, 3, 2, 1, 5, 4, 3, 2, 1],
            'target': [10, 15, 20, 25, 30, 10, 15, 20, 25, 30, 10, 15, 20, 25, 30],
        })
        # Single values only (not lists with >1 option) -> train_algos, not
        # train_algos_grid_search.
        self.algo_config = {
            'rf': [{'n_estimators': 10}],
            'mlp': [{'hidden_layer_sizes': (5,)}],
        }
        self.uncertainty_cfg = {
            'forestci': [{'fci_flag': True}],
            'bagging': [{'n_algos': 5}],
            'mapie': [{'alpha': [0.2], 'method': 'plus', 'cv': 5, 'agg_function': 'median'}],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            self.dir_out_alg_ds = Path(tmpdir)
        # Tight bounds relative to target's [10, 30] range so uncn_bnd_algo
        # correction actually clips something for both values and intervals.
        self.algo = raftsalgo.AlgoTrainEval(
            df=self.df, attrs=['attr1', 'attr2'], algo_config=self.algo_config,
            uncertainty=self.uncertainty_cfg, dir_out_alg_ds=self.dir_out_alg_ds,
            dataset_id='verbose_test_ds', metr='target', test_size=0.3, rs=42,
            test_id_col='comid', confidence_levels=[90], verbose=True,
            uncn_bnd_algo=True, min_lim=15.0, max_lim=25.0,
        )

    def test_train_eval_runs_verbosely_with_bounds_and_forestci(self):
        with self.assertLogs(level='INFO') as cm:
            self.algo.train_eval()
        log_text = "\n".join(cm.output)

        # Verbose logging branches (split_data, train_algos, predict_algos,
        # evaluate_algos, save_algos all gate their info logs on self.verbose).
        self.assertIn("Performing train/test split", log_text)
        self.assertIn("Performing Random Forest Training", log_text)
        self.assertIn("Performing Multilayer Perceptron Training", log_text)
        self.assertIn("Generating predictions", log_text)
        self.assertIn("Evaluating predictions", log_text)

        # forestci is now actually wired up (was dead due to the 'forestfci'
        # typo) -- confirm it really ran and populated real values.
        self.assertIn('forestci', self.algo.algs_dict['rf']['Uncertainty'])
        self.assertIn('ci_90', self.algo.algs_dict['rf']['Uncertainty']['forestci'])

        # bagging still works via the non-grid-search 'algo' path.
        self.assertIn('bagging_mean_pred', self.algo.algs_dict['rf']['Uncertainty'])

    def test_uncn_bnd_algo_correction_applied(self):
        self.algo.train_eval()
        for algo_str in ('rf', 'mlp'):
            y_pred = self.algo.preds_dict[algo_str]['y_pred']
            self.assertTrue(np.all(y_pred >= 15.0) and np.all(y_pred <= 25.0))
            # MAPIE intervals should also have been clipped to the same bounds.
            for pis_df in self.algo.preds_dict[algo_str]['y_pis']:
                self.assertTrue((pis_df.values >= 15.0).all() and (pis_df.values <= 25.0).all())


class TestAlgoTrainEvalVerboseRegressionGridSearch(unittest.TestCase):
    """Covers train_algos_grid_search's verbose logging branches for rf/mlp,
    not reached by TestAlgoTrainEvalBasic (verbose=False)."""

    def setUp(self):
        self.df = pd.DataFrame({
            'comid': [f'id_{i}' for i in range(15)],
            'attr1': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            'attr2': [5, 4, 3, 2, 1, 5, 4, 3, 2, 1, 5, 4, 3, 2, 1],
            'target': [10, 15, 20, 25, 30, 10, 15, 20, 25, 30, 10, 15, 20, 25, 30],
        })
        self.algo_config = {
            'rf': [{'n_estimators': [10, 20]}],
            'mlp': [{'hidden_layer_sizes': [(5,), (5, 5)]}],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            self.dir_out_alg_ds = Path(tmpdir)
        self.algo = raftsalgo.AlgoTrainEval(
            df=self.df, attrs=['attr1', 'attr2'], algo_config=self.algo_config,
            uncertainty={}, dir_out_alg_ds=self.dir_out_alg_ds,
            dataset_id='verbose_gs_ds', metr='target', test_size=0.3, rs=42,
            test_id_col='comid', verbose=True,
        )

    def test_grid_search_logs_verbosely_for_both_algos(self):
        with self.assertLogs(level='INFO') as cm:
            self.algo.train_eval()
        log_text = "\n".join(cm.output)
        self.assertIn("Performing Random Forest Training with Grid Search", log_text)
        self.assertIn("Performing Multilayer Perceptron Training with Grid Search", log_text)
        self.assertIn('gridsearchcv', self.algo.algs_dict['rf'])
        # Note: unlike 'rf', the mlp grid-search branch doesn't store a
        # 'gridsearchcv' key -- its 'pipeline' entry *is* the GridSearchCV object.
        self.assertIsInstance(self.algo.algs_dict['mlp']['pipeline'], GridSearchCV)


class TestAlgoTrainEvalKmeansSingleRun(unittest.TestCase):
    """Covers train_algos' kmeans single-value (non-grid-search) branch,
    which TestAlgoTrainEvalClustering doesn't reach (its kmeans config uses
    two n_clusters options, routing it through train_algos_grid_search
    instead). Also exercises split_data's clustering branch and
    all_X_all_y's y=None branch, both only reachable for task_type='clustering'."""

    def setUp(self):
        self.df = pd.DataFrame({
            'comid': [f'id_{i}' for i in range(20)],
            'attr1': np.tile([1.0, 2.0, 3.0, 4.0], 5),
            'attr2': np.tile([4.0, 3.0, 2.0, 1.0], 5),
        })
        self.algo_config = {'kmeans': [{'n_clusters': [3]}]}  # single value -> train_algos
        with tempfile.TemporaryDirectory() as tmpdir:
            self.dir_out_alg_ds = Path(tmpdir)
        self.algo = raftsalgo.AlgoTrainEval(
            df=self.df, attrs=['attr1', 'attr2'], algo_config=self.algo_config,
            uncertainty={}, dir_out_alg_ds=self.dir_out_alg_ds,
            dataset_id='kmeans_single_ds', metr='cluster_labels', task_type='clustering',
            test_size=0.3, rs=42, test_id_col='comid', verbose=True,
        )

    def test_kmeans_single_run_trains_directly(self):
        with self.assertLogs(level='INFO') as cm:
            self.algo.train_eval()
        log_text = "\n".join(cm.output)
        self.assertIn("Performing clustering split", log_text)
        self.assertIn("Performing KMeans Clustering", log_text)
        self.assertIn('kmeans_k3', self.algo.algs_dict)
        self.assertNotIn('gridsearchcv', self.algo.algs_dict['kmeans_k3'])

        # all_X_all_y's y=None branch, only reachable for clustering.
        X_all, y_all = self.algo.all_X_all_y()
        self.assertIsNone(y_all)
        self.assertEqual(len(X_all), len(self.df))


class TestAlgoTrainEvalClusteringGridSearchDefaults(unittest.TestCase):
    """
    Regression test for a real bug found while increasing coverage here:
    train_algos_grid_search's clustering branch computed `opt_k` only inside
    `if self.verbose:`, then referenced it unconditionally right after --
    with the constructor's actual defaults (verbose=False,
    save_all_clusters=False), this raised UnboundLocalError for any
    clustering grid search. It also logged best_model.n_clusters directly
    instead of the opt_k it had just computed, which would additionally
    raise AttributeError for gower_agglomerative specifically (its
    UniversalDistanceClusterer wrapper has no .n_clusters of its own).
    Both are fixed by computing opt_k unconditionally and using it in the
    log message too.
    """

    def setUp(self):
        self.df = pd.DataFrame({
            'comid': [f'id_{i}' for i in range(24)],
            'attr1': np.tile([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 4),
            'attr2': np.tile([6.0, 5.0, 4.0, 3.0, 2.0, 1.0], 4),
        })
        self.algo_config = {
            'kmeans': [{'n_clusters': [2, 3]}],
            'gower_agglomerative': [{'n_clusters': [2, 3]}],
        }

    def _make_algo(self, **kwargs):
        with tempfile.TemporaryDirectory() as tmpdir:
            dir_out_alg_ds = Path(tmpdir)
        kwargs.setdefault('save_all_clusters', False)
        return raftsalgo.AlgoTrainEval(
            df=self.df, attrs=['attr1', 'attr2'], algo_config=dict(self.algo_config),
            uncertainty={}, dir_out_alg_ds=dir_out_alg_ds,
            dataset_id='cluster_defaults_ds', metr='cluster_labels', task_type='clustering',
            test_size=0.3, rs=42, test_id_col='comid', **kwargs
        )

    def test_default_verbose_and_save_all_clusters_does_not_raise(self):
        # verbose=False, save_all_clusters=False (both defaults) -- this used
        # to raise UnboundLocalError before the fix.
        algo = self._make_algo()
        algo.train_eval()
        kmeans_keys = [k for k in algo.algs_dict if k.startswith('kmeans_k')]
        gower_keys = [k for k in algo.algs_dict if k.startswith('gower_agglomerative_k')]
        self.assertEqual(len(kmeans_keys), 1, "Only the single best kmeans model should be kept")
        self.assertEqual(len(gower_keys), 1, "Only the single best gower_agglomerative model should be kept")

    def test_verbose_gower_agglomerative_does_not_raise(self):
        # verbose=True -- this used to raise AttributeError for
        # gower_agglomerative specifically (best_model.n_clusters on a
        # UniversalDistanceClusterer, which has no such attribute).
        algo = self._make_algo(verbose=True)
        with self.assertLogs(level='INFO') as cm:
            algo.train_eval()
        log_text = "\n".join(cm.output)
        self.assertIn("Optimal gower_agglomerative clusters chosen", log_text)
        self.assertIn("Optimal kmeans clusters chosen", log_text)

    def test_verbose_save_all_clusters_logs_each_saved_iteration(self):
        # save_all_clusters=True takes the "save every iteration" branch
        # instead, which has its own separate verbose log line.
        algo = self._make_algo(verbose=True, save_all_clusters=True)
        with self.assertLogs(level='INFO') as cm:
            algo.train_eval()
        log_text = "\n".join(cm.output)
        self.assertIn("Saved kmeans_k2", log_text)
        self.assertIn("Saved kmeans_k3", log_text)
        self.assertIn("Saved gower_agglomerative_k2", log_text)
        self.assertIn("Saved gower_agglomerative_k3", log_text)


class TestAlgoTrainEvalSingleClusterPredicted(unittest.TestCase):
    """Covers evaluate_algos' branch for when only one cluster label is
    predicted on the test set (silhouette_score/davies_bouldin_score are
    undefined for a single cluster) -- forced here by requesting n_clusters=1."""

    def test_single_cluster_predicted_logs_warning_and_uses_nan(self):
        df = pd.DataFrame({
            'comid': [f'id_{i}' for i in range(12)],
            'attr1': np.linspace(0, 1, 12),
            'attr2': np.linspace(1, 0, 12),
        })
        with tempfile.TemporaryDirectory() as tmpdir:
            dir_out_alg_ds = Path(tmpdir)
        algo = raftsalgo.AlgoTrainEval(
            df=df, attrs=['attr1', 'attr2'], algo_config={'kmeans': [{'n_clusters': [1]}]},
            uncertainty={}, dir_out_alg_ds=dir_out_alg_ds, dataset_id='single_cluster_ds',
            metr='cluster_labels', task_type='clustering', test_size=0.3, rs=42,
            test_id_col='comid',
        )
        with self.assertLogs(level='WARNING') as cm:
            algo.train_eval()
        self.assertTrue(any("Only 1 cluster predicted" in log for log in cm.output))
        eval_entry = algo.eval_dict['kmeans_k1']
        self.assertTrue(np.isnan(eval_entry['silhouette_score']))
        self.assertTrue(np.isnan(eval_entry['davies_bouldin_score']))


class TestAlgoTrainEvalUncertaintyHelperEdgeCases(unittest.TestCase):
    """Covers calculate_bagging_ci's and calculate_mapie's error/edge branches
    directly, using a real fitted rf model (via split_data + train_algos)
    rather than mocking, per the project's mocking-avoidance preference."""

    def setUp(self):
        self.df = pd.DataFrame({
            'comid': [f'id_{i}' for i in range(15)],
            'attr1': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            'attr2': [5, 4, 3, 2, 1, 5, 4, 3, 2, 1, 5, 4, 3, 2, 1],
            'target': [10, 15, 20, 25, 30, 10, 15, 20, 25, 30, 10, 15, 20, 25, 30],
        })
        with tempfile.TemporaryDirectory() as tmpdir:
            dir_out_alg_ds = Path(tmpdir)
        self.algo = raftsalgo.AlgoTrainEval(
            df=self.df, attrs=['attr1', 'attr2'], algo_config={'rf': [{'n_estimators': 10}]},
            uncertainty={'bagging': [{'n_algos': 3}], 'mapie': [{'method': 'bogus', 'cv': 3, 'agg_function': 'median'}]},
            dir_out_alg_ds=dir_out_alg_ds, dataset_id='uncn_helpers_ds', metr='target',
            test_size=0.3, rs=42, test_id_col='comid',
        )
        self.algo.split_data()
        self.algo.select_algs_grid_search()
        self.algo.train_algos()

    def test_calculate_bagging_ci_raises_keyerror_for_unknown_algo(self):
        with self.assertRaises(KeyError):
            self.algo.calculate_bagging_ci('not_a_real_algo', self.algo.algs_dict['rf']['algo'])

    def test_calculate_bagging_ci_initializes_missing_uncertainty_key(self):
        # Real code paths (train_algos, calculate_forestci_uncertainty's caller
        # in train_eval) always pre-populate algs_dict[algo]['Uncertainty'] = {},
        # so this defensive branch is otherwise never exercised -- simulate an
        # algs_dict entry that lacks the key entirely, as if some other caller
        # built it without that default.
        del self.algo.algs_dict['rf']['Uncertainty']
        self.algo.calculate_bagging_ci('rf', self.algo.algs_dict['rf']['algo'])
        self.assertIn('Uncertainty', self.algo.algs_dict['rf'])
        self.assertIn('bagging_mean_pred', self.algo.algs_dict['rf']['Uncertainty'])

    def test_calculate_mapie_invalid_method_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.algo.calculate_mapie()


class TestCombineRespGdfComidWrap(unittest.TestCase):
    def test_combine_resp_gdf_comid_wrap(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            ds_dir = base_dir / "test_ds"
            ds_dir.mkdir()
            
            # 1. Create a real temporary NetCDF file
            ds = xr.Dataset(
                {"metric1": (("gage_id",), [10.0, 20.0])},
                coords={"gage_id": ["gage1", "gage2"]}
            )
            ds.to_netcdf(ds_dir / "test_dataset.nc")
            
            # 2. Create a real temporary GPKG file
            gdf = gpd.GeoDataFrame({
                "gage_id": ["gage1", "gage2"],
                "comid": ["111", "222"]
            }, geometry=[Point(0,0), Point(1,1)], crs="EPSG:4326")
            gdf.to_file(ds_dir / "test_dataset_loc.gpkg", driver="GPKG", layer="outlet")
            
            # 3. Use an existing config file path for the test
            path_attr_config = dir_test_data / "attr_config.yaml"
            
            # 4. Run the wrapper
            result = raftsutil.combine_resp_gdf_comid_wrap(
                dir_std_base=base_dir,
                ds="test_ds",
                path_attr_config=path_attr_config
            )
            
            self.assertIn('dat_resp', result)
            self.assertIn('gdf_comid', result)
            self.assertEqual(result['gdf_comid'].shape[0], 2)


def _write_minimal_attr_config(tmpdir) -> Path:
    """A minimal, real (not mocked) attr_config.yaml sufficient for
    AttrConfigAndVars._read_attr_config() and _read_metadata(). Points
    dir_std_base at a directory with no metadata parquet, so _read_metadata
    returns None and combine_resp_gdf_comid_wrap takes its dynamic
    featureID-mapping fallback branch.
    """
    tmpdir = Path(tmpdir)
    cfg = {
        'attr_select': [{'attr_vars': ['slope']}],
        'file_io': [
            {'dir_base': str(tmpdir)},
            {'dir_db_attrs': '{dir_base}/attrs'},
            {'dir_std_base': '{dir_base}/std'},  # deliberately not where the test data lives
            {'ds_type': 'training'},
            {'path_meta': '{dir_std_base}/{ds}/{ds}_{ds_type}.parquet'},
        ],
        'formulation_metadata': [{'datasets': ['test_ds']}],
        # combine_resp_gdf_comid_wrap accesses attr_config['col_schema'] directly
        # (not .get()), so this key must exist even though dat_resp.attrs
        # (set in the fixtures below) is what actually supplies the values.
        'col_schema': [{'featureID': 'USGS-{gage_id}'}, {'featureSource': 'nwissite'}],
    }
    path_cfg = tmpdir / "attr_config.yaml"
    with open(path_cfg, "w") as f:
        yaml.safe_dump(cfg, f)
    return path_cfg


class TestCombineRespGdfComidWrapMismatchBranches(unittest.TestCase):
    """
    Extends TestCombineRespGdfComidWrap.test_combine_resp_gdf_comid_wrap (whose
    2-gage, perfectly-aligned fixture never triggers the gage-count mismatch,
    NA-featureID removal, or 'too many .nc files' branches) with real (not
    mocked) xr.Dataset/GeoDataFrame objects built directly in a temp dir, sized
    and shaped to actually exercise those branches.
    """

    def _build_fixture(self, tmpdir, n_gages=20, featureSource='nwissite'):
        base_dir = Path(tmpdir)
        ds_dir = base_dir / "test_ds"
        ds_dir.mkdir()
        gage_ids = [f"gage_{i:03d}" for i in range(n_gages)]

        dat_resp = xr.Dataset(
            {"metric1": (("gage_id",), list(range(n_gages)))},
            coords={"gage_id": gage_ids},
            attrs={'featureSource': featureSource, 'featureID': 'USGS-{gage_id}'},
        )
        dat_resp.to_netcdf(ds_dir / "test_dataset.nc")
        return base_dir, ds_dir, gage_ids

    def test_too_many_nc_files_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir, ds_dir, _ = self._build_fixture(tmpdir)
            # A second .nc file in the same ds directory.
            xr.Dataset({"metric1": (("gage_id",), [1.0])}, coords={"gage_id": ["x"]}).to_netcdf(
                ds_dir / "a_second_copy.nc"
            )
            gpd.GeoDataFrame({"gage_id": ["x"], "comid": ["1"]}, geometry=[Point(0, 0)], crs="EPSG:4326") \
                .to_file(ds_dir / "test_dataset_loc.gpkg", driver="GPKG", layer="outlet")

            path_attr_config = _write_minimal_attr_config(tmpdir)
            with self.assertRaises(ValueError):
                raftsutil.combine_resp_gdf_comid_wrap(
                    dir_std_base=base_dir, ds="test_ds", path_attr_config=path_attr_config
                )

    def test_gage_count_mismatch_reduces_dat_resp(self):
        """
        gdf_comid has fewer gages than dat_resp (5 dropped from the companion
        GPKG) -- forces the gage-count mismatch warning and the dat_resp
        reduction. featureSource is set to a non-'nwissite'/'comid' value so
        rafts_retr_nhdp_comids_geom_wrap takes its offline-only branch
        regardless of gage coverage -- this test must never touch the network.

        NOTE: the function has a second, narrower warning ("...is less than
        the number of gage_ids in the response variable...") that only fires
        if sub_gdf_comid.shape[0] ends up *smaller* than dat_resp's gage count
        post-reduction. Given unique gage_ids on both sides (the normal case,
        reproduced here), reduction always brings them to exact equality, so
        that second warning is effectively unreachable through realistic data.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            n_gages = 20
            base_dir, ds_dir, gage_ids = self._build_fixture(
                tmpdir, n_gages=n_gages, featureSource='test_offline_source'
            )

            # Companion GPKG missing the first 5 gage_ids entirely.
            surviving_ids = gage_ids[5:]
            gdf = gpd.GeoDataFrame({
                "gage_id": surviving_ids,
                "comid": [f"c{i}" for i in range(len(surviving_ids))],
            }, geometry=[Point(i, i) for i in range(len(surviving_ids))], crs="EPSG:4326")
            gdf.to_file(ds_dir / "test_dataset_loc.gpkg", driver="GPKG", layer="outlet")

            path_attr_config = _write_minimal_attr_config(tmpdir)
            with self.assertLogs(level='WARNING') as cm:
                result = raftsutil.combine_resp_gdf_comid_wrap(
                    dir_std_base=base_dir, ds="test_ds", path_attr_config=path_attr_config
                )
            log_text = "\n".join(cm.output)
            self.assertIn("does not match the number of gage_ids", log_text)
            self.assertEqual(result['dat_resp'].sizes['gage_id'], n_gages - 5)

    def test_na_featureid_removal_via_incomplete_metadata(self):
        """
        The NA-featureID-removal branch is only reachable through the
        df_meta-present path (the dynamic-mapping fallback always computes a
        fresh, non-null featureID for every gage via string formatting, so it
        can never produce a NaN to remove). Provides a real metadata parquet
        file covering only some of dat_resp's gage_ids, so the
        `.merge(df_meta_map, on='gage_id', how='left')` leaves the
        uncovered gage_ids with a NaN featureID.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            n_gages = 15
            base_dir, ds_dir, gage_ids = self._build_fixture(
                tmpdir, n_gages=n_gages, featureSource='test_offline_source'
            )
            # Companion GPKG covers every gage (no count mismatch here).
            gdf = gpd.GeoDataFrame({
                "gage_id": gage_ids,
                "comid": [f"c{i}" for i in range(n_gages)],
            }, geometry=[Point(i, i) for i in range(n_gages)], crs="EPSG:4326")
            gdf.to_file(ds_dir / "test_dataset_loc.gpkg", driver="GPKG", layer="outlet")

            # Real metadata parquet, covering only 12 of the 15 gage_ids, at
            # the exact path _read_metadata resolves to.
            path_attr_config = _write_minimal_attr_config(tmpdir)
            meta_dir = Path(tmpdir) / "std" / "test_ds"  # matches file_io's dir_std_base: '{dir_base}/std'
            meta_dir.mkdir(parents=True)
            covered_ids = gage_ids[:12]
            pd.DataFrame({
                'gage_id': covered_ids,
                'featureID': [f"USGS-{g}" for g in covered_ids],
                'featureSource': ['test_offline_source'] * len(covered_ids),
            }).to_parquet(meta_dir / "test_ds_training.parquet")

            with self.assertLogs(level='INFO') as cm:
                result = raftsutil.combine_resp_gdf_comid_wrap(
                    dir_std_base=base_dir, ds="test_ds", path_attr_config=path_attr_config
                )
            log_text = "\n".join(cm.output)
            self.assertIn("returned location IDs are NA values", log_text)
            # 15 gages - 3 uncovered by metadata (NaN featureID, removed) = 12
            self.assertEqual(result['dat_resp'].sizes['gage_id'], 12)
            self.assertFalse(result['gdf_comid']['featureID'].isna().any())


class TestSimpleUtilities(unittest.TestCase):
    
    def test_check_attr_rm_dupes(self):
        # Create a dataframe with a duplicate attribute entry
        df = pd.DataFrame({
            'featureID': ['1', '1', '2'],
            'featureSource': ['src', 'src', 'src'],
            'data_source': ['ds', 'ds', 'ds'],
            'attribute': ['A', 'A', 'B'],
            'value': [10, 10, 20],
            'dl_timestamp': ['2023-01-01', '2023-01-02', '2023-01-01']
        })
        
        # Test that it drops the older duplicate (keeps 2023-01-02 by default ascending=True)
        # Note: ascending=True actually keeps the 'first' row after sorting.
        cleaned_df = raftsutil._check_attr_rm_dupes(df, sort_col='dl_timestamp', ascending=False)
        self.assertEqual(len(cleaned_df), 2)
        
    def test_find_common_comid(self):
        # Pass a dictionary of multiple DataFrames to find intersecting IDs
        df1 = gpd.GeoDataFrame({'featureID': ['A', 'B', 'C']})
        df2 = gpd.GeoDataFrame({'featureID': ['B', 'C', 'D']})
        
        common = raftsutil.find_common_comid({'ds1': df1, 'ds2': df2}, column='featureID')
        self.assertCountEqual(common, ['B', 'C'])

    def test_standard_path_generators(self):
        # Quickly hit all the basic Path generation functions
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "fake_dir"
            
            p1 = raftsutil._std_rafts_prep_ds_companion_gpkg_path(base / "data.nc")
            self.assertEqual(p1.name, "data_loc.gpkg")
            
            p2 = raftsutil.std_eval_metrs_path(base, "my_ds", "NSE")
            self.assertEqual(p2.name, "algo_eval_my_ds_NSE.csv")
            
            p3 = raftsutil.std_test_pred_obs_path(base, "my_ds", "NSE")
            self.assertEqual(p3.name, "pred_obs_my_ds_NSE.csv")

    def test_other_validation_wrappers(self):
        # 1. validate_gdf_comid_schema
        good_gdf = gpd.GeoDataFrame({'comid': ['123'], 'geometry': [Point(0,0)]})
        bad_gdf = pd.DataFrame({'wrong_col': [1]}) # Not a GeoDataFrame, missing cols
        
        raftsutil.validate_gdf_comid_schema(good_gdf, arg_val=False) # Happy path (disabled)
        with self.assertRaises(SystemExit):
            raftsutil.validate_gdf_comid_schema(bad_gdf, arg_val=True) # Fatal crash
            
        # 2. validate_df_comids
        good_attr = pd.DataFrame({'featureID': ['1'], 'attribute': ['A'], 'value': [1.0]})
        bad_attr = pd.DataFrame({'featureID': [1]}) # Int instead of str
        
        raftsutil.validate_df_comids(good_attr, arg_val=False)
        with self.assertRaises(SystemExit):
             raftsutil.validate_df_comids(bad_attr, arg_val=True)
             
        # 3. write_validated_prediction_output
        bad_pred = pd.DataFrame({'bad': ['data']})
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "out.parquet"
            with self.assertRaises(SystemExit):
                raftsutil.write_validated_prediction_output(bad_pred, out_path, True, ["NSE"])

class TestValidationUtilities(unittest.TestCase):
    def setUp(self):
        # Reuse existing dummy data strategy
        self.good_df = pd.DataFrame({
            'comid': ['123', '456'],
            'attr1': [1.0, 2.0]
        })
        self.bad_df = pd.DataFrame({
            'comid': [123, 456], # Int instead of string!
            'attr1': ['bad', 'data']
        })

    def test_validate_input_attributes_exit(self):
        # 1. Test happy path (no exit)
        raftsutil.validate_input_attributes(self.good_df, arg_val=False) 
        
        # 2. Test fatal exit using your existing bad dataframe
        with self.assertRaises(SystemExit):
            raftsutil.validate_input_attributes(self.bad_df, arg_val=True)

    def test_other_validation_wrappers(self):
        # 1. validate_gdf_comid_schema
        good_gdf = gpd.GeoDataFrame({'comid': ['123'], 'geometry': [Point(0,0)]})
        bad_gdf = pd.DataFrame({'wrong_col': [1]}) # Not a GeoDataFrame, missing cols
        
        raftsutil.validate_gdf_comid_schema(good_gdf, arg_val=False) # Happy path (disabled)
        with self.assertRaises(SystemExit):
            raftsutil.validate_gdf_comid_schema(bad_gdf, arg_val=True) # Fatal crash
            
        # 2. validate_df_comids
        good_attr = pd.DataFrame({'featureID': ['1'], 'attribute': ['A'], 'value': [1.0]})
        bad_attr = pd.DataFrame({'featureID': [1]}) # Int instead of str
        
        raftsutil.validate_df_comids(good_attr, arg_val=False)
        with self.assertRaises(SystemExit):
             raftsutil.validate_df_comids(bad_attr, arg_val=True)
             
        # 3. write_validated_prediction_output
        bad_pred = pd.DataFrame({'bad': ['data']})
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "out.parquet"
            with self.assertRaises(SystemExit):
                raftsutil.write_validated_prediction_output(bad_pred, out_path, True, ["NSE"])

    # DROP THE NEW TESTS RIGHT HERE!
    def test_load_validated_pipeline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pipe_path = Path(tmpdir) / "model.joblib"
            
            # Dump a dummy dictionary using joblib
            dummy_pipe = {"model_type": "RandomForest", "target": "NSE"}
            joblib.dump(dummy_pipe, pipe_path)
            
            # Happy path (no validation, just loads the file)
            loaded = raftsutil.load_validated_pipeline(pipe_path, arg_val=False)
            self.assertEqual(loaded["target"], "NSE")
            
            # Fatal path: Missing file
            with self.assertRaises(FileNotFoundError):
                raftsutil.load_validated_pipeline(Path(tmpdir) / "nope.joblib")
                
            # Fatal path: Fails Pydantic validation (assuming dummy_pipe is incomplete)
            with self.assertRaises(SystemExit):
                raftsutil.load_validated_pipeline(pipe_path, arg_val=True)

    def test_write_validated_evaluation_output_exit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            bad_df = pd.DataFrame({"random_col": [1, 2]}) # Wrong schema
            
            # Triggers the validation exit
            with self.assertRaises(SystemExit):
                raftsutil.write_validated_evaluation_output(
                    rslt_eval_df=bad_df, 
                    dir_out_alg_ds=out_dir, 
                    ds="my_ds", 
                    valid_metrics=["NSE"], 
                    arg_val=True
                )

class TestValidationHappyPaths(unittest.TestCase):
    """
    This class builds fixtures that genuinely satisfy each
    schema and drives arg_val=True end-to-end without hitting sys.exit.
    """

    def test_validate_input_attributes_happy_path(self):
        good_long_df = pd.DataFrame({
            'featureID': ['gage_1', 'gage_2'],
            'featureSource': ['hf_id', 'hf_id'],
            'data_source': ['usgs', 'usgs'],
            'attribute': ['slope', 'slope'],
            'value': [0.1, 0.2],
        })
        raftsutil.validate_input_attributes(good_long_df, arg_val=True)  # No SystemExit

    def test_validate_gdf_comid_schema_happy_path(self):
        good_gdf = gpd.GeoDataFrame({
            'comid': ['123'],
            'gage_id': ['USGS-01'],
            'featureID': ['USGS-01'],
            'featureSource': ['nwissite'],
            'X': [-100.0],
            'Y': [40.0],
            'tot_na': [0],
            'geometry': [Point(-100.0, 40.0)],
        })
        raftsutil.validate_gdf_comid_schema(good_gdf, arg_val=True)  # No SystemExit

    def test_validate_dat_resp_schema_happy_path(self):
        dat_resp = xr.Dataset(
            data_vars={
                'featureID': ('gage_id', ['f1', 'f2']),
                'featureSource': ('gage_id', ['nwissite', 'nwissite']),
                'NSE': ('gage_id', [0.5, 0.7]),
            },
            coords={'gage_id': ['g1', 'g2']},
            attrs={'respvar_mappings': 'NSE'},
        )
        raftsutil.validate_dat_resp_schema(
            dat_resp, valid_metrics=['NSE'], col_locid='featureID', arg_val=True
        )  # No SystemExit

    def test_validate_dat_resp_schema_exits_on_missing_column(self):
        # Missing 'featureSource' entirely -> KeyError inside the try block,
        # caught by the wrapper's broad `except Exception`.
        dat_resp = xr.Dataset(
            data_vars={'featureID': ('gage_id', ['f1', 'f2'])},
            coords={'gage_id': ['g1', 'g2']},
            attrs={'respvar_mappings': ''},
        )
        with self.assertRaises(SystemExit):
            raftsutil.validate_dat_resp_schema(
                dat_resp, valid_metrics=[], col_locid='featureID', arg_val=True
            )

    def test_read_validated_attribute_selection_happy_path(self):
        # attr_cfig only needs to duck-type .attrs_cfg_dict for the
        # name_attr_csv=None branch of _id_attrs_sel_wrap.
        attr_cfig = SimpleNamespace(attrs_cfg_dict={'attrs_sel': ['slope', 'elevation']})
        result = raftsutil.read_validated_attribute_selection(
            attr_cfig=attr_cfig, path_cfig='unused.yaml',
            name_attr_csv=None, colname_attr_csv=None, arg_val=True
        )
        self.assertEqual(sorted(result), ['elevation', 'slope'])

    def test_read_validated_attribute_selection_exits_on_empty_selection(self):
        # An empty attrs_sel produces pd.DataFrame([]), which has no column
        # named 0 -- schema_attrs_sel requires it, so validation fails.
        attr_cfig = SimpleNamespace(attrs_cfg_dict={'attrs_sel': []})
        with self.assertRaises(SystemExit):
            raftsutil.read_validated_attribute_selection(
                attr_cfig=attr_cfig, path_cfig='unused.yaml',
                name_attr_csv=None, colname_attr_csv=None, arg_val=True
            )


class TestTrainTestSplitWrap(unittest.TestCase):
    def test_split_train_test_comid_wrap(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            ds_dir = base_dir / "test_ds"
            ds_dir.mkdir()
            
            # Create dummy NetCDF
            ds = xr.Dataset(
                {"metric1": (("gage_id",), [10.0, 20.0, 30.0, 40.0])},
                coords={"gage_id": ["g1", "g2", "g3", "g4"]}
            )
            ds.to_netcdf(ds_dir / "test_dataset.nc")
            
            # Create dummy GPKG
            gdf = gpd.GeoDataFrame({
                "gage_id": ["g1", "g2", "g3", "g4"],
                "comid": ["11", "22", "33", "44"]
            }, geometry=[Point(0,0), Point(1,1), Point(2,2), Point(3,3)], crs="EPSG:4326")
            gdf.to_file(ds_dir / "test_dataset_loc.gpkg", driver="GPKG", layer="outlet")
            
            path_attr_config = dir_test_data / "attr_config.yaml"
            
            # Run the split wrapper
            result = raftsutil.split_train_test_comid_wrap(
                dir_std_base=base_dir,
                datasets=["test_ds"],
                path_attr_config=path_attr_config,
                id_col='comid',
                test_size=0.5 # Split the 4 rows perfectly in half
            )
            
            # Validate the output dictionary
            self.assertIn('dict_gdf_comids', result)
            self.assertEqual(len(result['sub_test_ids']), 2)
            self.assertEqual(len(result['sub_train_ids']), 2)

class TestProcessSingleMetric(unittest.TestCase):
    def test_process_single_metric_execution(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            
            # 1. Build tiny dummy DataFrames
            df_pred_resp = pd.DataFrame({
                'comid': [str(i) for i in range(1, 11)],
                'featureSource': ['COMID'] * 10,
                'target_metric': list(range(10, 110, 10)),
                'attr1': list(range(1, 11))
            })
            
            gdf_comid = gpd.GeoDataFrame({
                'comid': [str(i) for i in range(1, 11)]
            }, geometry=[Point(0,0)] * 10, crs="EPSG:4326")
            
            # 2. Package the arguments dictionary exactly as the ProcessPoolExecutor does
            args_dict = {
                'metr': 'target_metric',
                'task_type': 'regression',
                'df_pred_resp': df_pred_resp,
                'algo_config': {'rf': [{'n_estimators': [5,11]}]}, # Extremely fast 5-tree RF
                'attrs_sel': ['attr1'],
                'uncertainty_cfg': {},
                'dir_out_alg_ds': tmp_path,
                'ds': 'test_ds',
                'test_size': 0.2,
                'seed': 42,
                'col_locid': 'comid',
                'verbose': False,
                'confidence_levels': [95],
                'uncn_bnd_algo': False,
                'min_lim': None,
                'max_lim': None,
                'make_plots': False, # Skip plotting to keep this unit test lightning fast
                'save_all_clusters': False,
                'dir_out_viz_base': tmp_path,
                'dir_out_anlys_base': tmp_path,
                'gdf_comid': gdf_comid,
                'n_jobs': None
            }
            
            # 3. Execute the worker function
            metr, eval_df = raftsalgo._process_single_metric(args_dict)
            
            # 4. Assertions: Check Returns and File I/O
            self.assertEqual(metr, 'target_metric')
            self.assertIsInstance(eval_df, pd.DataFrame)
            self.assertFalse(eval_df.empty)
            
            # Verify the worker successfully wrote the prediction observation CSV to disk
            expected_csv = tmp_path / "test_ds" / "pred_obs_test_ds_target_metric.csv"
            self.assertTrue(expected_csv.exists(), "Worker failed to write pred_obs CSV!")


class TestAlgoTrainEvalClustering(unittest.TestCase):
    def setUp(self):
        # Create dummy data
        self.df = pd.DataFrame({
            'comid': [f'id_{i}' for i in range(30)],
            'attr1': np.random.rand(30),
            'attr2': np.random.rand(30),
        })
        
        self.algo_config = {
            'kmeans': [{'n_clusters': [2, 3]}],
            'gower_agglomerative': [{'n_clusters': [2]}]
        }
        
        self.algo_train_eval = raftsalgo.AlgoTrainEval(
            df=self.df, attrs=['attr1', 'attr2'], algo_config=self.algo_config,
            uncertainty={}, dir_out_alg_ds='./', dataset_id='test',
            metr='cluster_labels', task_type='clustering', test_size=0.3, rs=42,
            test_id_col='comid', save_all_clusters=True
        )

    def tearDown(self):
        """Clean up .joblib files created in the local directory upon completion."""
        files_to_remove = [
            'algo_kmeans_k3_cluster_labels__test.joblib',
            'algo_kmeans_k2_cluster_labels__test.joblib',
            'algo_gower_agglomerative_k2_cluster_labels__test.joblib'
        ]
        
        for file_name in files_to_remove:
            file_path = Path('./') / file_name
            if file_path.exists():
                file_path.unlink()

    def test_clustering_pipeline(self):
        # This single test will hit split_data, train_algos_grid_search, 
        # predict_algos, and evaluate_algos for the clustering branches!
        self.algo_train_eval.train_eval()
        
        # Verify the dynamic saving of all k-iterations worked
        self.assertIn('kmeans_k2', self.algo_train_eval.algs_dict)
        self.assertIn('kmeans_k3', self.algo_train_eval.algs_dict)
        self.assertIn('gower_agglomerative_k2', self.algo_train_eval.algs_dict)
        
        # Verify evaluation metrics generated successfully
        eval_dict = self.algo_train_eval.eval_dict
        self.assertIn('silhouette_score', eval_dict['kmeans_k2'])

    def test_universal_distance_clusterer(self):
        from sklearn.cluster import AgglomerativeClustering
        X_train = pd.DataFrame({'a': [0, 0, 10, 10], 'b': [0, 1, 10, 11]})
        X_test = pd.DataFrame({'a': [0.5, 9.5], 'b': [0.5, 10.5]})
        
        base_algo = AgglomerativeClustering(n_clusters=2, metric='precomputed', linkage='average')
        clusterer = raftsalgo.UniversalDistanceClusterer(estimator=base_algo, metric='euclidean')
        
        # Test fitting
        clusterer.fit(X_train)
        self.assertTrue(hasattr(clusterer, 'labels_'))
        
        # Test out-of-sample prediction (the KNN fallback)
        preds = clusterer.predict(X_test)
        self.assertEqual(len(preds), 2)
        self.assertNotEqual(preds[0], preds[1]) # They should be assigned to different clusters

    def test_process_single_metric_exception_handling(self):
        # Pass an intentionally broken args dictionary (missing 'attrs_sel')
        bad_args = {
            'metr': 'bad_metric',
            # Intentionally causing a KeyError by omitting required arguments
        }
        
        # Ensure it doesn't crash the test runner, but returns the safe failure tuple
        with self.assertLogs(level='ERROR') as cm:
            metr, eval_df = raftsalgo._process_single_metric(bad_args)
            
        self.assertEqual(metr, 'bad_metric')
        self.assertIsNone(eval_df)
        self.assertTrue(any("FAILED with error" in log for log in cm.output))

class TestMapieInferenceUtilities(unittest.TestCase):
    print("Testing MAPIE Inference Utilities")

    def test_infer_mapie_alphas(self):
        """Test extraction of alpha values from column names."""
        # 1. Happy path: Multiple alphas out of order with noise
        cols = ['prediction', 'mapie_lower_0.32', 'mapie_upper_0.32', 'forestci', 'mapie_lower_0.05']
        alphas = raftsutil.infer_mapie_alphas(cols)
        self.assertEqual(alphas, [0.05, 0.32], "Failed to extract and sort valid alpha floats.")

        # 2. Edge case: No mapie columns exist
        cols_empty = ['prediction', 'forestci', 'algo_name']
        self.assertEqual(raftsutil.infer_mapie_alphas(cols_empty), [], "Should return empty list when no MAPIE cols exist.")

        # 3. Edge case: Bad format (e.g., string cannot be cast to float)
        cols_bad = ['mapie_lower_abc', 'mapie_lower_0.10']
        self.assertEqual(raftsutil.infer_mapie_alphas(cols_bad), [0.10], "Failed to gracefully ignore malformed MAPIE column names.")
        print("✅ test_infer_mapie_alphas passed.")

    def test_infer_mapie_errors(self):
        """Test error calculation math and missing column handling."""
        # Setup a dummy dataframe with known math
        df = pd.DataFrame({
            'prediction': [10.0, 20.0, 30.0],
            'mapie_lower_0.05': [8.0, 19.0, 25.0],
            'mapie_upper_0.05': [12.0, 22.0, 31.0]
        })
        
        # 1. Happy path
        err_dict = raftsutil.infer_mapie_errors(df, alpha_val=0.05, colname_data='prediction')
        
        # Check that all keys were generated
        expected_keys = ['lower_err', 'upper_err', 'total_err', 'min_err', 'max_err']
        for key in expected_keys:
            self.assertIn(key, err_dict, f"Missing key {key} in returned dictionary.")
            
        # Check the underlying math 
        # Row 0: pred=10, low=8, up=12 -> lower_err=2, upper_err=2, total_err=4
        # Row 1: pred=20, low=19, up=22 -> total_err=3
        # Row 2: pred=30, low=25, up=31 -> total_err=6
        self.assertEqual(err_dict['lower_err'].iloc[0], 2.0, "Lower error math is incorrect.")
        self.assertEqual(err_dict['upper_err'].iloc[0], 2.0, "Upper error math is incorrect.")
        self.assertEqual(err_dict['total_err'].iloc[0], 4.0, "Total error math is incorrect.")
        
        # Min/Max total_err across all rows -> min=3.0, max=6.0
        self.assertEqual(err_dict['min_err'], 3.0, "Global minimum error calculation failed.")
        self.assertEqual(err_dict['max_err'], 6.0, "Global maximum error calculation failed.")

        # 2. Edge case: Missing alpha columns
        with self.assertRaises(KeyError) as context:
            raftsutil.infer_mapie_errors(df, alpha_val=0.10)
        self.assertIn("MAPIE columns for alpha 0.10 not found", str(context.exception))
        
        print("✅ test_infer_mapie_errors passed.")


class TestHfatlAggPathHelpers(unittest.TestCase):
    """std_dir_ds_agg, std_path_agg_ds, generate_vpu_attr_filepath, and
    std_impute_log_path had zero unit test coverage prior to this class."""

    def test_std_dir_ds_agg_creates_and_returns_agg_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dir_db_attrs = str(Path(tmpdir) / "attrs" / "{ds}")
            result = raftsutil.std_dir_ds_agg(dir_db_attrs, ds="my_ds")
            self.assertEqual(result, Path(tmpdir) / "attrs" / "my_ds_agg_hfatl")
            self.assertTrue(result.is_dir())

    def test_std_path_agg_ds(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = raftsutil.std_path_agg_ds(Path(tmpdir), ds="my_ds")
            self.assertEqual(result, Path(tmpdir) / "my_ds_agg_hfatlas.parquet")

    def test_generate_vpu_attr_filepath_dataset_name_not_in_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = raftsutil.generate_vpu_attr_filepath(Path(tmpdir), dataset_name="my_ds", vpuid="01")
            self.assertEqual(result, Path(tmpdir) / "my_ds" / "01" / "attr_01.parquet")
            self.assertTrue(result.parent.is_dir())

    def test_generate_vpu_attr_filepath_dataset_name_already_in_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dir_db_attrs = Path(tmpdir) / "my_ds"
            result = raftsutil.generate_vpu_attr_filepath(dir_db_attrs, dataset_name="my_ds", vpuid="02")
            self.assertEqual(result, dir_db_attrs / "02" / "attr_02.parquet")

    def test_std_impute_log_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = raftsutil.std_impute_log_path(Path(tmpdir), ds="my_ds", algo="kmeans_k3", resp_var="cluster_labels")
            self.assertEqual(result, Path(tmpdir) / "my_ds" / "imputed_locations_kmeans_k3_cluster_labels__my_ds.csv")
            self.assertTrue(result.parent.is_dir())


class TestAssignDonorsToReceivers(unittest.TestCase):
    def setUp(self):
        """Set up realistic, unmocked DataFrames for donor/receiver pairing."""
        self.attrs = ['attr1', 'attr2']
        self.id_col = 'featureID'
        self.cluster_col = 'prediction'

        # Donors DataFrame
        self.df_donors = pd.DataFrame({
            self.id_col: ['donor_1', 'donor_2', 'donor_3'],
            self.cluster_col: [1, 1, 3],
            'attr1': [0.0, 10.0, 5.0],
            'attr2': [0.0, 10.0, 5.0]
        })

        # Receivers DataFrame
        # recv_3 has no donor in cluster 2, recv_nan has a missing cluster prediction
        self.df_receivers = pd.DataFrame({
            self.id_col: ['recv_1', 'recv_2', 'recv_3', 'recv_4', 'recv_nan'],
            self.cluster_col: [1, 1, 2, 3, np.nan], 
            'attr1': [0.1, 9.9, 5.0, 4.9, 1.0],
            'attr2': [0.1, 9.9, 5.0, 4.9, 1.0]
        })

    def test_assign_donors_to_receivers_euclidean(self):
        """Test default euclidean metric handling, pairing logic, and missing donor warnings."""
        with self.assertLogs(level='WARNING') as cm:
            result = raftsalgo.assign_donors_to_receivers(
                df_donors=self.df_donors, 
                df_receivers=self.df_receivers, 
                attrs=self.attrs,
                metric='euclidean', 
                cluster_col=self.cluster_col, 
                id_col=self.id_col
            )

        # Confirm exact output format
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('distance_to_donor', result.columns)
        
        # We expect exactly 3 successful pairings (recv_3 missing donor, recv_nan missing cluster)
        self.assertEqual(len(result), 3) 

        # Validate specific pairings based on NearestNeighbor Euclidean distance
        pairings = result.set_index('receiver_id')['donor_id'].to_dict()
        self.assertEqual(pairings['recv_1'], 'donor_1')
        self.assertEqual(pairings['recv_2'], 'donor_2')
        self.assertEqual(pairings['recv_4'], 'donor_3')

        # Validate that the function logged warnings for the expected unassigned cases
        logs = str(cm.output)
        self.assertTrue("receivers have NaN cluster predictions" in logs)
        self.assertTrue("No donors found for Cluster 2.0" in logs)
        self.assertTrue("NOT assigned a donor" in logs)

    def test_assign_donors_to_receivers_no_pairings_possible(self):
        """When every receiver's cluster has no matching donor (or a NaN
        cluster), pairing_results never accumulates any rows -- previously
        untested branch where df_final_pairings falls back to an empty
        DataFrame rather than a pd.concat() result."""
        df_receivers_unpairable = pd.DataFrame({
            self.id_col: ['recv_1', 'recv_2'],
            self.cluster_col: [99, np.nan],  # cluster 99 has no donors; NaN is skipped
            'attr1': [0.1, 1.0],
            'attr2': [0.1, 1.0],
        })
        with self.assertLogs(level='WARNING'):
            result = raftsalgo.assign_donors_to_receivers(
                df_donors=self.df_donors,
                df_receivers=df_receivers_unpairable,
                attrs=self.attrs,
                metric='euclidean',
                cluster_col=self.cluster_col,
                id_col=self.id_col,
            )
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(result.empty)

    def test_assign_donors_to_receivers_gower(self):
        """Test alternative Gower metric execution."""
        # Use a cleaned receiver set to avoid triggering the unassigned warnings in this specific test
        df_recv_clean = self.df_receivers.dropna(subset=[self.cluster_col])
        df_recv_clean = df_recv_clean[df_recv_clean[self.cluster_col] != 2.0]

        result = raftsalgo.assign_donors_to_receivers(
            df_donors=self.df_donors, 
            df_receivers=df_recv_clean, 
            attrs=self.attrs,
            metric='gower', 
            cluster_col=self.cluster_col, 
            id_col=self.id_col
        )
        
        # Validate that Gower matrix operations succeeded and returned populated distances
        self.assertEqual(len(result), 3)
        self.assertFalse(result['distance_to_donor'].isna().any())

class TestAlgoTrainEvalCoverage(unittest.TestCase):
    def setUp(self):
        """Set up a basic AlgoTrainEval without relying on mocks."""
        df = pd.DataFrame({
            'comid': ['id1', 'id2', 'id3', 'id4'],
            'attr1': [1, 2, 3, 4],
            'metric': [10, 20, 30, 40]
        })
        
        # Provide minimal real arguments to initialize the object safely
        self.ate = raftsalgo.AlgoTrainEval(
            df=df, attrs=['attr1'], algo_config={}, uncertainty={},
            dir_out_alg_ds='.', dataset_id='test', metr='metric',
            test_id_col='comid'
        )

    def test_list_to_dict(self):
        """Test conversion of a list of dictionaries to a single dictionary."""
        # Test when input is a list of dicts
        list_input = [{'param1': 10}, {'param2': 20}]
        expected_dict = {'param1': 10, 'param2': 20}
        self.assertEqual(self.ate.list_to_dict(list_input), expected_dict)
        
        # Test when input is already a dict
        dict_input = {'param1': 10}
        self.assertEqual(self.ate.list_to_dict(dict_input), dict_input)

    def test_all_X_all_y(self):
        """Test the concatenation of train and test datasets."""
        # Manually assign splits using Pandas 
        self.ate.X_train = pd.DataFrame({'attr1': [1, 2]})
        self.ate.X_test = pd.DataFrame({'attr1': [3, 4]})
        self.ate.y_train = pd.Series([10, 20])
        self.ate.y_test = pd.Series([30, 40])
        
        X, y = self.ate.all_X_all_y()
        
        # Verify concatenated lengths
        self.assertEqual(len(X), 4)
        self.assertEqual(len(y), 4)
        
        # Based on the source code, X combines [X_train, X_test] and y combines [y_test, y_train]
        self.assertEqual(X['attr1'].tolist(), [1, 2, 3, 4])
        self.assertEqual(y.tolist(), [30, 40, 10, 20])

    def test_extr_rf_algo(self):
        """Test extracting the Random Forest model and its fallback warning log."""
        # Case 1: 'rf' exists in the algs_dict
        rf_model = RandomForestRegressor(n_estimators=5)
        self.ate.algs_dict = {'rf': {'algo': rf_model}}
        
        extracted = raftsalgo._extr_rf_algo(self.ate)
        self.assertIs(extracted, rf_model)
        
        # Case 2: 'rf' does not exist in the algs_dict
        self.ate.algs_dict = {'mlp': {'algo': MLPRegressor()}}
        
        with self.assertLogs(level='INFO') as cm:
            extracted_none = raftsalgo._extr_rf_algo(self.ate)
            
            self.assertIsNone(extracted_none)
            self.assertTrue(
                any("Trained random forest object 'rf' non-existent" in log for log in cm.output)
            )

    def test_extr_modl_algo_train(self):
        """Test that extr_modl_algo_train traverses the algs dictionary safely."""
        # Setup the plotting object with empty dummy structures
        plot_obj = raftsalgo.AlgoEvalPlotLC(X=pd.DataFrame(), y=pd.Series())
        
        # Populate the parent evaluation dictionary
        self.ate.algs_dict = {
            'rf': {'algo': RandomForestRegressor()},
            'mlp': {'algo': MLPRegressor()}
        }
        
        # The method currently performs no returns or explicit state mutations.
        # This test ensures it successfully iterates over the valid keys without raising a KeyError or Exception.
        try:
            plot_obj.extr_modl_algo_train(self.ate)
        except Exception as e:
            self.fail(f"extr_modl_algo_train raised an unexpected exception: {e}")


class TestAlgoTrainEvalBoosting(unittest.TestCase):
    def setUp(self):
        """Set up a sufficiently large DataFrame to survive cv=5 splits without mocking."""
        np.random.seed(42)
        # 30 rows is enough to satisfy sklearn's default 5-fold cross-validation
        self.df = pd.DataFrame({
            'comid': [f'id_{i}' for i in range(30)],
            'attr1': np.random.rand(30),
            'attr2': np.random.rand(30),
            'metric': np.random.rand(30)
        })
        self.attrs = ['attr1', 'attr2']
        self.dir_out_alg_ds = Path(tempfile.gettempdir())

    def test_train_algos_boosting_single(self):
        """Test direct initialization and training of all boosting algorithms without grid search."""
        # Use minimal iterations/estimators for extremely fast unmocked test execution
        algo_config = {
            'hgbr': {'max_iter': 5},
            'gbr': {'n_estimators': 5},
            'adaboost': {'n_estimators': 5},
            'xgb': {'n_estimators': 5, 'n_jobs': 1}
        }
        
        ate = raftsalgo.AlgoTrainEval(
            df=self.df, attrs=self.attrs, algo_config=algo_config,
            uncertainty={}, dir_out_alg_ds=self.dir_out_alg_ds,
            dataset_id='test_boost', metr='metric', test_size=0.2, test_id_col='comid'
        )
        
        # Execute raw train/predict flow
        ate.split_data()
        ate.train_algos()
        preds = ate.predict_algos()
        
        # 1. Verify all 4 models trained successfully and populated the dictionary
        self.assertIn('hgbr', ate.algs_dict)
        self.assertIn('gbr', ate.algs_dict)
        self.assertIn('adaboost', ate.algs_dict)
        self.assertIn('xgb', ate.algs_dict)
        
        # 2. Verify predictions were made accurately based on the X_test split size
        expected_test_len = len(ate.X_test)
        self.assertEqual(len(preds['hgbr']['y_pred']), expected_test_len)
        self.assertEqual(len(preds['xgb']['y_pred']), expected_test_len)

    def test_grid_search_and_bagging_extraction(self):
        """Test that grid search wraps the boosting models correctly and Bagging CI can unpack them."""
        # Provide lists to force select_algs_grid_search to trigger the GridSearchCV branches
        algo_config = {
            'adaboost': [{'n_estimators': [2, 4]}],
            'xgb': [{'n_estimators': [2, 4], 'n_jobs': [1]}]
        }
        
        # Enable Bagging to trigger the calculate_bagging_ci pipeline extraction logic
        # 2 algos is enough to test the math without wasting time
        uncertainty = {'bagging': [{'n_algos': 2}]}
        
        ate = raftsalgo.AlgoTrainEval(
            df=self.df, attrs=self.attrs, algo_config=algo_config,
            uncertainty=uncertainty, dir_out_alg_ds=self.dir_out_alg_ds,
            dataset_id='test_grid_boost', metr='metric', test_size=0.2, test_id_col='comid'
        )
        
        # Run full train_eval() to hit select_grid_search -> split -> train_grid -> calculate_bagging
        ate.train_eval()
        
        # 1. Assert GridSearchCV branches executed successfully
        self.assertIn('adaboost', ate.algs_dict)
        self.assertIn('xgb', ate.algs_dict)
        self.assertIn('gridsearchcv', ate.algs_dict['xgb'])
        
        # 2. Assert Bagging CI successfully unpacked the Pipeline to find the base estimator
        self.assertIn('Uncertainty', ate.algs_dict['xgb'])
        self.assertIn('bagging_confidence_intervals', ate.algs_dict['xgb']['Uncertainty'])
        self.assertIn('bagging_confidence_intervals', ate.algs_dict['adaboost']['Uncertainty'])

    def test_hgbr_gbr_grid_search(self):
        """hgbr/gbr grid search, previously untested (only their single-run
        branches in train_algos, and adaboost/xgb's grid-search branches,
        had coverage)."""
        algo_config = {
            'hgbr': [{'max_iter': [5, 10]}],
            'gbr': [{'n_estimators': [2, 4]}],
        }
        ate = raftsalgo.AlgoTrainEval(
            df=self.df, attrs=self.attrs, algo_config=algo_config,
            uncertainty={}, dir_out_alg_ds=self.dir_out_alg_ds,
            dataset_id='test_grid_hgbr_gbr', metr='metric', test_size=0.2,
            test_id_col='comid', verbose=True,
        )
        with self.assertLogs(level='INFO') as cm:
            ate.train_eval()
        log_text = "\n".join(cm.output)
        self.assertIn("Performing HistGradientBoostingRegressor Training with Grid Search", log_text)
        self.assertIn("Performing GradientBoostingRegressor Training with Grid Search", log_text)
        self.assertIn('gridsearchcv', ate.algs_dict['hgbr'])
        self.assertIn('gridsearchcv', ate.algs_dict['gbr'])


class TestAlgoTrainEvalDataAndIO(unittest.TestCase):
    """No-mock tests for data splitting edge cases and physical file I/O operations."""
    
    def setUp(self):
        np.random.seed(42)
        # Create a temporary directory that acts as a real filesystem for this test
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir_out_alg_ds = Path(self.temp_dir.name)
        
        # INCREASED SIZE: Create a dataframe with 30 rows to satisfy MAPIE sample requirements
        self.df = pd.DataFrame({
            'featureID': [f'id_{i}' for i in range(30)],
            'attr1': np.random.rand(30),
            'attr2': np.random.rand(30),
            'metric': np.random.rand(30)
        })
        # Inject an explicit NA to test the NA dropping logic and warning
        self.df.loc[29, 'attr1'] = np.nan
        
        self.attrs = ['attr1', 'attr2']
        # Very small estimator count for lightning-fast execution
        self.algo_config = {'rf': {'n_estimators': 2, 'random_state': 42}}
        
    def tearDown(self):
        # Physically clean up the directory and files after the test
        self.temp_dir.cleanup()
        
    def test_split_data_drops_nas(self):
        """Test that split_data physically drops rows with NAs and logs the correct warning."""
        ate = raftsalgo.AlgoTrainEval(
            df=self.df, attrs=self.attrs, algo_config=self.algo_config,
            uncertainty={}, dir_out_alg_ds=self.dir_out_alg_ds,
            dataset_id='test_io', metr='metric', test_size=0.2, test_id_col='featureID'
        )
        
        with self.assertLogs(level='WARNING') as cm:
            ate.split_data()
        
        # 29 valid rows. Verify the 30th row (NA) is completely gone.
        self.assertEqual(len(ate.X_train) + len(ate.X_test), 29)
        self.assertTrue(any("NA VALUES FOUND IN INPUT DATASET" in log for log in cm.output))
        
    def test_split_data_explicit_test_ids(self):
        """Test data splitting when test_ids are explicitly provided instead of a random split."""
        # Use the clean subset (first 29 rows) and set the index
        clean_df = self.df.iloc[:29].copy()
        clean_df.set_index('featureID', inplace=True)
        
        # Explicitly select 3 specific IDs to act as the holdout test set
        test_indices = ['id_0', 'id_4', 'id_8']
        test_ids_series = pd.Series(test_indices, index=test_indices)
        
        ate = raftsalgo.AlgoTrainEval(
            df=clean_df, attrs=self.attrs, algo_config=self.algo_config,
            uncertainty={}, dir_out_alg_ds=self.dir_out_alg_ds,
            dataset_id='test_io', metr='metric', test_size=0.2, test_id_col='featureID',
            test_ids=test_ids_series
        )
        ate.split_data()
        
        # Verify the exact indices ended up in the test set, and nowhere else
        self.assertEqual(len(ate.X_test), 3)
        self.assertEqual(len(ate.X_train), 26)
        self.assertTrue(all(idx in ate.X_test.index for idx in test_indices))
        self.assertFalse(any(idx in ate.X_train.index for idx in test_indices))

    def test_save_algos_and_metadata_physical_io(self):
        """Test actual saving of .joblib files and metadata dataframe population without mocks."""
        clean_df = self.df.iloc[:29].copy()
        
        # Include MAPIE to ensure it gets packed into the joblib dictionary correctly
        ate = raftsalgo.AlgoTrainEval(
            df=clean_df, attrs=self.attrs, algo_config=self.algo_config,
            uncertainty={'mapie': [{'alpha': [0.1], 'method': 'plus', 'cv': 2, 'agg_function': 'median'}]},
            dir_out_alg_ds=self.dir_out_alg_ds,
            dataset_id='test_io', metr='metric', test_size=0.2, test_id_col='featureID'
        )
        
        # Execute the required prerequisites
        ate.split_data()
        ate.train_algos()
        ate.calculate_mapie()
        ate.predict_algos()
        ate.evaluate_algos()
        
        # 1. Test Physical File Writing
        ate.save_algos()
        
        expected_joblib = self.dir_out_alg_ds / "algo_rf_metric__test_io.joblib"
        self.assertTrue(expected_joblib.exists(), "Joblib file was not written to disk!")
        
        # Actually load the joblib file off the disk to verify its contents
        saved_data = joblib.load(expected_joblib)
        self.assertIn('pipeline', saved_data)
        self.assertIn('mapie', saved_data)
        self.assertIn('X_train_shape', saved_data)
        self.assertEqual(saved_data['X_train_shape'], ate.X_train.shape)
        
        # 2. Test Metadata Organization
        ate.org_metadata_alg()
        
        # Verify eval_df was built and populated with the correct string references
        self.assertFalse(ate.eval_df.empty)
        self.assertEqual(ate.eval_df['dataset'].iloc[0], 'test_io')
        self.assertEqual(ate.eval_df['file_pipe'].iloc[0], "algo_rf_metric__test_io.joblib")
        self.assertIn('mse', ate.eval_df.columns)
        self.assertIn('r2', ate.eval_df.columns)

#%% functions corresponding to rafts_regn_params_gpkg.py

@pytest.fixture
def sample_dataframe():
    """Fixture providing a sample DataFrame for database operations."""
    return pd.DataFrame({
        'divide_id': ['cat-1', 'cat-2'],
        'param_a': [1.5, 2.5],
        'param_b': [100, 200]
    })

# =====================================================================
# Tests for resolve_fstrings (zero coverage prior to this)
# =====================================================================

def test_resolve_fstrings_non_string_returned_unchanged():
    assert raftsutil.resolve_fstrings(42, {'a': '1'}) == 42

def test_resolve_fstrings_no_braces_returned_unchanged():
    assert raftsutil.resolve_fstrings("plain string", {'a': '1'}) == "plain string"

def test_resolve_fstrings_resolves_within_max_depth():
    result = raftsutil.resolve_fstrings("{a}/{b}", {'a': 'x', 'b': 'y'})
    assert result == "x/y"

def test_resolve_fstrings_missing_key_left_unresolved():
    # SafeDict leaves an unknown key as the literal unresolved placeholder.
    result = raftsutil.resolve_fstrings("{a}/{missing}", {'a': 'x'})
    assert result == "x/{missing}"

def test_resolve_fstrings_returns_partial_result_when_depth_exhausted():
    # A resolution chain longer than max_depth never reaches a fixed point
    # (new_val keeps changing each iteration) -- the loop exhausts and
    # returns whatever the last iteration produced, rather than the fully
    # resolved string.
    context = {'a': '{b}', 'b': '{c}', 'c': '{d}', 'd': 'final'}
    result = raftsutil.resolve_fstrings("{a}", context, max_depth=3)
    assert result == "{d}"  # not fully resolved to "final" -- one iteration short

# =====================================================================
# Tests for register_gpkg_attributes_table
# =====================================================================

def test_register_gpkg_attributes_table_success():
    """Test that a table is successfully registered when gpkg_contents exists."""
    # Use a real in-memory SQLite database
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    
    # Manually create the GeoPackage master metadata table
    cursor.execute("""
        CREATE TABLE gpkg_contents (
            table_name TEXT, 
            data_type TEXT, 
            identifier TEXT, 
            description TEXT
        )
    """)
    
    table_name = "test_formulation"
    
    # Call the actual function
    raftsutil.register_gpkg_attributes_table(conn, table_name)
    
    # Verify the insertion was successful
    cursor.execute("SELECT table_name, data_type FROM gpkg_contents")
    result = cursor.fetchone()
    
    assert result is not None
    assert result[0] == table_name
    assert result[1] == 'attributes'
    conn.close()

def test_register_gpkg_attributes_table_sqlite_error_logs_warning(caplog):
    """A closed connection raises sqlite3.ProgrammingError (a sqlite3.Error
    subclass) on .cursor() -- previously untested exception branch."""
    conn = sqlite3.connect(':memory:')
    conn.close()
    raftsutil.register_gpkg_attributes_table(conn, "test_formulation")
    assert "Could not register table 'test_formulation' in gpkg_contents" in caplog.text

def test_register_gpkg_attributes_table_missing_contents():
    """Test that the function safely exits without errors if gpkg_contents is missing."""
    conn = sqlite3.connect(':memory:')
    table_name = "test_formulation"
    
    # Call without creating gpkg_contents; per the function's logic, it should check 
    # sqlite_master, find nothing, and bypass execution without crashing.
    raftsutil.register_gpkg_attributes_table(conn, table_name)
    
    cursor = conn.cursor()
    # Verify no random tables were created
    cursor.execute("SELECT count(name) FROM sqlite_master WHERE type='table'")
    assert cursor.fetchone()[0] == 0
    conn.close()

# =====================================================================
# Tests for create_sqlite_index
# =====================================================================

def test_create_sqlite_index_success():
    """Test that a valid SQLite index is explicitly created on the specified column."""
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    
    # Create a real dummy table
    table_name = "test_params"
    index_col = "divide_id"
    cursor.execute(f"CREATE TABLE {table_name} ({index_col} TEXT, value REAL)")
    
    # Call the indexing function
    raftsutil.create_sqlite_index(conn, table_name, index_col)
    
    # Verify the index exists in the SQLite master schema
    cursor.execute(f"SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='{table_name}'")
    
    # Fetch all indices for the table to ensure ours is included
    indices = [row[0] for row in cursor.fetchall()]
    assert f"idx_{table_name}_{index_col}" in indices
    conn.close()

def test_create_sqlite_index_invalid_table(caplog):
    """Test that attempting to index a non-existent table is caught and logged as a sqlite3.Error."""
    conn = sqlite3.connect(':memory:')
    
    # Call function on a table that doesn't exist; triggers a SQLite syntax error
    raftsutil.create_sqlite_index(conn, "missing_table", "divide_id")
    
    # Check that the error was caught and logged gracefully without raising an exception
    assert "Could not create SQLite index on 'missing_table'" in caplog.text
    conn.close()

# =====================================================================
# Tests for update_database
# =====================================================================

def test_update_database_create_new(tmp_path, sample_dataframe):
    """Test writing a brand-new table to a real file-based SQLite database."""
    # Use Pytest's tmp_path to create a real file that gets cleaned up automatically
    db_path = tmp_path / "test_output.sqlite"
    table_name = "formulation_kmeans"
    id_col = "divide_id"
    
    raftsutil.update_database(db_path, sample_dataframe, table_name, id_col, overwrite=False)
    
    with sqlite3.connect(db_path) as conn:
        # 1. Verify data exists
        df_result = pd.read_sql(f"SELECT * FROM {table_name}", conn)
        assert len(df_result) == 2
        assert id_col in df_result.columns
        
        # 2. Verify the formal index was successfully created via the helper
        cursor = conn.cursor()
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='{table_name}'")
        
        # FIX: Check if our explicit index is IN the list of indices, 
        # bypassing the auto-generated Pandas 'ix_' index.
        indices = [row[0] for row in cursor.fetchall()]
        assert f"idx_{table_name}_{id_col}" in indices

def test_update_database_overwrite(tmp_path, sample_dataframe):
    """Test that the overwrite=True flag completely drops and replaces existing data."""
    db_path = tmp_path / "test_output.sqlite"
    table_name = "formulation_kmeans"
    id_col = "divide_id"
    
    # Initial write to establish the table
    raftsutil.update_database(db_path, sample_dataframe, table_name, id_col, overwrite=False)
    
    # Create a new DataFrame with completely different data and IDs
    overwrite_df = pd.DataFrame({
        'divide_id': ['cat-99'],
        'param_a': [9.9],
        'param_b': [999]
    })
    
    # Execute the overwrite
    raftsutil.update_database(db_path, overwrite_df, table_name, id_col, overwrite=True)
    
    with sqlite3.connect(db_path) as conn:
        df_result = pd.read_sql(f"SELECT * FROM {table_name}", conn)
        # Verify the table was replaced: it should only contain the 1 new row, not 3
        assert len(df_result) == 1
        assert df_result.iloc[0]['divide_id'] == 'cat-99'

def test_update_database_append_new_only(tmp_path, sample_dataframe):
    """Test that the append logic strictly ignores redundant records and appends missing IDs."""
    db_path = tmp_path / "test_output.sqlite"
    table_name = "formulation_kmeans"
    id_col = "divide_id"
    
    # Initial write: contains 'cat-1' and 'cat-2'
    raftsutil.update_database(db_path, sample_dataframe, table_name, id_col, overwrite=False)
    
    # DataFrame containing 1 old ID ('cat-2') and 1 new ID ('cat-3')
    append_df = pd.DataFrame({
        'divide_id': ['cat-2', 'cat-3'], 
        'param_a': [2.5, 3.5],
        'param_b': [200, 300]
    })
    
    # Attempt to append
    raftsutil.update_database(db_path, append_df, table_name, id_col, overwrite=False)
    
    with sqlite3.connect(db_path) as conn:
        df_result = pd.read_sql(f"SELECT * FROM {table_name}", conn)
        
        # Should contain 'cat-1', 'cat-2', and 'cat-3'. The duplicated 'cat-2' was ignored.
        assert len(df_result) == 3
        assert set(df_result['divide_id']) == {'cat-1', 'cat-2', 'cat-3'}
        
        # Ensure our custom index was reapplied after appending
        cursor = conn.cursor()
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='{table_name}'")
        indices = [row[0] for row in cursor.fetchall()]
        assert f"idx_{table_name}_{id_col}" in indices

def test_update_database_append_missing_id_col(tmp_path, sample_dataframe, caplog):
    """Test that attempting to append to a table lacking the designated id_col is caught."""
    db_path = tmp_path / "test_output.sqlite"
    table_name = "formulation_kmeans"
    id_col = "divide_id"
    
    # Manually create a table without the expected identifier column
    bad_df = pd.DataFrame({
        'wrong_id': ['cat-1'], 
        'param_a': [1.0]
    })
    with sqlite3.connect(db_path) as conn:
        bad_df.to_sql(table_name, conn, index=False)
    
    # Attempt to use the append feature which requires reading the existing id_col
    raftsutil.update_database(db_path, sample_dataframe, table_name, id_col, overwrite=False)
    
    # Ensure the DatabaseError was gracefully caught and logged by our updated except block
    assert f"Identifier column '{id_col}' missing in the existing table '{table_name}'" in caplog.text

def test_update_database_no_new_records_to_append(tmp_path, sample_dataframe, caplog):
    """When every id in df_data already exists in the table, df_new ends up
    empty and the function should log that rather than attempting an append."""
    import logging as _logging
    db_path = tmp_path / "test_output.sqlite"
    table_name = "formulation_kmeans"
    id_col = "divide_id"

    raftsutil.update_database(db_path, sample_dataframe, table_name, id_col, overwrite=False)

    with caplog.at_level(_logging.INFO):
        # Re-submit the exact same rows -- nothing new to append.
        raftsutil.update_database(db_path, sample_dataframe.copy(), table_name, id_col, overwrite=False)
    assert f"No new records to append. Table '{table_name}' is up to date." in caplog.text

def test_update_database_outer_sqlite_error_exits(tmp_path, sample_dataframe):
    """A genuine SQLite-level error at the outer scope (not the inner,
    already-tested missing-id_col DatabaseError) should log and sys.exit(1)."""
    # A directory, not a file, is not a valid sqlite3.connect() target.
    bad_db_path = tmp_path / "not_a_valid_sqlite_file"
    bad_db_path.mkdir()
    with pytest.raises(SystemExit):
        raftsutil.update_database(bad_db_path, sample_dataframe, "formulation_kmeans", "divide_id", overwrite=False)

def test_get_crosswalk_target_col_priority_order():
    # 1. Explicitly configured column wins even if a default-named column also exists.
    df = pd.DataFrame({'pred_id': ['a'], 'divide_id': ['b'], 'custom_target': ['c']})
    assert raftsutil.get_crosswalk_target_col(df, 'pred_id', crosswalk_target_col='custom_target') == 'custom_target'

def test_get_crosswalk_target_col_default_fallback():
    # 2. No explicit target configured (or it's not present) -> standard hydrofabric name.
    df = pd.DataFrame({'pred_id': ['a'], 'divide_id': ['b']})
    assert raftsutil.get_crosswalk_target_col(df, 'pred_id', crosswalk_target_col=None) == 'divide_id'

def test_get_crosswalk_target_col_subtraction_fallback():
    # 3. No default hydrofabric name present -> subtract known metadata columns.
    df = pd.DataFrame({'pred_id': ['a'], 'vpuid': ['01'], 'areasqkm': [1.0], 'my_custom_id': ['x']})
    result = raftsutil.get_crosswalk_target_col(df, 'pred_id', crosswalk_target_col=None)
    assert result == 'my_custom_id'

def test_get_crosswalk_target_col_missing_pred_id_returns_none():
    df = pd.DataFrame({'divide_id': ['b']})
    assert raftsutil.get_crosswalk_target_col(df, 'pred_id_not_present') is None

def test_get_crosswalk_target_col_no_candidates_returns_none():
    # After subtracting pred_gpkg_id_col and all known metadata columns, nothing is left.
    df = pd.DataFrame({'pred_id': ['a'], 'vpuid': ['01'], 'gage_id': ['g1']})
    assert raftsutil.get_crosswalk_target_col(df, 'pred_id', crosswalk_target_col=None) is None

if __name__ == '__main__':
    unittest.main()
    logging.shutdown()