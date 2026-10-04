# plots.py
from __future__ import annotations # enables the | operator for function typehints back to python 3.7
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.figure import Figure
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import pathlib
from pathlib import Path
from shapely.geometry import LineString
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import geopandas as gpd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import zipfile
from typing import Iterable
import rafts_algo.utils as raftsutil
import os
import gc

# Set up basic logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def plot_corr_mat(df_X: pd.DataFrame,
                title='Feature Correlation Matrix'
                ) -> matplotlib.figure.Figure:
    """Generate a plot of the correlation matrix

    :param df_X: The dataset dataframe
    :type df_X: pd.DataFrame
    :param title: Plot title, defaults to 'Feature Correlation Matrix'
    :type title: str, optional
    :return: The correlation matrix figure
    :rtype: matplotlib.figure.Figure
    """
    # Calculate the correlation matrix
    df_corr = df_X.corr()

    #  Plot the correlation matrix
    plt.figure(figsize=(10,8))
    sns.heatmap(df_corr, annot=True, cmap ='coolwarm',linewidths=0.5, fmt='.2f')
    plt.title(title)

    fig = plt.gcf()
    return fig

def std_corr_mat_plot_path(dir_out_viz_base: str | Path,
                            ds: str
                            ) -> pathlib.PosixPath:
    """Standardize the filepath for saving correlation matrix above a threshold

    :param dir_out_viz_base: The base visualization output directory
    :type dir_out_viz_base: str | os.PathLike
    :param ds: The dataset name
    :type ds: str
    :return: The correlation matrix filepath
    :rtype: pathlib.PosixPath
    """
    path_corr_mat = Path(f"{dir_out_viz_base}/{ds}/correlation_matrix_{ds}.png")
    path_corr_mat.parent.mkdir(parents=True,exist_ok=True)
    return path_corr_mat

def plot_corr_mat_save_wrap(df_X:pd.DataFrame, title:str,
                            dir_out_viz_base:str | Path,
                            ds:str)-> matplotlib.figure.Figure:
    """Wrapper to plot and save the dataset correlation matrix

    :param df_X: The full dataset of interest, e.g. used for training/validation
    :type df_X: pd.DataFrame
    :param title: Title to place in the correlation matrix plot
    :type title: str
    :param dir_out_viz_base: base directory for saving visualization
    :type dir_out_viz_base: str | os.PathLike
    :param ds: The dataset name to use in plot title and filename
    :type ds: str
    :return: The correlation matrix plot
    :rtype: matplotlib.figure.Figure
    """
    fig_corr_mat = plot_corr_mat(df_X, title)
    path_corr_mat = std_corr_mat_plot_path(dir_out_viz_base,ds)
    fig_corr_mat.savefig(path_corr_mat)
    logging.info(f"Wrote the {ds} dataset correlation matrix to:\n{path_corr_mat}")
    return fig_corr_mat

def std_corr_path(dir_out_anlys_base: str|Path, ds:str,
                   cstm_str:str=None) -> pathlib.PosixPath:
    """Standardize the filepath that saves correlated attributes

    :param dir_out_anlys_base: The standardized analysis output directory
    :type dir_out_anlys_base: str | os.PathLike
    :param ds: the dataset name
    :type ds: str
    :param cstm_str: The option to add in a custom string such as the correlation threshold, defaults to None
    :type cstm_str: str, optional
    :return: Full filepath for saving correlated attributes table
    :rtype: pathlib.PosixPath
    """
    # TODO generate a file of the correlated attributes:
    if cstm_str:
        path_corr_attrs = Path(f"{dir_out_anlys_base}/{ds}/correlated_attrs_{ds}_{cstm_str}.csv")
    else:
        path_corr_attrs = Path(f"{dir_out_anlys_base}/{ds}/correlated_attrs_{ds}.csv")
    path_corr_attrs.parent.mkdir(parents=True,exist_ok=True)
    return path_corr_attrs

def corr_attrs_thr_table(df_X:pd.DataFrame, 
                        corr_thr:float = 0.8) -> pd.DataFrame:
    """Create a table of correlated attributes exceeding a threshold, with correlation values

    :param df_X: The attribute dataset
    :type df_X: pd.DataFrame
    :param corr_thr: The correlation threshold, between 0 & 1. Absolute values above this should be reduced, defaults to 0.8
    :type corr_thr: float, optional
    :return: The table of attribute pairings whose absolute correlations exceed a threshold
    :rtype: pd.DataFrame
    """
    df_corr = df_X.corr()

    # TODO Change code to selecting upper triangle of correlation matrix
    upper = df_corr.abs().where(np.triu(np.ones(df_corr.shape), k=1).astype(bool))

    # Find attributes with correlation greater than a certain threshold
    row_idx, col_idx = np.where(df_corr.abs() > corr_thr)
    df_corr_rslt = pd.DataFrame({'attr1': df_corr.columns[row_idx],
                'attr2': df_corr.columns[col_idx],
                'corr' : [df_corr.iat[row, col] for row, col in zip(row_idx, col_idx)]
                })
    # Remove the identical attributes
    df_corr_rslt = df_corr_rslt[df_corr_rslt['attr1']!= df_corr_rslt['attr2']].drop_duplicates()
    return df_corr_rslt

def write_corr_attrs_thr(df_corr_rslt:pd.DataFrame,path_corr_attrs: str | Path):
    """Wrapper to generate high correlation pairings table and write to file

    :param df_corr_rslt: _description_
    :type df_corr_rslt: pd.DataFrame
    :param path_corr_attrs: csv write path
    :type path_corr_attrs: str | os.PathLike
    """

    df_corr_rslt.to_csv(path_corr_attrs) # INSPECT THIS FILE 
    logging.info(f"Wrote highly correlated attributes to {path_corr_attrs}")
    logging.info("The user may now inspect the correlated attributes and make decisions on which ones to exclude")

def corr_thr_write_table_wrap(df_X:pd.DataFrame,dir_out_anlys_base:str|Path,
                              ds:str,corr_thr:float=0.8)->pd.DataFrame:
    """Wrapper to generate high correlation pairings table above an absolute threshold of interest and write to file
    
    :param df_X: The attribute dataset
    :type df_X: pd.DataFrame
    :param dir_out_anlys_base: The standard analysis directory
    :type path_corr_attrs: str | os.PathLike
    :param ds: The dataset name
    :type ds: str
    :param corr_thr: The correlation threshold, between 0 & 1. Absolute values above this detected, defaults to 0.8
    :type corr_thr: float, optional
    :return: The table of attribute pairings whose absolute correlations exceed a threshold
    :rtype: pd.DataFrame
    """
    # Generate the paired table of attributes correlated above an absolute threshold
    df_corr_rslt = corr_attrs_thr_table(df_X,corr_thr)
    path_corr_attrs_cstm = std_corr_path(dir_out_anlys_base=dir_out_anlys_base,
                                          ds=ds,
                                         cstm_str=f'thr{corr_thr}') 
    write_corr_attrs_thr(df_corr_rslt,path_corr_attrs_cstm)
    return df_corr_rslt

def pca_stdscaled_tfrm(df_X:pd.DataFrame, 
                       std_scale:bool=True
                       )->PCA:
    """Generate the PCA object, and perform a standardized scaler transformation if desired

    :param df_X: Dataframe of attribute data
    :type df_X: pd.DataFrame
    :param std_scale: Should the data be standard scaled?, defaults to True
    :type std_scale: bool, optional
    :return: The principal components analysis object
    :rtype: PCA
    """
    
    # Fit using the scaled data
    if std_scale:
        scaler = StandardScaler().fit(df_X)
        df_X_scaled = pd.DataFrame(scaler.transform(df_X), index=df_X.index.values, columns=df_X.columns.values)
    else:
        df_X_scaled = df_X.copy()
    pca_scaled = PCA()
    pca_scaled.fit(df_X_scaled)
    #cpts_scaled = pd.DataFrame(pca.transform(df_X_scaled))

    return pca_scaled

def plot_pca_stdscaled_tfrm(pca_scaled:PCA, 
                            title:str = 'Explained Variance Ratio by Principal Component',
                            std_scale:bool=True)-> matplotlib.figure.Figure:
    """Generate variance explained by PCA plot

    :param pca_scaled:  The PCA object generated from dataset
    :type pca_scaled: PCA
    :param title: plot title, defaults to 'Explained Variance Ratio by Principal Component'
    :type title: str, optional
    :param std_scale: Have the data been standardized,, defaults to True
    :type std_scale: bool, optional
    :return: Plot of the variance explained by PCA
    :rtype: matplotlib.figure.Figure
    """
    
    if std_scale:
        xlabl = 'Principal Component of Standardized Data'
    else:
        xlabl = 'Principal Component'
    # Create the plot for explained variance ratio
    x_axis = np.arange(1, pca_scaled.n_components_ + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, pca_scaled.explained_variance_ratio_, marker='o', linestyle='--', color='b')
    plt.xlabel(xlabl)
    plt.ylabel('Explained Variance Ratio')
    plt.title(title)
    plt.xticks(x_axis)
    plt.grid(True)

    fig = plt.gcf()
    return fig

def plot_pca_stdscaled_cumulative_var(pca_scaled:PCA, 
                                      title='Cumulative Proportion of Variance Explained vs Principal Components',
                                      std_scale:bool=True) -> matplotlib.figure.Figure:
    """Generate cumulative variance PCA plot

    :param pca_scaled: The PCA object
    :type pca_scaled: PCA
    :param title: plot title, defaults to 'Cumulative Proportion of Variance Explained vs Principal Components'
    :type title: str, optional
    :param std_scale: Have the data been standardized, defaults to True
    :type std_scale: bool, optional
    :return: Plot of the cumulative PCA variance
    :rtype: matplotlib.figure.Figure
    """
    if std_scale:
        xlabl = 'Principal Component of Standardized Data'
    else:
        xlabl = 'Principal Component'

    # Calculate the cumulative variance explained
    cumulative_variance_explained = np.cumsum(pca_scaled.explained_variance_ratio_)
    x_axis = np.arange(1, pca_scaled.n_components_ + 1)

    # Create the plot for cumulative proportion of variance explained
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, cumulative_variance_explained, marker='o', linestyle='-', color='b')
    plt.xlabel(xlabl)
    plt.ylabel('Cumulative Proportion of Variance Explained')
    plt.title(title)
    plt.xticks(x_axis)
    plt.grid(True)

    fig = plt.gcf()
    return fig 


def std_pca_plot_path(dir_out_viz_base: str|Path,
                      ds:str, cstm_str:str=None
                      ) -> pathlib.PosixPath:
    """Standardize the filepath for saving principal component analysis plots

    :param dir_out_viz_base: The base visualization output directory
    :type dir_out_viz_base: str | os.PathLike
    :param ds: The dataset name
    :type ds: str
    :param cstm_str: The option to add in a custom string such as the plot type, defaults to None, defaults to None
    :type cstm_str: str, optional
    :return: The PCA plot filepath
    :rtype: pathlib.PosixPath
    """
    if cstm_str:
        path_pca_plot = Path(f"{dir_out_viz_base}/{ds}/correlation_matrix_{ds}_{cstm_str}.png")
    else:
        path_pca_plot = Path(f"{dir_out_viz_base}/{ds}/correlation_matrix_{ds}.png")
    path_pca_plot.parent.mkdir(parents=True,exist_ok=True)

    return path_pca_plot


def plot_pca_save_wrap(df_X:pd.DataFrame, 
                        dir_out_viz_base:str|Path,
                        ds:str, 
                        std_scale:bool=True)->PCA:
    """Wrapper function to generate PCA plots on dataset

    :param df_X: The attribute dataset of interest
    :type df_X: pd.DataFrame
    :param dir_out_viz_base: Standardized output directory for visualization
    :type dir_out_viz_base: str | os.PathLike
    :param ds: The dataset name
    :type ds: str
    :param std_scale: Should dataset be standardized using StandardScaler, defaults to True
    :type std_scale: bool, optional
    :return: The principal components analysis object
    :rtype: PCA
    """
    # CREATE THE EXPLAINED VARIANCE RATIO PLOT
    cstm_str = ''
    if std_scale:
        cstm_str = 'std_scaled'
    pca_scaled = pca_stdscaled_tfrm(df_X,std_scale)
    fig_pca_stdscale = plot_pca_stdscaled_tfrm(pca_scaled)
    path_pca_stdscaled_fig = std_pca_plot_path(dir_out_viz_base,ds,cstm_str=cstm_str)
    fig_pca_stdscale.savefig(path_pca_stdscaled_fig)
    logging.info(f"Wrote the {ds} PCA explained variance ratio plot to\n{path_pca_stdscaled_fig}")
    plt.clf()
    plt.close()
    # CREATE THE CUMULATIVE VARIANCE PLOT
    cstm_str_cum = 'cumulative_var'
    if std_scale:
        cstm_str_cum = 'cumulative_var_std_scaled'
    path_pca_stdscaled_cum_fig = std_pca_plot_path(dir_out_viz_base,ds,cstm_str=cstm_str_cum)
    fig_pca_cumulative = plot_pca_stdscaled_cumulative_var(pca_scaled)
    fig_pca_cumulative.savefig(path_pca_stdscaled_cum_fig)
    logging.info(f"Wrote the {ds} PCA cumulative variance explained plot to\n{path_pca_stdscaled_cum_fig}")
    plt.clf()
    plt.close('all')
    gc.collect()
    return None

def plot_feature_importance(feat_imprt:np.ndarray, attrs:Iterable[str],
                        title:str) -> Figure:
    """Generate the feature importance plot

    :param feat_imprt: Feature importance array from `rfr.feature_importances_`
    :type feat_imprt: np.ndarray
    :param attrs: The catchment attributes of interest
    :type attrs: Iterable[str]
    :param title: The feature importance plot title
    :type title: str
    :return: The feature importance plot
    :rtype: Figure
    """
    df_feat_imprt = pd.DataFrame({'attribute': attrs,
                                'importance': feat_imprt}).sort_values(by='importance', ascending=False)
    # Calculate the correlation matrix
    plt.figure(figsize=(10,6))
    plt.barh(df_feat_imprt['attribute'], df_feat_imprt['importance'])
    plt.xlabel('Importance')
    plt.ylabel('Attribute')
    plt.title(title)

    fig = plt.gcf()
    return fig

def save_feat_imp_fig_wrap(model:any,
                           attrs: Iterable[str],
                           dir_out_viz_base:str|Path,
                           ds:str, metr:str, algo_str:str):
    """Wrapper to generate & save to file the feature importance plot

    :param model: The trained regressor/classifier object containing .feature_importances_
    :type model: any
    :param attrs: The attributes 
    :type attrs: Iterable[str]
    :param dir_out_viz_base: The standard output base directory for visualizations
    :type dir_out_viz_base: str | os.PathLike
    :param ds: The unique dataset name
    :type ds: str
    :param metr: The metric/response variable of interest
    :type metr: str
    :param algo_str: The algorithm string identifier (e.g. 'xgb', 'rf')
    :type algo_str: str
    """ 
    # Changelog/contributions
    #     FY25 - originally created, GL
    #     2026-08-14 refactor: generalize to other algoirthm strings, Gemni3.1Pro

    feat_imprt = getattr(model, "feature_importances_", None)
    
    if feat_imprt is None:
        logging.warning(f"Model {algo_str} does not contain .feature_importances_")
        return
        
    # Generate dynamic title
    title_imp = f"{algo_str.upper()} feature importance of {metr}: {ds}"
    
    # Use the renamed core plotting function
    fig_feat_imp = plot_feature_importance(feat_imprt, attrs=attrs, title=title_imp)

    # Generate dynamic save path (e.g., 'xgb_feature_importance_...')
    path_fig_imp = std_feat_imp_path(dir_out_viz_base=dir_out_viz_base,
                      ds=ds, algo_str=algo_str, metr=metr)
    path_fig_imp.parent.mkdir(parents=True, exist_ok=True)

    fig_feat_imp.savefig(path_fig_imp)
    logging.info(f"Wrote feature importance plot to {path_fig_imp}")
    
    plt.close(fig_feat_imp)
    plt.close('all')
    gc.collect()

def std_feat_imp_path(dir_out_viz_base: str|Path,
                      ds:str, algo_str:str, metr:str):
    path_fig_imp = Path(dir_out_viz_base) / ds / f"{algo_str}_feature_importance_{ds}_{metr}.png"
    path_fig_imp.parent.mkdir(parents=True, exist_ok=True)
    return path_fig_imp

def std_lc_plot_path(dir_out_viz_base: str|Path,
                      ds:str, metr:str, algo_str:str
                      ) -> pathlib.PosixPath:

    path_lc_plot = Path(f"{dir_out_viz_base}/{ds}/learning_curve_{ds}_{metr}_{algo_str}.png")
    path_lc_plot.parent.mkdir(parents=True,exist_ok=True)
    return path_lc_plot

def std_regr_pred_obs_path(dir_out_viz_base:str|Path, ds:str,
                            metr:str,algo_str:str,
                            split_type:str='') -> pathlib.PosixPath:
    """Generate a filepath of the predicted vs observed regresion plot

    :param dir_out_viz_base: The base directory for saving plots
    :type dir_out_viz_base: str | os.PathLike
    :param ds: The unique dataset name
    :type ds: str
    :param metr: The metric/response variable of interest
    :type metr: str
    :param algo_str: The type of algorithm used to create predictions
    :type algo_str: str
    :param split_type: The type of data being displayed (e.g. training, testing), defaults to ''
    :type split_type: str, optional
    :return: The path to save the regression of predicted vs observed values.
    :rtype: pathlib.PosixPath
    """

    path_regr_pred_plot = Path(f"{dir_out_viz_base}/{ds}/regr_pred_obs_{ds}_{metr}_{algo_str}_{split_type}.png")
    path_regr_pred_plot.parent.mkdir(parents=True,exist_ok=True)
    return path_regr_pred_plot

def std_regr_pred_obs_path_mapie(dir_out_viz_base:str|Path, ds:str,
                            metr:str,algo_str:str,alpha_val:float,
                            split_type:str='') -> pathlib.PosixPath:
    """Generate a filepath of the predicted vs observed regresion plot

    :param dir_out_viz_base: The base directory for saving plots
    :type dir_out_viz_base: str | os.PathLike
    :param ds: The unique dataset name
    :type ds: str
    :param metr: The metric/response variable of interest
    :type metr: str
    :param algo_str: The type of algorithm used to create predictions
    :type algo_str: str
    :param split_type: The type of data being displayed (e.g. training, testing), defaults to ''
    :type split_type: str, optional
    :return: The path to save the regression of predicted vs observed values.
    :rtype: pathlib.PosixPath
    """

    path_regr_pred_plot = Path(f"{dir_out_viz_base}/{ds}/regr_pred_obs_{ds}_{metr}_{algo_str}_{split_type}_alpha{alpha_val}.png")
    path_regr_pred_plot.parent.mkdir(parents=True,exist_ok=True)
    return path_regr_pred_plot

def _estimate_decimals_for_plotting(val:float)-> int:
    """Determine how many decimals should be used when rounding
    :param val: The value of interest for rounding
    :type val: np.float
    :return: The number of decimal places to round to
    :rtype: int
    """

    fmt_positional = np.format_float_positional(val)
    round_decimals = 2
    if fmt_positional[0:2] == '0.':
        sub_fmt_positional = fmt_positional[2:]
        count = 0
        for char in sub_fmt_positional:
            if char == '0':
                count += 1
            else:
                round_decimals = count+3
                break

    return round_decimals

def plot_pred_vs_obs_regr(y_pred: np.ndarray, y_obs: np.ndarray, ds:str, metr:str, 
                          r2_val:float=None,spearman_val:float=None)->Figure:
    """Plot the observed vs. predicted module performance

    :param y_pred: The predicted response variable
    :type y_pred: np.ndarray
    :param y_obs: The observed response variable
    :type y_obs: np.ndarray
    :param ds: The unique dataset name
    :type ds: str
    :param metr: The metric/response variable name of interest
    :type metr: str
    :param r2_val: The r-squared value from regression
    :type r2_val: float
    :param spearman_val: The spearman rank correlation coefficient
    :type spearman_val: float
    :return: The predicted vs observed regression plot
    :rtype: Figure
    """
    max_val = np.max([y_pred,y_obs])
    tot_rnd_max = _estimate_decimals_for_plotting(max_val)
    min_val = np.min([y_pred,y_obs])
    tot_rnd_min = _estimate_decimals_for_plotting(min_val)
    tot_rnd = np.max([tot_rnd_max,tot_rnd_min])
    min_val_rnd = np.round(np.min([min_val,0]),tot_rnd)
    max_val_rnd = np.round(max_val,tot_rnd)
    min_vals = (min_val_rnd,min_val_rnd)
    max_vals = (max_val_rnd,max_val_rnd)

    # Adapted from plot in bolotinl's rafts_perf_viz.py
    plt.scatter(x=y_obs,y=y_pred,alpha=0.3)
    plt.axline(min_vals, max_vals, color='black', linestyle='--')
    plt.ylabel('Predicted {}'.format(metr))
    plt.xlabel('Actual {}'.format(metr))
    plt.title('RaFTS-predicted vs. observed: {}'.format(ds))

    metrics_text = []
    if r2_val is not None:
        metrics_text.append(f'$R^2 = {r2_val:.2f}$')

    if spearman_val is not None:
        metrics_text.append(f'$\\rho = {spearman_val:.2f}$')

    if metrics_text:
        plt.gca().text(0.05, 0.95, '\n'.join(metrics_text), transform=plt.gca().transAxes,
                       fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

    fig = plt.gcf()
    return fig

def plot_pred_vs_obs_wrap(y_pred: np.ndarray, y_obs:np.ndarray, dir_out_viz_base:str|Path,
                           ds:str, metr:str, algo_str:str, r2_val:float=None,
                           split_type:str='',spearman_val:float=None):
    """Wrapper to create & save predicted vs. observed regression plot

    :param y_pred: The predicted response variable
    :type y_pred: np.ndarray
    :param y_obs: The observed response variable
    :type y_obs: np.ndarray
    :param dir_out_viz_base: The base directory for saving plots
    :type dir_out_viz_base: str | os.PathLike
    :param ds: The unique dataset name
    :type ds: str
    :param metr: The metric/response variable name of interest
    :type metr: str
    :param algo_str: The type of algorithm used to create predictions
    :type algo_str: str
    :param split_type: The type of data being displayed (e.g. training, testing), defaults to ''
    :type split_type: str, optional
    :param r2_val: The r-squared value from regression
    :type r2_val: float
    :param spearman_val: The spearman rank correlation coefficient
    :type spearman_val: float
    """
    # Generate figure
    fig_regr = plot_pred_vs_obs_regr(y_pred, y_obs, ds, metr, r2_val,spearman_val)
    # Generate filepath for saving figure
    path_regr_plot = std_regr_pred_obs_path(dir_out_viz_base, ds,
                            metr,algo_str,split_type)
    # Save the plot as a .png file
    fig_regr.savefig(path_regr_plot, dpi=300, bbox_inches='tight')
    plt.close(fig_regr)
    plt.close('all')
    gc.collect()

def plot_pred_vs_obs_regr_mapie(y_pred: np.ndarray, y_obs: np.ndarray, ds:str,
                                metr:str, y_pis: list, alpha_val:float,
                                r2_val:float=None,spearman_val:float=None)->Figure:
    """Plot the observed vs. predicted module performance

    :param y_pred: The predicted response variable
    :type y_pred: np.ndarray
    :param y_obs: The observed response variable
    :type y_obs: np.ndarray
    :param ds: The unique dataset name
    :type ds: str
    :param metr: The metric/response variable name of interest
    :type metr: str
    :param y_pis: y prediction intervals
    :type y_pis: list
    :param alpha_val: The alpha value for the prediction intervals
    :type alpha_val: float
    :param r2_val: The r-squared value from regression
    :type r2_val: float
    :param spearman_val: The spearman rank correlation coefficient
    :type spearman_val: float
    :return: The predicted vs observed regression plot
    :rtype: Figure
    """
    max_val = np.max([y_pred,y_obs])
    tot_rnd_max = _estimate_decimals_for_plotting(max_val)
    min_val = np.min([y_pred,y_obs])
    tot_rnd_min = _estimate_decimals_for_plotting(min_val)
    tot_rnd = np.max([tot_rnd_max,tot_rnd_min])
    min_val_rnd = np.round(np.min([min_val,0]),tot_rnd)
    max_val_rnd = np.round(max_val,tot_rnd)
    min_vals = (min_val_rnd,min_val_rnd)
    max_vals = (max_val_rnd,max_val_rnd)

    # Extract error bars (lower and upper limits) for the first alpha value
    lower_err = np.abs(y_pred - np.array([y_pis[i].loc['lower_limit', f'alpha_{alpha_val:.2f}'] for i in range(len(y_pred))]))
    upper_err = np.abs(np.array([y_pis[i].loc['upper_limit', f'alpha_{alpha_val:.2f}'] for i in range(len(y_pred))]) - y_pred)
    
    # Adapted from plot in bolotinl's rafts_perf_viz.py
    plt.errorbar(y_obs, y_pred, yerr=[lower_err, upper_err], fmt='o', alpha=0.3, ecolor='gray', capsize=3)
    plt.axline(min_vals, max_vals, color='black', linestyle='--')
    plt.ylabel('Predicted {}'.format(metr))
    plt.xlabel('Actual {}'.format(metr))
    plt.title('RaFTS-predicted vs. observed: {}'.format(ds))

    # Add alpha values as text box
    plt.gca().text(0.05, 0.95, f'alpha = {alpha_val:.2f}', transform=plt.gca().transAxes,
                   fontsize=10, verticalalignment='bottom', horizontalalignment='right',
                   bbox=dict(facecolor='white', alpha=0.5))
    metrics_text = []
    if r2_val is not None:
        metrics_text.append(f'$R^2 = {r2_val:.2f}$')

    if spearman_val is not None:
        metrics_text.append(f'$\\rho = {spearman_val:.2f}$')

    if metrics_text:
        plt.gca().text(0.05, 0.95, '\n'.join(metrics_text), transform=plt.gca().transAxes,
                       fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
    fig = plt.gcf()
    return fig

def plot_pred_vs_obs_wrap_mapie(y_pred: np.ndarray, y_obs:np.ndarray, dir_out_viz_base:str|Path,
                           ds:str, metr:str, algo_str:str,y_pis: list, alpha_val:float,
                           split_type:str='',r2_val:float=None,spearman_val:float=None):
    """Wrapper to create & save predicted vs. observed regression plot

    :param y_pred: The predicted response variable
    :type y_pred: np.ndarray
    :param y_obs: The observed response variable
    :type y_obs: np.ndarray
    :param dir_out_viz_base: The base directory for saving plots
    :type dir_out_viz_base: str | os.PathLike
    :param ds: The unique dataset name
    :type ds: str
    :param metr: The metric/response variable name of interest
    :type metr: str
    :param algo_str: The type of algorithm used to create predictions
    :type algo_str: str
    :param split_type: The type of data being displayed (e.g. training, testing), defaults to ''
    :type split_type: str, optional
    :param r2_val: The r-squared value from regression
    :type r2_val: float
    :param spearman_val: The spearman rank correlation coefficient
    :type spearman_val: float
    """
    # Generate figure
    fig_regr = plot_pred_vs_obs_regr_mapie(y_pred, y_obs, ds, metr, y_pis, alpha_val, r2_val,spearman_val)
    # Generate filepath for saving figure
    path_regr_plot = std_regr_pred_obs_path_mapie(dir_out_viz_base, ds,
                            metr,algo_str,alpha_val,split_type)
    # Save the plot as a .png file
    fig_regr.savefig(path_regr_plot, dpi=300, bbox_inches='tight')
    plt.close(fig_regr)
    plt.close('all')
    gc.collect()

def std_map_pred_path(dir_out_viz_base:str|Path, ds:str,
                      metr:str,algo_str:str,
                      split_type:str='') -> pathlib.PosixPath:
    """Generate a filepath of the predicted response variables map:

    :param dir_out_viz_base: The base directory for saving plots
    :type dir_out_viz_base: str | os.PathLike
    :param ds: The unique dataset name
    :type ds: str
    :param metr: The metric/response variable of interest
    :type metr: str
    :param algo_str: The type of algorithm used to create predictions
    :type algo_str: str
    :param split_type: The type of data being displayed (e.g. training, testing), defaults to ''
    :type split_type: str, optional
    :return: _description_
    :rtype: pathlib.PosixPath
    """
    
    # 
    path_pred_map_plot = Path(f"{dir_out_viz_base}/{ds}/prediction_map_{ds}_{metr}_{algo_str}_{split_type}.png")
    path_pred_map_plot.parent.mkdir(parents=True,exist_ok=True)
    return path_pred_map_plot

def gen_conus_basemap(dir_out_basemap:str | Path, # This should be the data_visualizations directory
                    url:str = 'https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_state_500k.zip',
                    fn_basemap:str='cb_2018_us_state_500k.shp') -> gpd.geodataframe.GeoDataFrame:
    """Retrieve the basemap for CONUS

    :param dir_out_basemap: The standard directory for saving the CONUS basemap
    :type dir_out_basemap: str | os.PathLike
    :param url: The url of a basemap of interest
    :type url: str
    :param fn_basemap: The filename to use for saving basemap, defaults to 'cb_2018_us_state_500k.shp'
    :type fn_basemap: str, optional
    :return: The geopandas dataframe of the basemap
    :rtype: gpd.geodataframe.GeoDataFrame
    """
    # Changelog / contributions:
    #     2024 Originally created
    #     2025-09-22, changed from using urlib to using requests to avoid SSL error, GL

    path_zip_basemap = f'{dir_out_basemap}/cb_2018_us_state_500k.zip'
    path_shp_basemap = f'{dir_out_basemap}/{fn_basemap}'

    if not Path(path_zip_basemap).exists():
        logging.info("Downloading shapefile...")
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        session.mount('https://', HTTPAdapter(max_retries=retries))
        response = session.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        response.raise_for_status()
        with open(path_zip_basemap, "wb") as out_file:
            out_file.write(response.content)
        logging.info("Shapefile downloaded.")

    if not Path(path_shp_basemap).exists():
        with zipfile.ZipFile(path_zip_basemap, "r") as zip_ref:
            zip_ref.extractall(path_shp_basemap)

    states = gpd.read_file(path_shp_basemap)
    states = states.to_crs("EPSG:4326")

    return states
    
def plot_map_pred(geo_df:gpd.GeoDataFrame, states:gpd.GeoDataFrame,
                  title:str,metr:str,colname_data:str='prediction',
                  plot_style:str='auto', task_type:str='regression',
                  gdf_missing:gpd.GeoDataFrame=None
                  )->Figure:
    # Calculate vmin and vmax based on the data
    vmin = geo_df[colname_data].min(skipna=True)
    vmax = geo_df[colname_data].max(skipna=True)

    fig, ax = plt.subplots(1, 1, figsize=(20, 24))

    # Determine plot style
    if plot_style == 'auto':
        plot_style = 'hexbin' if geo_df.shape[0] > 20000 else 'points'
    map_msg = f"{title} :\n \
        - Plotting {geo_df.shape[0]} locations on map with {plot_style} style and {task_type} task_type"
    logging.info(map_msg)
    print(map_msg)

    # --- 0. "NO DATA" LAYER (real polygons, drawn first / lowest zorder) ---
    # Rows that have a real geometry but no prediction reached them (e.g. a
    # crosswalk gap upstream) get their own visually distinct fill rather than
    # being left out of geo_df entirely -- an area with genuinely no data
    # should never look identical to an area that just happens to fall in a
    # gap of the hexbin/point rendering below.
    if gdf_missing is not None and not gdf_missing.empty:
        logging.info(f"Rendering {len(gdf_missing)} 'no crosswalk data' divides as a distinct hatched layer.")
        gdf_missing.plot(ax=ax, facecolor='lightgray', hatch='///', edgecolor='dimgray',
                          linewidth=0.2, alpha=0.6, zorder=1.2, label='No crosswalk data')

    # --- 1a. CATEGORICAL PLOTTING (any size, clustering) ---
    points_legend_built = False
    if task_type == 'clustering':
        # Originally routed large (>20000-row) clustering datasets through
        # ax.hexbin, then through geo_df.dissolve(by=colname_data) -- both
        # replaced by this direct per-geometry categorical plot:
        #
        # hexbin bins *centroids*, not polygon area: a hexagon is only drawn
        # if at least one divide's centroid falls inside it, so coverage
        # tracks centroid density, not the ground the divides actually tile.
        # Invisible almost everywhere (network-type divides are small and
        # dense enough that every hexagon holds dozens of centroids), but
        # hydrofabric v4's closed-basin divides (type='landscape', no
        # flowline) are enormous by comparison -- confirmed empirically: 52
        # are each larger than a whole hexagon at gridsize=150, the biggest
        # spanning ~10, so 9 of those 10 never receive a centroid and render
        # blank despite having a real, valid cluster prediction.
        #
        # dissolve(by=colname_data) is areally complete (fixes the above),
        # but its internal GEOS union_all() is (a) slow at CONUS scale --
        # benchmarked at 43.5s for 80,000 real hydrofabric polygons merged
        # into 8 groups, vs. 4.8s for this direct per-geometry plot on the
        # same input, ~9x faster and the gap widens with polygon count since
        # union cost is superlinear -- and (b) intolerant of invalid
        # geometry in a way individual-polygon rendering never was: hit
        # GEOSException: TopologyException on the real CONUS master GPKG,
        # which make_valid() could patch but the union step itself remains
        # both the slowest part of this function and an unnecessary one --
        # dissolving to fewer polygons only reduces the number of draw calls,
        # which was never the bottleneck geopandas/matplotlib needed help
        # with at this scale.
        #
        # Plotting each geometry with its own fill color, keyed categorically
        # off colname_data, needs no union at all: it's areally complete by
        # construction (same as dissolve) and immune to topology errors (same
        # as the old hexbin/points paths), while being the fastest of the three.
        logging.info(f"Using categorical mapping for cluster labels ({len(geo_df)} features).")
        plot_kwargs = {}
        if geo_df.geometry.geom_type.iloc[0] in ('Point', 'MultiPoint'):
            plot_kwargs['markersize'] = 150
        geo_df.plot(column=colname_data, ax=ax, categorical=True, cmap='tab20',
                    legend=True, zorder=2, **plot_kwargs)

        # Customize the discrete legend
        legend = ax.get_legend()
        if legend:
            legend.set_title("Clusters", prop={'size': 24})
            for text in legend.get_texts():
                text.set_fontsize(20)

            if gdf_missing is not None and not gdf_missing.empty:
                # A second, independent legend for the "no data" patch --
                # NOT folded into the one above via get_legend_handles_labels()
                # + a rebuilt ax.legend() call: geopandas' categorical .plot()
                # legend handles are a single PatchCollection, which
                # matplotlib's Legend doesn't render per-category from when
                # reconstructed that way (silently produced a blank/incorrect
                # legend here during development). ax.add_artist() keeps this
                # first legend intact while a second ax.legend() call adds
                # the "no data" entry alongside it instead of replacing it.
                ax.add_artist(legend)
                missing_patch = mpatches.Patch(facecolor='lightgray', hatch='///', edgecolor='dimgray',
                                                label='No crosswalk data')
                ax.legend(handles=[missing_patch], prop={'size': 20}, loc='lower left')
                points_legend_built = True

    # --- 1b. HEXBIN PLOTTING (large regression datasets) ---
    elif plot_style == 'hexbin':
        logging.info(f"Using hexbin mapping for large dataset ({len(geo_df)} points).")

        hb = ax.hexbin(
            x=geo_df.geometry.centroid.x,
            y=geo_df.geometry.centroid.y,
            C=geo_df[colname_data],
            reduce_C_function=np.mean,
            gridsize=150,
            cmap='viridis',
            vmin=vmin, vmax=vmax,
            zorder=2, alpha=0.9, edgecolors='none'
        )
        cbar_mappable = hb

    # --- 2. POINTS PLOTTING (small regression datasets) ---
    else:
        logging.info("Using continuous point mapping.")
        ms = 150 if len(geo_df) < 10000 else max(0.5, 500000 / len(geo_df))
        geo_df.plot(column=colname_data, ax=ax, markersize=ms, cmap='viridis', legend=False, zorder=2)
        cbar_mappable = plt.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax), cmap='viridis')

    # Plot states boundary once for both styles
    states.boundary.plot(ax=ax, color="#555555", linewidth=1, zorder=1, alpha=0.5)

    # Standalone legend for the "no crosswalk data" patch when it wasn't already
    # folded into the clustering legend above (regression hexbin uses a
    # colorbar instead of a legend, and regression points plotting doesn't
    # build a legend at all, so neither has anywhere to fold this patch into).
    if gdf_missing is not None and not gdf_missing.empty and not points_legend_built:
        missing_patch = mpatches.Patch(facecolor='lightgray', hatch='///', edgecolor='dimgray', label='No crosswalk data')
        ax.legend(handles=[missing_patch], prop={'size': 20}, loc='lower right')

    # Formatting
    ax.tick_params(axis='x', labelsize= 24)
    ax.tick_params(axis='y', labelsize= 24)
    plt.xlabel('Longitude',fontsize = 26) 
    plt.ylabel('Latitude',fontsize = 26)  
    
    # Only draw the continuous colorbar for regression tasks
    if task_type != 'clustering':
        cbar_ax = plt.colorbar(cbar_mappable, ax=ax,fraction=0.02, pad=0.04)
        cbar_ax.set_label(label=metr,size=24)
        cbar_ax.ax.tick_params(labelsize=24) 
    
    plt.title(title, fontsize = 28)
    bounds = geo_df.total_bounds
    if gdf_missing is not None and not gdf_missing.empty:
        # Extend the view to cover the "no crosswalk data" divides too -- a
        # region that's entirely missing from geo_df (e.g. an area with zero
        # successful crosswalk matches) would otherwise get cropped out of the
        # frame regardless of the hatched layer drawn for it above.
        missing_bounds = gdf_missing.total_bounds
        bounds = np.array([
            min(bounds[0], missing_bounds[0]), min(bounds[1], missing_bounds[1]),
            max(bounds[2], missing_bounds[2]), max(bounds[3], missing_bounds[3]),
        ])
    x_buffer = (bounds[2] - bounds[0]) * 0.05
    y_buffer = (bounds[3] - bounds[1]) * 0.05
    if x_buffer == 0: x_buffer = 1.0 
    if y_buffer == 0: y_buffer = 1.0
    ax.set_xlim(bounds[0] - x_buffer, bounds[2] + x_buffer)
    ax.set_ylim(bounds[1] - y_buffer, bounds[3] + y_buffer)
    fig = plt.gcf()

    return fig

def plot_map_pred_wrap(test_gdf:gpd.GeoDataFrame,
                       dir_out_viz_base:str | os.PathLike,
                      ds:str,
                      metr:str,algo_str:str,
                      split_type:str='test',
                      colname_data:str='prediction',
                      epsg_reproj:int=3857,
                      task_type:str='regression',
                      gdf_missing:gpd.GeoDataFrame=None):
    """Wrapper for plotting map displays

    :param test_gdf: The geodataframe to plotting on maps
    :type test_gdf: gpd.GeoDataFrame
    :param dir_out_viz_base: The base directory for storing visualization data
    :type dir_out_viz_base: str|os.PathLike
    :param ds: The dataset name
    :type ds: str
    :param metr: The response variable of interest
    :type metr: str
    :param algo_str: The algorithm shortstring
    :type algo_str: str
    :param split_type: The type of data, either algo testing or actual prediction, defaults to 'test'
    :type split_type: str, optional
    :param colname_data: The standard colname for data values in `test_gdf`, defaults to 'prediction'
    :type colname_data: str, optional
    :param epsg_reproj: The EPSG code for reprojecting data for map display, defaults to 3857
    :type epsg_reproj: int
    :param gdf_missing: Rows with real geometry but no prediction (e.g. dropped by an
        upstream crosswalk gap), rendered as a distinct hatched 'no data' layer instead
        of being silently absent from the map. Defaults to None.
    :type gdf_missing: gpd.GeoDataFrame, optional
    """

    path_pred_map_plot = std_map_pred_path(dir_out_viz_base,ds,metr,algo_str,split_type)
    dir_out_basemap = path_pred_map_plot.parent.parent
    states = gen_conus_basemap(dir_out_basemap = dir_out_basemap)

    # TODO add option to print oconus basemap
    # TODO change epsg_reproj for oconus basemap

    # Ensure the gdf matches the 4326 epsg used for states:
    test_gdf = test_gdf.to_crs(4326)

    # Re-project for visualization
    states = states.to_crs(epsg=epsg_reproj)
    geo_df = test_gdf.to_crs(epsg=epsg_reproj)
    # NOTE: geometry.is_empty is True only for a real-but-zero-area geometry
    # (e.g. Polygon()) -- it returns False for a None/missing geometry, which
    # is a *different* condition (geometry.isna()). A None geometry that
    # slipped through here would silently produce a NaN centroid, which
    # matplotlib's hexbin just drops from the bin count with no warning at
    # all -- catch both so nothing vanishes from the map without a log line.
    invalid_geom_mask = geo_df.geometry.is_empty | geo_df.geometry.isna()
    geo_df_valid = geo_df[~invalid_geom_mask]
    if geo_df_valid.shape[0] < geo_df.shape[0]:
        logging.warning(f"Lost a total {geo_df.shape[0]-geo_df_valid.shape[0]} \
                         of {geo_df.shape[0]} data rows due to invalid/missing geometries")
        if colname_data in geo_df_valid.columns:
            geo_df_valid = geo_df_valid.dropna(subset=[colname_data])
        geo_df_valid = geo_df_valid.reset_index(drop=True)
        geo_df = geo_df_valid.copy()

    if gdf_missing is not None and not gdf_missing.empty:
        gdf_missing = gdf_missing.to_crs(4326).to_crs(epsg=epsg_reproj)
        gdf_missing = gdf_missing[~(gdf_missing.geometry.is_empty | gdf_missing.geometry.isna())]

    # Generate the map
    plot_title = f"Predicted Values: {metr} - {ds}: {algo_str} algorithm"
    plot_pred_map = plot_map_pred(geo_df=geo_df, states=states,title=plot_title,
                                  metr=metr,colname_data=colname_data,
                                  task_type=task_type, gdf_missing=gdf_missing)

    # Save the plot as a .png file
    plot_pred_map.savefig(path_pred_map_plot, dpi=300, bbox_inches='tight')
    logging.info(f"Wrote prediction map to \n{path_pred_map_plot}")
    plt.close(plot_pred_map)
    plt.close('all')
    gc.collect()

def plot_map_pred_uncn(geo_df: gpd.GeoDataFrame, states: gpd.GeoDataFrame, title: str, metr: str,
                       alpha_val: float = None, uncn_col: str = None,
                       colname_data: str = 'prediction'):
    """Generate a map where color = prediction value, and size = uncertainty variance/spread."""
    
    # 1. Fetch errors dynamically based on the requested uncertainty method
    if alpha_val is not None:
        err_dict = raftsutil.infer_mapie_errors(geo_df, alpha_val, colname_data)
        uncn_series = err_dict['total_err']
        min_err, max_err = err_dict['min_err'], err_dict['max_err']
        legend_title = f"{(1 - alpha_val) * 100:.0f}% MAPIE CI Range"
    elif uncn_col is not None and uncn_col in geo_df.columns:
        uncn_series = geo_df[uncn_col]
        min_err, max_err = uncn_series.min(), uncn_series.max()
        legend_title = "ForestCI Variance"
    else:
        raise ValueError("Must provide either a valid alpha_val for MAPIE or a valid uncn_col for ForestCI.")

    # 2. Normalize marker size (scale from 100 to 400)
    if max_err > min_err:
        marker_sizes = 100 + 300 * (uncn_series - min_err) / (max_err - min_err)
    else:
        # Fallback if all points have the exact same uncertainty
        marker_sizes = pd.Series(200, index=geo_df.index)

    # 3. Plotting Setup
    fig, ax = plt.subplots(1, 1, figsize=(20, 24))
    states.boundary.plot(ax=ax, color="#555555", linewidth=1)
    
    # 4. Plot the data: Color by prediction, Size by Uncertainty
    geo_df.plot(column=colname_data, ax=ax, markersize=marker_sizes, cmap='viridis', legend=False, zorder=2)
    states.boundary.plot(ax=ax, color="#555555", linewidth=1, zorder=1)
    
    # 5. Add Colorbar for the Prediction Value
    vmin, vmax = geo_df[colname_data].min(), geo_df[colname_data].max()
    cbar = plt.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax), cmap='viridis')
    cbar_ax = plt.colorbar(cbar, ax=ax, fraction=0.02, pad=0.04)
    cbar_ax.set_label(label=f"Predicted {metr}", size=24)
    cbar_ax.ax.tick_params(labelsize=24)

    plt.title(title, fontsize=28)

    # 6. Dynamic Map Bounds (with 5% buffer)
    bounds = geo_df.total_bounds
    x_buffer = (bounds[2] - bounds[0]) * 0.05
    y_buffer = (bounds[3] - bounds[1]) * 0.05
    if x_buffer == 0: x_buffer = 1.0 
    if y_buffer == 0: y_buffer = 1.0
    ax.set_xlim(bounds[0] - x_buffer, bounds[2] + x_buffer)
    ax.set_ylim(bounds[1] - y_buffer, bounds[3] + y_buffer)

    # 7. Scale Legend for the Uncertainty Circles
    legend_handles = [
        plt.scatter([], [], s=100, color='gray', label=f'{min_err:.2f}'),
        plt.scatter([], [], s=400, color='gray', label=f'{max_err:.2f}')
    ]
    ax.legend(handles=legend_handles, title=legend_title, 
              loc='lower left', fontsize=20, title_fontsize=22)

    return plt.gcf()

def plot_map_pred_wrap_uncn(test_gdf, dir_out_viz_base, ds, metr, algo_str,
                            alpha_val: float = None, uncn_col: str = None,
                            split_type='test', colname_data='prediction',
                            epsg_reproj: int = 3857):
    """Wrapper to handle file I/O, reprojection, and plotting for uncertainty maps."""
    
    # Standardize output filenames
    path_pred_map_plot = std_map_pred_path(dir_out_viz_base, ds, metr, algo_str, split_type)
    if alpha_val is not None:
        new_filename = path_pred_map_plot.stem + f"_mapie_alpha{alpha_val:.2f}" + path_pred_map_plot.suffix
    else:
        new_filename = path_pred_map_plot.stem + f"_{uncn_col}" + path_pred_map_plot.suffix
    path_uncn_plot = path_pred_map_plot.with_name(new_filename)
    
    dir_out_basemap = path_pred_map_plot.parent.parent
    states = gen_conus_basemap(dir_out_basemap=dir_out_basemap)

    # Safely Reproject
    test_gdf = test_gdf.to_crs(4326)
    states = states.to_crs(epsg=epsg_reproj)
    geo_df = test_gdf.to_crs(epsg=epsg_reproj)

    plot_title = f"Predicted Values and Uncertainty: {metr} - {ds}"
    plot_pred_map = plot_map_pred_uncn(geo_df=geo_df, states=states, title=plot_title,
                                       metr=metr, alpha_val=alpha_val, uncn_col=uncn_col,
                                       colname_data=colname_data)

    plot_pred_map.savefig(path_uncn_plot, dpi=300, bbox_inches='tight')
    logging.info(f"Wrote uncertainty map to \n{path_uncn_plot}")
    plt.close(plot_pred_map)
    plt.close('all')
    gc.collect()

def plot_best_perf_map(geo_df,states, title, comparison_col = 'dataset'):

    """Generate a map of the best-predicted response variables as determined from multiple datasets

    :param geo_df: Geodataframe of response variable results
    :type geo_df: gpd.GeoDataFrame
    :param states: The states basemap
    :type states: gpd.GeoDataFrame
    :param title: Map title
    :type title: str
    :param comparison_col: The geo_df column name representing data of interest, defaults to 'dataset'
    :type comparison_col: str, optional
    :return: Map of best-predicted response variables
    :rtype: Figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(20, 24))
    base = states.boundary.plot(ax=ax, color="#555555", linewidth=1)


    # Plot points based on the 'best_algo' column
    geo_df.plot(column=comparison_col, ax=ax, markersize=150, cmap='viridis', legend=True,zorder=2)

    # Plot states boundary again with lower zorder
    states.boundary.plot(ax=ax, color="#555555", linewidth=1, zorder=1)

    # Set title and axis limits
    plt.title(title, fontsize=28)
    # Dynamically bound the map to the data points rather than the whole US basemap
    bounds = geo_df.total_bounds  # Returns [minx, miny, maxx, maxy]
    # Calculate a 5% spatial buffer so edge points aren't cut off by the plot borders
    x_buffer = (bounds[2] - bounds[0]) * 0.05
    y_buffer = (bounds[3] - bounds[1]) * 0.05
    # Fallback just in case all points share the exact same X or Y coordinate
    if x_buffer == 0: x_buffer = 1.0 
    if y_buffer == 0: y_buffer = 1.0
    ax.set_xlim(bounds[0] - x_buffer, bounds[2] + x_buffer)
    ax.set_ylim(bounds[1] - y_buffer, bounds[3] + y_buffer)

    # Customize the legend, specifically for the geo_df plot
    legend = ax.get_legend()
    if legend:
        legend.set_title("Formulations", prop={'size': 20})
        for text in legend.get_texts():
            text.set_fontsize(20)

    fig = plt.gcf()
    return fig

def std_map_best_path(dir_out_viz_base:str|Path,metr:str,ds:str
                      )->pathlib.PosixPath:
    """# Generate a filepath of the best-performing dataset map:

    :param dir_out_viz_base: _description_
    :type dir_out_viz_base: str | os.PathLike
    :param metr: The metric/response variable of interest
    :type metr: str
    :param ds: The unique dataset of interest
    :type ds: str
    :return: Path to the map figure in png
    :rtype: pathlib.PosixPath
    """
    
    path_best_map_plot = Path(f"{dir_out_viz_base}/{ds}/performance_map_best_formulation_{metr}.png")
    path_best_map_plot.parent.mkdir(parents=True,exist_ok=True)
    return path_best_map_plot


def plot_best_algo_wrap(geo_df, dir_out_viz_base,subdir_anlys, metr,comparison_col = 'dataset'):
    """Generate the map of the best performance across each formulation

    note:: saves the plot inside the directory {ds}
    """
    path_best_map_plot = std_map_best_path(dir_out_viz_base,metr,subdir_anlys)
    states = gen_conus_basemap(dir_out_basemap = dir_out_viz_base)
    title = f"Top predicted value: {metr}"

    plot_best_perf = plot_best_perf_map(geo_df, states,title, comparison_col)
    plot_best_perf.savefig(path_best_map_plot, dpi=300, bbox_inches='tight')
    logging.info(f"Wrote top predicted value map to \n{path_best_map_plot}")
    plt.close(plot_best_perf)
    plt.close('all')
    gc.collect()

def std_donor_receiver_map_path(dir_out_viz_base: str | Path, ds: str, metr: str,
                                 algo_str: str, region_str: str) -> pathlib.PosixPath:
    """Generate a filepath for a donor-receiver pairing map

    :param dir_out_viz_base: The base directory for saving plots
    :type dir_out_viz_base: str | os.PathLike
    :param ds: The unique dataset name
    :type ds: str
    :param metr: The metric/response variable of interest
    :type metr: str
    :param algo_str: The type of algorithm used to create the donor-receiver pairing
    :type algo_str: str
    :param region_str: A short identifier for the mapped region (e.g. 'FL')
    :type region_str: str
    :return: Standardized filepath for the saved map
    :rtype: pathlib.PosixPath
    """
    path_map_plot = Path(f"{dir_out_viz_base}/{ds}/donor_receiver_map_{region_str}_{ds}_{metr}_{algo_str}.png")
    path_map_plot.parent.mkdir(parents=True, exist_ok=True)
    return path_map_plot

def plot_donor_receiver_map(gdf_receivers: gpd.GeoDataFrame, gdf_donors: gpd.GeoDataFrame,
                             gdf_lines: gpd.GeoDataFrame, states: gpd.GeoDataFrame, title: str,
                             colname_cluster: str = 'cluster_id',
                             gdf_no_pairing: gpd.GeoDataFrame = None,
                             colname_divide_proxy: str = 'is_divide_proxy') -> Figure:
    """Plot a donor-receiver pairing map for one region: receivers filled by their
    predicted cluster (categorical choropleth, same 'tab20' convention as
    :func:`plot_map_pred`), donor gage locations as solid points, receiver centroids as
    hollow squares, and a thin line from each receiver to its assigned donor.

    :param gdf_receivers: Receiver geometries with `colname_cluster`, already reprojected
     to the same CRS as `gdf_donors`/`gdf_lines`/`states`. May mix ordinary HUC12-shaped
     rows with divide-shaped proxy rows flagged via `colname_divide_proxy` -- both are
     colored by the same categorical call (so cluster colors stay consistent across the
     two), with proxy rows additionally outlined to flag the substitution to the viewer.
    :type gdf_receivers: gpd.GeoDataFrame
    :param gdf_donors: Donor gage point geometries (one row per donor)
    :type gdf_donors: gpd.GeoDataFrame
    :param gdf_lines: One LineString per receiver, connecting it to its assigned donor
    :type gdf_lines: gpd.GeoDataFrame
    :param states: State boundary geometries for map context
    :type states: gpd.GeoDataFrame
    :param title: The plot title
    :type title: str
    :param colname_cluster: Column in `gdf_receivers` holding the predicted cluster label, defaults to 'cluster_id'
    :type colname_cluster: str, optional
    :param gdf_no_pairing: Receiver geometries with no donor pairing, rendered as a distinct hatched layer, defaults to None
    :type gdf_no_pairing: gpd.GeoDataFrame, optional
    :param colname_divide_proxy: Boolean column in `gdf_receivers` marking rows whose geometry is
     a hydrofabric divide (not the original HUC12) standing in for a HUC12 too small to carry its
     own crosswalk entry, with a pairing borrowed from wherever the crosswalk actually assigned that
     divide -- see rafts_map_donor_receiver.py. Ignored if the column isn't present. Defaults to 'is_divide_proxy'
    :type colname_divide_proxy: str, optional
    :return: The rendered figure
    :rtype: Figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 16))

    # "No pairing" layer drawn first / lowest zorder, same convention as plot_map_pred's
    # "no crosswalk data" layer -- a real gap should never look identical to blank space.
    if gdf_no_pairing is not None and not gdf_no_pairing.empty:
        gdf_no_pairing.plot(ax=ax, facecolor='lightgray', hatch='///', edgecolor='dimgray',
                             linewidth=0.2, alpha=0.6, zorder=1, label='No donor pairing')

    gdf_receivers.plot(column=colname_cluster, ax=ax, categorical=True, cmap='tab20',
                        legend=True, zorder=2, edgecolor='white', linewidth=0.2)
    legend = ax.get_legend()
    if legend:
        legend.set_title("Predicted cluster", prop={'size': 15})
        for text in legend.get_texts():
            text.set_fontsize(12)
        # add_artist keeps this legend intact while the marker-shape legend below is
        # added as a second, independent legend -- see the analogous note in plot_map_pred.
        ax.add_artist(legend)

    # Divide-shaped proxy receivers: same fill color as any other receiver (assigned by
    # the categorical call above, off the same colname_cluster column, so a proxy's
    # borrowed cluster reads identically to a real receiver in that cluster) with an
    # added dashed outline so the viewer can tell "this shape is a divide standing in
    # for its HUC12, not the HUC12 itself" without it looking like a data error.
    has_proxy_col = colname_divide_proxy in gdf_receivers.columns
    if has_proxy_col and gdf_receivers[colname_divide_proxy].any():
        gdf_proxy = gdf_receivers[gdf_receivers[colname_divide_proxy]]
        gdf_proxy.boundary.plot(ax=ax, color='#7b2cbf', linewidth=1.6,
                                 linestyle=(0, (4, 2)), zorder=2.5)

    if not gdf_lines.empty:
        gdf_lines.plot(ax=ax, color='#0b0b0b', linewidth=0.3, alpha=0.15, zorder=3)

    # Receiver centroids: hollow squares, small and thin so the choropleth fill underneath
    # stays visible even at HUC12 density (a marker sized near the polygon spacing blots
    # out the fill entirely -- confirmed empirically during development).
    centroids = gdf_receivers.geometry.centroid
    ax.scatter(centroids.x, centroids.y, marker='s', s=5, facecolors='none',
               edgecolors='#0b0b0b', linewidths=0.45, zorder=4)

    # Donor gages: solid dots with a white halo for contrast against any fill color
    ax.scatter(gdf_donors.geometry.x, gdf_donors.geometry.y, marker='o', s=90,
               facecolors='#0b0b0b', edgecolors='#ffffff', linewidths=1.5, zorder=5)

    states.boundary.plot(ax=ax, color="#555555", linewidth=1, zorder=0, alpha=0.5)

    marker_handles = [
        Line2D([0], [0], marker='o', color='none', markerfacecolor='#0b0b0b', markeredgecolor='#ffffff',
               markeredgewidth=1.2, markersize=10, label='Donor gage'),
        Line2D([0], [0], marker='s', color='none', markerfacecolor='none', markeredgecolor='#0b0b0b',
               markeredgewidth=1.2, markersize=9, label='Receiver (centroid)'),
    ]
    if has_proxy_col and gdf_receivers[colname_divide_proxy].any():
        marker_handles.append(Line2D([0], [0], color='#7b2cbf', linewidth=1.6, linestyle=(0, (4, 2)),
                                      label='Divide-shaped proxy (large divide, borrowed pairing)'))
    if gdf_no_pairing is not None and not gdf_no_pairing.empty:
        marker_handles.append(mpatches.Patch(facecolor='lightgray', hatch='///', edgecolor='dimgray',
                                              label='No donor pairing'))
    ax.legend(handles=marker_handles, loc='lower left', fontsize=11, frameon=True)

    # Bounds are anchored to the ordinary (HUC12-shaped) receivers only. A divide-shaped
    # proxy's geometry can legitimately extend far outside the mapped region -- a divide
    # only needs a small corner inside the region to be selected as some HUC12's dominant
    # divide, but the divide's own footprint (drawn in full) can reach deep into a
    # neighboring region -- confirmed empirically: a single Great-Basin-scale proxy divide
    # stretched a WA/OR map's bounding box hundreds of miles south into California,
    # squeezing the actual region into a small corner of the frame. Excluding proxy rows
    # here doesn't hide any of their geometry (matplotlib still draws and simply clips
    # whatever falls outside the view), it just keeps the view itself anchored correctly.
    bounds_source = gdf_receivers
    if has_proxy_col and gdf_receivers[colname_divide_proxy].any():
        non_proxy = gdf_receivers[~gdf_receivers[colname_divide_proxy]]
        if not non_proxy.empty:
            bounds_source = non_proxy
    bounds = bounds_source.total_bounds
    if gdf_no_pairing is not None and not gdf_no_pairing.empty:
        nb = gdf_no_pairing.total_bounds
        bounds = np.array([min(bounds[0], nb[0]), min(bounds[1], nb[1]),
                            max(bounds[2], nb[2]), max(bounds[3], nb[3])])
    x_buffer = (bounds[2] - bounds[0]) * 0.08 or 1.0
    y_buffer = (bounds[3] - bounds[1]) * 0.08 or 1.0
    ax.set_xlim(bounds[0] - x_buffer, bounds[2] + x_buffer)
    ax.set_ylim(bounds[1] - y_buffer, bounds[3] + y_buffer)

    ax.set_title(title, fontsize=17)
    ax.set_axis_off()
    return plt.gcf()

def plot_donor_receiver_map_wrap(gdf_receivers: gpd.GeoDataFrame, gdf_donors: gpd.GeoDataFrame,
                                  dir_out_viz_base: str | Path, ds: str, metr: str, algo_str: str,
                                  region_str: str, colname_donor_id: str = 'donor_id',
                                  colname_cluster: str = 'cluster_id', epsg_reproj: int = 3857,
                                  gdf_no_pairing: gpd.GeoDataFrame = None,
                                  colname_divide_proxy: str = 'is_divide_proxy') -> Path:
    """Wrapper for building, saving, and closing a donor-receiver pairing map

    :param gdf_receivers: Receiver geometries with `colname_cluster` and `colname_donor_id` columns.
     May include divide-shaped proxy rows flagged via `colname_divide_proxy` -- see
     :func:`plot_donor_receiver_map` and rafts_map_donor_receiver.py.
    :type gdf_receivers: gpd.GeoDataFrame
    :param gdf_donors: Donor gage point geometries, with a 'gage_id' column matching `colname_donor_id` values
    :type gdf_donors: gpd.GeoDataFrame
    :param dir_out_viz_base: The base directory for storing visualization data
    :type dir_out_viz_base: str | os.PathLike
    :param ds: The dataset name
    :type ds: str
    :param metr: The response variable of interest
    :type metr: str
    :param algo_str: The algorithm shortstring
    :type algo_str: str
    :param region_str: A short identifier for the mapped region (e.g. 'FL')
    :type region_str: str
    :param colname_donor_id: Column in `gdf_receivers` naming each receiver's assigned donor, defaults to 'donor_id'
    :type colname_donor_id: str, optional
    :param colname_cluster: Column in `gdf_receivers` holding the predicted cluster label, defaults to 'cluster_id'
    :type colname_cluster: str, optional
    :param epsg_reproj: The EPSG code for reprojecting data for map display, defaults to 3857
    :type epsg_reproj: int, optional
    :param gdf_no_pairing: Receiver geometries with no donor pairing, defaults to None
    :type gdf_no_pairing: gpd.GeoDataFrame, optional
    :param colname_divide_proxy: Boolean column in `gdf_receivers` marking divide-shaped proxy rows,
     defaults to 'is_divide_proxy'. Ignored if the column isn't present.
    :type colname_divide_proxy: str, optional
    :return: The saved map's filepath
    :rtype: Path
    """
    path_map_plot = std_donor_receiver_map_path(dir_out_viz_base, ds, metr, algo_str, region_str)
    dir_out_basemap = path_map_plot.parent.parent
    states = gen_conus_basemap(dir_out_basemap=dir_out_basemap).to_crs(epsg=epsg_reproj)

    gdf_receivers = gdf_receivers.to_crs(4326).to_crs(epsg=epsg_reproj)
    gdf_donors = gdf_donors.to_crs(4326).to_crs(epsg=epsg_reproj)
    if gdf_no_pairing is not None and not gdf_no_pairing.empty:
        gdf_no_pairing = gdf_no_pairing.to_crs(4326).to_crs(epsg=epsg_reproj)

    # Build one donor->receiver line per receiver, in the same projected CRS as the
    # points being connected (a straight connector for visual reference only, not a
    # geodesic distance measurement, so a single consistent projected CRS is sufficient).
    # For a divide-shaped proxy row this naturally draws from the DIVIDE's own centroid
    # (gdf_receivers.geometry is already the divide, not the original HUC12, for those rows).
    donor_lookup = gdf_donors.set_index('gage_id')['geometry']
    centroids = gdf_receivers.geometry.centroid
    line_geoms = [LineString([c, donor_lookup[d]]) for c, d in zip(centroids, gdf_receivers[colname_donor_id])]
    gdf_lines = gpd.GeoDataFrame({colname_donor_id: gdf_receivers[colname_donor_id].values},
                                  geometry=line_geoms, crs=gdf_receivers.crs)

    n_donors = gdf_donors['gage_id'].nunique()
    n_proxy = int(gdf_receivers[colname_divide_proxy].sum()) if colname_divide_proxy in gdf_receivers.columns else 0
    proxy_note = f" ({n_proxy} via divide-shaped proxy)" if n_proxy else ""
    title = (f"Donor-Receiver Pairing -- {region_str}\n"
             f"{ds} | {algo_str} | {metr} | {len(gdf_receivers)} receivers{proxy_note} -> {n_donors} donors")

    fig = plot_donor_receiver_map(gdf_receivers, gdf_donors, gdf_lines, states, title,
                                   colname_cluster=colname_cluster, gdf_no_pairing=gdf_no_pairing,
                                   colname_divide_proxy=colname_divide_proxy)
    fig.savefig(path_map_plot, dpi=300, bbox_inches='tight')
    logging.info(f"Wrote donor-receiver map to \n{path_map_plot}")
    plt.close(fig)
    plt.close('all')
    gc.collect()
    return path_map_plot

def std_huc12_gap_classification_map_path(dir_out_qa: str | Path, region_str: str) -> pathlib.PosixPath:
    """Generate a filepath for a qa_check_missing_huc12_divide_existence.py classification map

    :param dir_out_qa: The QA output directory for this dataset (``ctx.dir_qa_out``)
    :type dir_out_qa: str | os.PathLike
    :param region_str: A short identifier for the mapped region (e.g. 'FL' or 'CONUS')
    :type region_str: str
    :return: Standardized filepath for the saved map
    :rtype: pathlib.PosixPath
    """
    path_map_plot = Path(f"{dir_out_qa}/qa_huc12_gap_classification_map_{region_str}.png")
    path_map_plot.parent.mkdir(parents=True, exist_ok=True)
    return path_map_plot

def plot_huc12_gap_classification_map(gdf_region: gpd.GeoDataFrame, gdf_classified: gpd.GeoDataFrame,
                                       states: gpd.GeoDataFrame, title: str,
                                       colname_class: str = 'classification') -> Figure:
    """Plot every HUC12 in scope with a crosswalk entry as light background context, and
    HUC12s missing from the crosswalk colored by their gap classification -- for quick
    visual triage of where the actionable gaps (a real divide exists but wasn't
    crosswalked) sit versus the likely-structural ones (no real divide there at all).

    :param gdf_region: Every HUC12 polygon in scope (both covered and missing)
    :type gdf_region: gpd.GeoDataFrame
    :param gdf_classified: The missing-HUC12 subset, with `colname_class` and geometry
    :type gdf_classified: gpd.GeoDataFrame
    :param states: State boundary geometries for map context
    :type states: gpd.GeoDataFrame
    :param title: The plot title
    :type title: str
    :param colname_class: Column in `gdf_classified` holding the classification label, defaults to 'classification'
    :type colname_class: str, optional
    :return: The rendered figure
    :rtype: Figure
    """
    # Size the figure to the data's actual aspect ratio, not a fixed portrait shape --
    # a hardcoded (14, 16) (right for a tall/narrow single state like Florida) squeezes
    # a landscape-shaped region like CONUS into a small corner of a mostly-blank canvas,
    # since geopandas locks the axis to equal aspect. Target a ~16in long edge.
    bounds = gdf_region.total_bounds
    data_w, data_h = bounds[2] - bounds[0], bounds[3] - bounds[1]
    aspect = (data_w / data_h) if data_h > 0 else 1.0
    if aspect >= 1:
        figsize = (16, max(6, 16 / aspect))
    else:
        figsize = (max(6, 16 * aspect), 16)
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Background: every HUC12 in scope (covered + missing), thin light fill for
    # geographic context -- the missing ones get painted over below.
    gdf_region.plot(ax=ax, facecolor='#e8e8e8', edgecolor='#bbbbbb', linewidth=0.15, zorder=1)

    # Fixed, non-cycled category colors -- only 2 categories here, and a warning red
    # for the actionable class reads more usefully than an arbitrary categorical hue.
    class_style = {
        'divides_exist_not_crosswalked': dict(facecolor='#d62728', edgecolor='#5c0000', hatch=None,
                                               label='divides_exist_not_crosswalked (ACTIONABLE)'),
        'no_divide_overlap': dict(facecolor='#6699cc', edgecolor='#1b3a5c', hatch='///',
                                   label='no_divide_overlap (likely structural)'),
    }
    for cls, style in class_style.items():
        subset = gdf_classified[gdf_classified[colname_class] == cls]
        if subset.empty:
            continue
        subset.plot(ax=ax, facecolor=style['facecolor'], edgecolor=style['edgecolor'],
                    linewidth=0.3, hatch=style['hatch'], alpha=0.9, zorder=3, label=style['label'])

    states.boundary.plot(ax=ax, color="#555555", linewidth=1, zorder=0, alpha=0.5)

    handles = [mpatches.Patch(facecolor='#e8e8e8', edgecolor='#bbbbbb', label='has crosswalk entry')]
    for cls in ('divides_exist_not_crosswalked', 'no_divide_overlap'):
        if (gdf_classified[colname_class] == cls).any():
            style = class_style[cls]
            handles.append(mpatches.Patch(facecolor=style['facecolor'], edgecolor=style['edgecolor'],
                                           hatch=style['hatch'], label=style['label']))
    ax.legend(handles=handles, loc='lower left', fontsize=11, frameon=True)

    x_buffer = data_w * 0.03 or 1.0
    y_buffer = data_h * 0.03 or 1.0
    ax.set_xlim(bounds[0] - x_buffer, bounds[2] + x_buffer)
    ax.set_ylim(bounds[1] - y_buffer, bounds[3] + y_buffer)

    ax.set_title(title, fontsize=17)
    ax.set_axis_off()
    return plt.gcf()

def plot_huc12_gap_classification_map_wrap(gdf_region: gpd.GeoDataFrame, gdf_classified: gpd.GeoDataFrame,
                                            dir_out_qa: str | Path, dir_out_basemap: str | Path,
                                            region_str: str, colname_class: str = 'classification',
                                            epsg_reproj: int = 3857) -> Path:
    """Wrapper for building, saving, and closing a HUC12 gap classification map

    :param gdf_region: Every HUC12 polygon in scope (both covered and missing)
    :type gdf_region: gpd.GeoDataFrame
    :param gdf_classified: The missing-HUC12 subset, with `colname_class` and geometry
    :type gdf_classified: gpd.GeoDataFrame
    :param dir_out_qa: The QA output directory for this dataset (``ctx.dir_qa_out``)
    :type dir_out_qa: str | os.PathLike
    :param dir_out_basemap: Directory to cache/read the CONUS state-boundary shapefile from
        -- pass ``ctx.dir_out_viz_base`` to reuse the same cache other maps already use.
    :type dir_out_basemap: str | os.PathLike
    :param region_str: A short identifier for the mapped region (e.g. 'FL' or 'CONUS')
    :type region_str: str
    :param colname_class: Column in `gdf_classified` holding the classification label, defaults to 'classification'
    :type colname_class: str, optional
    :param epsg_reproj: The EPSG code for reprojecting data for map display, defaults to 3857
    :type epsg_reproj: int, optional
    :return: The saved map's filepath
    :rtype: Path
    """
    path_map_plot = std_huc12_gap_classification_map_path(dir_out_qa, region_str)
    states = gen_conus_basemap(dir_out_basemap=dir_out_basemap).to_crs(epsg=epsg_reproj)

    gdf_region_proj = gdf_region.to_crs(4326).to_crs(epsg=epsg_reproj)
    gdf_classified_proj = gdf_classified.to_crs(4326).to_crs(epsg=epsg_reproj)

    n_actionable = int((gdf_classified[colname_class] == 'divides_exist_not_crosswalked').sum())
    n_structural = int((gdf_classified[colname_class] == 'no_divide_overlap').sum())
    title = (f"Crosswalk Gap Classification -- {region_str}\n"
             f"{n_actionable:,} actionable, {n_structural:,} likely structural, of {len(gdf_region):,} HUC12s in scope")

    fig = plot_huc12_gap_classification_map(gdf_region_proj, gdf_classified_proj, states, title,
                                             colname_class=colname_class)
    fig.savefig(path_map_plot, dpi=250, bbox_inches='tight')
    logging.info(f"Wrote HUC12 gap classification map to \n{path_map_plot}")
    plt.close(fig)
    plt.close('all')
    gc.collect()
    return path_map_plot

def plot_divide_reassignment_defects(df_defects: pd.DataFrame, gdf_missing_huc: gpd.GeoDataFrame,
                                      gdf_assigned_huc: gpd.GeoDataFrame, gdf_divides: gpd.GeoDataFrame,
                                      divide_id_col: str, title: str, huc12_col: str = 'huc12') -> Figure:
    """Small-multiples grid: one zoomed-in panel per reassignment "defect" row, showing the
    overlapping divide's footprint against both the "missing" HUC12 and whatever HUC12 the
    crosswalk actually assigned that divide to -- so a human can visually confirm each
    qa_check_huc12_divide_reassignment.py finding against the real geometry rather than
    trusting the area-fraction numbers alone.

    :param df_defects: Rows to plot (already the desired subset/ordering, e.g. the
        top-N by margin), with `huc12_col`, `divide_id_col`, 'assigned_huc12', 'margin_frac_divide'
    :type df_defects: pd.DataFrame
    :param gdf_missing_huc: Geometries for the "missing" HUC12s referenced in `df_defects`, with a `huc12_col` column
    :type gdf_missing_huc: gpd.GeoDataFrame
    :param gdf_assigned_huc: Geometries for the assigned HUC12s referenced in `df_defects`, with a `huc12_col` column
    :type gdf_assigned_huc: gpd.GeoDataFrame
    :param gdf_divides: Geometries for the divides referenced in `df_defects`, with `divide_id_col`
    :type gdf_divides: gpd.GeoDataFrame
    :param divide_id_col: Column name holding the divide identifier
    :type divide_id_col: str
    :param title: The figure's overall title
    :type title: str
    :param huc12_col: Column name holding the aggregation-unit (e.g. HUC12) identifier in
     `gdf_missing_huc`/`gdf_assigned_huc`/`df_defects` -- not assumed to literally be 'huc12',
     defaults to 'huc12'
    :type huc12_col: str, optional
    :return: The rendered figure
    :rtype: Figure
    """
    n = len(df_defects)
    ncols = min(5, max(n, 1))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.8 * nrows), squeeze=False)
    axes = axes.flatten()

    huc_missing_by_id = gdf_missing_huc.set_index(huc12_col).geometry
    huc_assigned_by_id = gdf_assigned_huc.set_index(huc12_col).geometry
    divide_by_id = gdf_divides.set_index(divide_id_col).geometry

    for i, row in enumerate(df_defects.itertuples(index=False)):
        ax = axes[i]
        divide_id = getattr(row, divide_id_col)
        missing_huc = getattr(row, huc12_col)
        assigned_huc = row.assigned_huc12

        div_geom = divide_by_id.get(divide_id)
        miss_geom = huc_missing_by_id.get(missing_huc)
        assn_geom = huc_assigned_by_id.get(assigned_huc) if assigned_huc else None

        if div_geom is not None:
            gpd.GeoSeries([div_geom], crs=gdf_divides.crs).plot(
                ax=ax, facecolor='#aec7e8', edgecolor='#1f77b4', linewidth=1, alpha=0.6, zorder=1)
        if miss_geom is not None:
            gpd.GeoSeries([miss_geom], crs=gdf_missing_huc.crs).boundary.plot(
                ax=ax, color='#d62728', linewidth=2.3, linestyle='--', zorder=3)
        if assn_geom is not None:
            gpd.GeoSeries([assn_geom], crs=gdf_assigned_huc.crs).boundary.plot(
                ax=ax, color='#2ca02c', linewidth=2.3, zorder=3)

        geoms = [g for g in (div_geom, miss_geom, assn_geom) if g is not None]
        if geoms:
            u = gpd.GeoSeries(geoms).union_all()
            minx, miny, maxx, maxy = u.bounds
            pad_x = (maxx - minx) * 0.12 or 1.0
            pad_y = (maxy - miny) * 0.12 or 1.0
            ax.set_xlim(minx - pad_x, maxx + pad_x)
            ax.set_ylim(miny - pad_y, maxy + pad_y)

        margin_str = f"{row.margin_frac_divide:.1%}" if pd.notna(row.margin_frac_divide) else "n/a"
        ax.set_title(f"{missing_huc}\n{divide_id}  |  margin={margin_str}", fontsize=9.5)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('#999999')

    for j in range(n, len(axes)):
        axes[j].axis('off')

    legend_handles = [
        mpatches.Patch(facecolor='#aec7e8', edgecolor='#1f77b4', label='divide footprint'),
        Line2D([0], [0], color='#d62728', linestyle='--', linewidth=2.3, label='"missing" HUC12 (no crosswalk entry)'),
        Line2D([0], [0], color='#2ca02c', linewidth=2.3, label='HUC12 the crosswalk actually assigned'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=3, fontsize=11, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(title, fontsize=16, y=1.02)
    fig.tight_layout()
    return fig

def plot_divide_reassignment_defects_wrap(df_defects: pd.DataFrame, gdf_missing_huc: gpd.GeoDataFrame,
                                           gdf_assigned_huc: gpd.GeoDataFrame, gdf_divides: gpd.GeoDataFrame,
                                           divide_id_col: str, dir_out_qa: str | Path, region_str: str,
                                           top_n: int = 10, epsg_reproj: int = 3857,
                                           huc12_col: str = 'huc12') -> Path | None:
    """Wrapper for building, saving, and closing the top-N divide-reassignment defect panels

    :param df_defects: The full reassignment result table, pre-sorted so the most
        actionable rows (e.g. by 'margin_frac_divide' descending) come first
    :type df_defects: pd.DataFrame
    :param gdf_missing_huc: Geometries for the "missing" HUC12s referenced in `df_defects`, with a `huc12_col` column
    :type gdf_missing_huc: gpd.GeoDataFrame
    :param gdf_assigned_huc: Geometries for the assigned HUC12s referenced in `df_defects`, with a `huc12_col` column
    :type gdf_assigned_huc: gpd.GeoDataFrame
    :param gdf_divides: Geometries for the divides referenced in `df_defects`, with `divide_id_col`
    :type gdf_divides: gpd.GeoDataFrame
    :param divide_id_col: Column name holding the divide identifier
    :type divide_id_col: str
    :param dir_out_qa: The QA output directory for this dataset (``ctx.dir_qa_out``)
    :type dir_out_qa: str | os.PathLike
    :param region_str: A short identifier for the mapped region (e.g. 'FL')
    :type region_str: str
    :param top_n: How many rows (panels) to plot, defaults to 10
    :type top_n: int, optional
    :param epsg_reproj: The EPSG code for reprojecting data for map display, defaults to 3857
    :type epsg_reproj: int, optional
    :param huc12_col: Column name holding the aggregation-unit (e.g. HUC12) identifier -- not
     assumed to literally be 'huc12', defaults to 'huc12'
    :type huc12_col: str, optional
    :return: The saved plot's filepath, or None if `df_defects` was empty
    :rtype: Path | None
    """
    if df_defects.empty:
        return None
    df_top = df_defects.head(top_n).reset_index(drop=True)

    path_plot = Path(dir_out_qa) / f"qa_huc12_divide_reassignment_defects_{region_str}.png"
    path_plot.parent.mkdir(parents=True, exist_ok=True)

    gdf_missing_proj = gdf_missing_huc.to_crs(4326).to_crs(epsg=epsg_reproj)
    gdf_assigned_proj = gdf_assigned_huc.to_crs(4326).to_crs(epsg=epsg_reproj)
    gdf_divides_proj = gdf_divides.to_crs(4326).to_crs(epsg=epsg_reproj)

    title = f"Top {len(df_top)} Crosswalk Reassignment Defects -- {region_str}"
    fig = plot_divide_reassignment_defects(df_top, gdf_missing_proj, gdf_assigned_proj, gdf_divides_proj,
                                            divide_id_col, title, huc12_col=huc12_col)
    fig.savefig(path_plot, dpi=200, bbox_inches='tight')
    logging.info(f"Wrote divide reassignment defect panels to \n{path_plot}")
    plt.close(fig)
    plt.close('all')
    gc.collect()
    return path_plot