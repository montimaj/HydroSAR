# Author: Sayantan Majumdar
# Email: smxnv@mst.edu

import numpy as np
import pandas as pd
import os
import seaborn
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sklearn.metrics as metrics
from glob import glob
from Python_Files.hydrolibs import rasterops as rops
from Python_Files.hydrolibs import vectorops as vops
from Python_Files.hydrolibs.sysops import makedirs, make_proper_dir_name
from datetime import datetime
from sklearn.linear_model import LinearRegression


def create_gw_time_series(actual_gw_file_dir, pred_gw_file_dir, grace_dir, actual_gw_pattern='GW*.tif',
                          pred_gw_pattern='pred*.tif', grace_pattern='GRACE*.tif', make_trend=False,
                          out_dir='../Outputs/'):
    """
    Create time series data for actual and predicted GW values (annual mean) along with GRACE
    :param actual_gw_file_dir: Actual GW pumping raster directory
    :param pred_gw_file_dir: Predicted GW pumping raster directory
    :param grace_dir: GRACE raster directory
    :param actual_gw_pattern: Actual GW pumping raster file pattern
    :param pred_gw_pattern: Predicted GW pumping raster file pattern
    :param grace_pattern: GRACE raster file pattern
    :param make_trend: Make trend data for the monthly grace values
    :param out_dir: Output directory for storing the CSV files
    :return: Dataframe containing year and corresponding mean values
    """

    actual_gw_raster_dict = rops.create_raster_dict(actual_gw_file_dir, pattern=actual_gw_pattern)
    pred_gw_raster_dict = rops.create_raster_dict(pred_gw_file_dir, pattern=pred_gw_pattern)
    grace_yearly_raster_dict = rops.create_yearly_avg_raster_dict(grace_dir, pattern=grace_pattern)
    grace_monthly_raster_dict = rops.create_monthly_avg_raster_dict(grace_dir, pattern=grace_pattern)
    years = sorted(list(pred_gw_raster_dict.keys()))
    dt_list = sorted(list(grace_monthly_raster_dict.keys()))
    mean_actual_gw = {}
    mean_pred_gw = {}
    mean_grace = {}
    mean_monthly_grace = {}
    for year in years:
        mean_actual_gw[year] = np.nanmean(actual_gw_raster_dict[year])
        mean_pred_gw[year] = np.nanmean(pred_gw_raster_dict[year])
        mean_grace[year] = np.nanmean(grace_yearly_raster_dict[year])
    for dt in dt_list:
        mean_monthly_grace[dt] = grace_monthly_raster_dict[dt]

    df1 = {'YEAR': years, 'Actual_GW': list(mean_actual_gw.values()), 'Pred_GW': list(mean_pred_gw.values()),
           'GRACE': list(mean_grace.values())}
    df1 = pd.DataFrame(data=df1)
    df1.to_csv(out_dir + 'ts_yearly.csv', index=False)
    time = [datetime.strftime(t, '%b %Y') for t in dt_list]
    grace_monthly_list = list(mean_monthly_grace.values())
    df2 = {'Time': time, 'GRACE': grace_monthly_list}
    if make_trend:
        grace_fit = get_trend(dt_list, grace_monthly_list)
        df2['Trend'] = grace_fit
    df2 = pd.DataFrame(data=df2)
    df2.to_csv(out_dir + 'grace_monthly.csv', index=False)
    return df1, df2


def create_gw_forecast_time_series(actual_gw_file_dir_list, pred_gw_file_dir_list, grace_csv, gw_name_list=None,
                                   use_gws=True, actual_gw_pattern='GW*.tif', pred_gw_pattern='pred*.tif',
                                   out_dir='../Outputs/', exclude_years=(), forecast_years=()):
    """
    Create GW and GRACE dataframes
    :param actual_gw_file_dir_list: Actual GW pumping raster directory list
    :param pred_gw_file_dir_list: Predicted GW pumping raster directory list
    :param grace_csv: GRACE TWS CSV file
    :param gw_name_list: List of GMD names
    :param use_gws: Set False to use entire GW raster for analysis
    :param actual_gw_pattern: Actual GW pumping raster file pattern
    :param pred_gw_pattern: Predicted GW pumping raster file pattern
    :param out_dir: Output directory for storing the CSV files
    :param exclude_years: Exclude these years from analysis
    :param forecast_years: Set these years as forecast years
    :return: Two dataframes, one with the GW pumping values and the other containing the monthly GRACE values
    """

    grace_df = pd.read_csv(grace_csv)
    grace_df = grace_df.dropna(axis=0)
    grace_df['GRACE'] = grace_df['GRACE'] * 10
    grace_df['DT'] = pd.to_datetime(grace_df['DT']).dt.date
    gw_df = pd.DataFrame()
    gw_raster_df = pd.DataFrame()
    for index, (actual_gw_file_dir, pred_gw_file_dir) in enumerate(zip(actual_gw_file_dir_list, pred_gw_file_dir_list)):
        actual_gw_raster_dict = rops.create_raster_dict(actual_gw_file_dir, pattern=actual_gw_pattern)
        pred_gw_raster_dict = rops.create_raster_dict(pred_gw_file_dir, pattern=pred_gw_pattern)
        years = sorted(list(pred_gw_raster_dict.keys()))
        for ey in exclude_years:
            years.remove(ey)
        mean_actual_gw = {}
        mean_pred_gw = {}
        for year in years:
            pred_raster = pred_gw_raster_dict[year]
            pred_raster = pred_raster.reshape(pred_raster.shape[0] * pred_raster.shape[1])
            mean_pred_gw[year] = np.nanmean(pred_raster)
            if year in actual_gw_raster_dict.keys() and year not in forecast_years:
                actual_raster = actual_gw_raster_dict[year]
                actual_raster = actual_raster.reshape(actual_raster.shape[0] * actual_raster.shape[1])
                mean_actual_gw[year] = np.nanmean(actual_raster)
                raster_dict = {'YEAR': [year] * actual_raster.shape[0], 'Actual_GW': actual_raster,
                               'Pred_GW': pred_raster}
                if use_gws:
                    raster_dict['GW_NAME'] = [gw_name_list[index]] * actual_raster.shape[0]
                gw_raster_df = gw_raster_df.append(pd.DataFrame(data=raster_dict))
            else:
                mean_actual_gw[year] = np.nan
        gw_dict = {'YEAR': years, 'Actual_GW': list(mean_actual_gw.values()), 'Pred_GW': list(mean_pred_gw.values())}
        if use_gws:
            gw_dict['GW_NAME'] = [gw_name_list[index]] * len(years)
        gw_df = gw_df.append(pd.DataFrame(data=gw_dict))
    gw_df.to_csv(out_dir + 'gw_yearly_new.csv', index=False)
    gw_raster_df = gw_raster_df.dropna(axis=0)
    gw_raster_df.to_csv(out_dir + 'GW_Raster.csv', index=False)
    return gw_df, gw_raster_df, grace_df


def get_trend(dt_list, value_list):
    """
    Obtain trend data
    :param dt_list:  List of time values as DateTime object
    :param value_list: List of values
    :return: Fitted values
    """

    dt_ordinal = [dt.toordinal() for dt in dt_list]
    trend_model = LinearRegression()
    dt_ordinal = np.array(dt_ordinal).reshape(-1, 1)
    values = np.array(value_list).reshape(-1, 1)
    trend_model.fit(X=dt_ordinal, y=values)
    fit = np.poly1d([trend_model.coef_[0][0]])(value_list)
    return fit


def create_time_series_plot(input_df_list):
    """
    Create time series plot
    :param input_df_list: Input data frames as constructed from #create_gw_time_series
    :return: None
    """

    df1, df2 = input_df_list
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    year = df1['YEAR']
    grace_df = df1[['YEAR', 'GRACE']]
    df1 = df1.drop('GRACE', axis=1)
    df1.set_index('YEAR').plot(ax=ax1)
    df2.set_index('Time').plot(ax=ax3)

    ax1.axvline(x=2011, color='k')
    ax1.set_ylabel('Mean GW Pumping (mm)')
    ax1.set_xticks(year)
    ax1.set_xticklabels(year)
    grace_df.set_index('YEAR').plot(ax=ax2)
    ax2.set_xticks(year)
    ax2.set_xticklabels(year)
    ax2.invert_yaxis()
    ax2.set_ylabel('Mean TWS (mm)')
    ax3.invert_yaxis()
    ax3.set_ylabel('Monthly TWS (mm)')
    plt.show()


def create_time_series_forecast_plot(input_df_list, forecast_years=(2019, ), plot_title=''):
    """
    Create time series plot (Note: the visualization is hardcoded, needs to be changed in future versions)
    :param input_df_list: Input data frames as constructed from #create_gw_time_series
    :param forecast_years: The line color changes for these years
    :param plot_title: Plot title
    :return: None
    """

    df1, df2 = input_df_list
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.suptitle(plot_title)
    df1.set_index('YEAR').plot(ax=ax1)
    df2.set_index('DT').plot(ax=ax2)
    df2_years = list(df2.DT)
    ax1.axvspan(2009.8, 2019.2, color='#a6bddb', alpha=0.6)
    min_forecast_yr = min(forecast_years)
    ax1.set_xlim(left=np.min(df1.YEAR) - 0.1, right=np.max(df1.YEAR) + 0.1)
    ax1.axvspan(2019.21, 2020.1, color='#fee8c8', alpha=1)
    ax1.legend(loc=2, ncol=2, frameon=False, fancybox=False, bbox_to_anchor=(0.12, 1),
               labels=['Actual GW', 'Predicted GW', 'Test Years', 'Forecast'])
    ax1.set_ylabel('Mean GW Pumping (mm)')
    ax1.set_xticks(df1.YEAR)
    ax1.set_xticklabels(df1.YEAR)
    ax1.set_xlabel('Year')
    ax2.set_ylim(bottom=-150, top=150)
    ax2.invert_yaxis()
    ax2.set_ylabel('Monthly TWS (mm)')
    ax2.set_xlabel('Year')
    ax2.legend(loc=2, bbox_to_anchor=(0.1, 1), frameon=False, fancybox=False, labels=['GRACE TWS'])
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    datemin = np.datetime64(df2_years[0], 'Y')
    datemax = np.datetime64(df2_years[-2], 'Y') + np.timedelta64(1, 'Y')
    ax2.set_xlim(datemin, datemax)
    ax2.format_xdata = mdates.DateFormatter('%Y')
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.show()


def create_gw_time_series_forecast_plot(input_df_list, gw_name_list, forecast_years=(2019, ), plot_title=''):
    """
    Create time series plot considering all GMDs (Note: the visualization is hardcoded, needs to be changed in
    future versions)
    :param input_df_list: Input data frames as constructed from #create_gw_time_series
    :param gw_name_list: GMD or AMA/INA labels
    :param forecast_years: The line color changes for these years
    :param plot_title: Plot title
    :return: None
    """

    gw_df, grace_df = input_df_list
    for gw in gw_name_list:
        fig, (ax1, ax2) = plt.subplots(2, 1)
        fig.suptitle(plot_title)
        df = gw_df[gw_df.GW_NAME == gw]
        df.set_index('YEAR').plot(ax=ax1)
        ax1.axvspan(2001.8, 2019.2, color='#a6bddb', alpha=0.6)
        min_forecast_yr = min(forecast_years)
        ax1.set_xlim(left=np.min(df.YEAR) - 0.1, right=np.max(df.YEAR) + 0.1)
        ax1.axvspan(2019.21, 2020.1, color='#fee8c8', alpha=1)
        ax1.legend(loc=2, ncol=2, frameon=False, fancybox=False, bbox_to_anchor=(0.1, 1),
                   labels=['Actual GW: ' + gw, 'Predicted GW: ' + gw, 'Test Years', 'Forecast'])
        ax1.set_ylabel('Mean GW Pumping (mm)')
        ax1.set_xticks(df.YEAR)
        ax1.set_xticklabels(df.YEAR)
        ax1.set_xlabel('Year')
        ax1.set_ylim(bottom=0, top=100)
        grace_df.set_index('DT').plot(ax=ax2)
        ax2.set_ylim(bottom=-150, top=150)
        ax2.invert_yaxis()
        ax2.set_ylabel('Monthly TWS (mm)')
        ax2.set_xlabel('Year')
        ax2.legend(loc=2, bbox_to_anchor=(0.1, 1), frameon=False, fancybox=False, labels=['GRACE TWS'])
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.show()


def preprocess_gws(actual_gw_dir, pred_gw_dir, input_gw_file, out_dir, actual_gw_pattern='GW*.tif',
                    pred_gw_pattern='pred*.tif'):
    """
    Preprocess GMD (Kansas) or AMA/INA (Arizona) shapefiles and rasters
    :param actual_gw_dir: Directory containing the actual data
    :param pred_gw_dir: Directory containing the predicted data
    :param input_gw_file: Input GMD (Kansas) or AMA/INA (Arizona) shapefile for creating GMD or AMA/INA specific plots
    :param out_dir: Output directory for storing intermediate files
    :param actual_gw_pattern: Actual GW pumping raster file pattern
    :param pred_gw_pattern: Predicted GW pumping raster file pattern
    :return: Directory paths (as tuple) of the actual and predicted GMD specific GW rasters along with the GMD names
    """

    out_shp_dir = make_proper_dir_name(out_dir + 'GW_SHP')
    out_actual_gw_dir = make_proper_dir_name(out_dir + 'Actual_GW_Rasters')
    out_pred_gw_dir = make_proper_dir_name(out_dir + 'Pred_GW_Rasters')
    makedirs([out_shp_dir, out_actual_gw_dir, out_pred_gw_dir])
    label_attr = 'NAME_ABBR'
    if 'KS' in out_dir:
        label_attr = 'GMD_label'
    vops.extract_polygons(input_gw_file, out_shp_dir, label_attr=label_attr)
    actual_gw_dir_list, pred_gw_dir_list, gw_name_list = [], [], []
    print('Preprocessing started...')
    for gw_shp in glob(out_shp_dir + '*.shp'):
        gw_name = gw_shp[gw_shp.rfind(os.sep) + 1: gw_shp.rfind('.')]
        gw_name_list.append(gw_name)
        actual_outdir = make_proper_dir_name(out_actual_gw_dir + gw_name)
        pred_outdir = make_proper_dir_name(out_pred_gw_dir + gw_name)
        makedirs([actual_outdir, pred_outdir])
        actual_gw_dir_list.append(actual_outdir)
        pred_gw_dir_list.append(pred_outdir)
        rops.crop_rasters(input_raster_dir=actual_gw_dir, input_mask_file=gw_shp, pattern=actual_gw_pattern,
                          outdir=actual_outdir, ext_mask=False)
        rops.crop_rasters(input_raster_dir=pred_gw_dir, input_mask_file=gw_shp, pattern=pred_gw_pattern,
                          outdir=pred_outdir, ext_mask=False)
    return actual_gw_dir_list, pred_gw_dir_list, gw_name_list


def calculate_gw_stats(gw_df, gw_name_list, out_dir, train_end=2010, test_start=2011):
    """
    Calculate error metrics for each GMD
    :param gw_df: Input GW Dataframe containing actual and predicted GW
    :param gw_name_list: GMD labels
    :param out_dir: Output directory to write results to
    :param train_end: Training year end
    :param test_start: Test year start
    :return: GMD metrics dataframe
    """

    gw_metrics_df = pd.DataFrame()
    gw_df = gw_df.dropna(axis=0)
    gw_df_list = gw_df[gw_df.YEAR <= train_end], gw_df[gw_df.YEAR >= test_start], gw_df
    gw_df_labels = ['TRAIN', 'TEST', 'ALL']
    for gw_df, gw_df_label in zip(gw_df_list, gw_df_labels):
        for gw in gw_name_list:
            actual_values = gw_df[gw_df.GW_NAME == gw].Actual_GW
            pred_values = gw_df[gw_df.GW_NAME == gw].Pred_GW
            r2, mae, rmse, nmae, nrmse = get_error_stats(actual_values, pred_values)
            gw_metrics_dict = {'GW_TYPE': [gw_df_label], 'GW_NAME': [gw], 'R2': [r2], 'RMSE': [rmse], 'MAE': [mae],
                               'NRMSE': [nrmse], 'NMAE': [nmae]}
            gw_metrics_df = gw_metrics_df.append(pd.DataFrame(data=gw_metrics_dict))
    out_csv = out_dir + 'GW_Metrics.csv'
    gw_metrics_df.to_csv(out_csv, index=False)
    return gw_metrics_df


def run_analysis(actual_gw_dir, pred_gw_dir, grace_csv, out_dir, input_gw_file=None, use_gws=True,
                 actual_gw_pattern='GW*.tif', pred_gw_pattern='pred*.tif', exclude_years=(), forecast_years=(),
                 show_plots=True):
    """
    Run model analysis to get actual vs predicted graph along with GRACE TWSA variations
    :param actual_gw_dir: Directory containing the actual data
    :param pred_gw_dir: Directory containing the predicted data
    :param grace_csv: GRACE TWSA CSV file
    :param out_dir: Output directory for storing intermediate files
    :param input_gw_file: Input GMD (Kansas) or AMA/INA (Arizona) shapefile for creating GMD or AMA/INA specific plots,
    required only if use_gws=True
    :param use_gws: Set False to use entire GW raster for analysis
    :param actual_gw_pattern: Actual GW pumping raster file pattern
    :param pred_gw_pattern: Predicted GW pumping raster file pattern
    :param exclude_years: Exclude these years from analysis
    :param forecast_years: Set these years as forecast years
    :param show_plots: Set True to show plots
    :return: None
    """

    out_dir = make_proper_dir_name(out_dir)
    makedirs([out_dir])
    if not use_gws:
        ts_df = create_gw_forecast_time_series([actual_gw_dir], [pred_gw_dir], grace_csv=grace_csv, out_dir=out_dir,
                                               actual_gw_pattern=actual_gw_pattern, pred_gw_pattern=pred_gw_pattern,
                                               use_gws=use_gws, exclude_years=exclude_years,
                                               forecast_years=forecast_years)
        ts_df = ts_df[0], ts_df[2]
        if show_plots:
            create_time_series_forecast_plot(ts_df)
    else:
        actual_gw_dir_list, pred_gw_dir_list, gw_name_list = preprocess_gws(actual_gw_dir, pred_gw_dir,
                                                                              input_gw_file, out_dir,
                                                                              actual_gw_pattern, pred_gw_pattern)

        ts_df = create_gw_forecast_time_series(actual_gw_dir_list, pred_gw_dir_list, gw_name_list=gw_name_list,
                                               grace_csv=grace_csv, use_gws=use_gws, out_dir=out_dir,
                                               actual_gw_pattern=actual_gw_pattern, pred_gw_pattern=pred_gw_pattern,
                                               exclude_years=exclude_years, forecast_years=forecast_years)

        print(calculate_gw_stats(ts_df[1], gw_name_list, out_dir))
        ts_df = ts_df[0], ts_df[2]
        if show_plots:
            create_gw_time_series_forecast_plot(ts_df, gw_name_list=gw_name_list)


def generate_feature_box_plots(input_csv_file, year_col='YEAR', temporal_features=('ET', 'P'), pred_attr='GW',
                               drop_attr=('GMD',)):
    """
    Generate box plots for all features
    :param input_csv_file: Input CSV file path
    :param year_col: Name of Year column
    :param temporal_features: Temporal feature names
    :param pred_attr: Prediction attribute name to be dropped from boxplot
    :param drop_attr: Drop these attributes from the plots
    :return: None
    """

    input_df = pd.read_csv(input_csv_file)
    input_df = input_df.drop(columns=list(drop_attr))
    feature_names = input_df.columns.values.tolist()
    feature_names.remove(pred_attr)
    for tf in temporal_features:
        sub_df = input_df[[year_col, tf]]
        fig, ax = plt.subplots(figsize=(12, 5))
        seaborn.boxplot(x='YEAR', y=tf, data=sub_df, ax=ax)
        plt.show()
        feature_names.remove(tf)
    feature_names.remove(year_col)
    feature_names.remove('Crop')
    sub_df = pd.melt(input_df.loc[input_df[year_col] == 2015][feature_names])
    sub_df = sub_df.rename(columns={'variable': 'Land-Use Features', 'value': 'Land-Use Density'})
    seaborn.boxplot(x='Land-Use Features', y='Land-Use Density', data=sub_df)
    plt.show()
    sub_df = pd.melt(input_df.loc[input_df[year_col] == 2015][['Crop']])
    sub_df['variable'] = ''
    sub_df = sub_df.rename(columns={'variable': 'Crop Coefficient', 'value': 'Value'})
    seaborn.boxplot(x='Crop Coefficient', y='Value', data=sub_df)
    plt.show()


def get_error_stats(actual_values, pred_values, round_places=2, normalize_metric='mean'):
    """
    Get R2, MAE, RMSE, NMAE (normalized MAE), and NRMSE (normalized RMSE)
    :param actual_values: List of actual values
    :param pred_values: List of predicted values
    :param round_places: Number of decimal places to round at, default 2.
    :param normalize_metric: Change to 'sd' to normalize MAE and RMSE by standard deviation
    :return: Tuple containing R2, MAE, RMSE, NMAE, and NRMSE (rounded to 2 decimal places by default)
    """

    if isinstance(actual_values, pd.DataFrame):
        actual_values = actual_values.iloc[:, 0].tolist()
    if isinstance(pred_values, pd.DataFrame):
        pred_values = pred_values.iloc[:, 0].tolist()
    mae = metrics.mean_absolute_error(actual_values, pred_values)
    r2_score = np.round(metrics.r2_score(actual_values, pred_values), round_places)
    rmse = metrics.mean_squared_error(actual_values, pred_values, squared=False)
    normalization_var = np.var(actual_values) ** 0.5
    if normalize_metric == 'mean':
        normalization_var = np.mean(actual_values)
    nrmse = np.round(rmse / normalization_var, round_places)
    nmae = np.round(mae / normalization_var, round_places)
    rmse = np.round(rmse, round_places)
    mae = np.round(mae, round_places)
    return r2_score, mae, rmse, nmae, nrmse


def subsidence_analysis(input_dir):
    """
    Analyze subsidence and predicted GW rasters
    :param input_dir: Input data directory
    :return: None
    """

    subsidence_gw_dir = input_dir + 'Subsidence_GW/'
    subsidence_dirs = glob(subsidence_gw_dir + '*')
    print(subsidence_dirs)
    # TODO: Create CSV files for each year
