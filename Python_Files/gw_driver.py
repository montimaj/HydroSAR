# Author: Sayantan Majumdar
# Email: smxnv@mst.edu

import pandas as pd
from Python_Files.hydrolibs import rasterops as rops
from Python_Files.hydrolibs import vectorops as vops
from Python_Files.hydrolibs import process_ssebop as pssebop
from Python_Files.hydrolibs.sysops import makedirs, make_proper_dir_name, copy_files
from Python_Files.hydrolibs import random_forest_regressor as rfr
from Python_Files.hydrolibs import model_analysis as ma
from glob import glob


class HydroML:

    def __init__(self, input_dir, file_dir, output_dir, input_ts_dir, output_shp_dir, output_gw_raster_dir,
                 input_state_file, input_cdl_file, gdal_path, input_gw_boundary_file=None, input_ama_ina_file=None,
                 input_watershed_file=None, ssebop_link=None):
        """
        Constructor for initializing class variables
        :param input_dir: Input data directory
        :param file_dir: Directory for storing intermediate files
        :param output_dir: Output directory
        :param input_ts_dir: Input directory containing the time series data
        :param output_shp_dir: Output shapefile directory
        :param output_gw_raster_dir: Output GW raster directory
        :param input_state_file: Input state shapefile
        :param input_cdl_file: Input NASS CDL file path
        :param gdal_path: GDAL directory path, in Windows replace with OSGeo4W directory path, e.g. '/usr/bin/gdal/' on
        Linux or Mac and 'C:/OSGeo4W64/' on Windows
        :param input_gw_boundary_file: Input GMD shapefile for Kansas or Well Registry shapefile for Arizona
        :param input_ama_ina_file: The file path to the AMA/INA shapefile required for Arizona. Set None for Kansas.
        :param input_watershed_file: The file path to the Arizona surface watershed shapefile. Set None for Kansas.
        :param ssebop_link: SSEBop data download link. SSEBop data are not downloaded if its set to None.
        """

        self.input_dir = make_proper_dir_name(input_dir)
        self.file_dir = make_proper_dir_name(file_dir)
        self.output_dir = make_proper_dir_name(output_dir)
        self.input_ts_dir = make_proper_dir_name(input_ts_dir)
        self.output_shp_dir = make_proper_dir_name(output_shp_dir)
        self.output_gw_raster_dir = make_proper_dir_name(output_gw_raster_dir)
        self.gdal_path = make_proper_dir_name(gdal_path)
        self.input_gw_boundary_file = input_gw_boundary_file
        self.input_ama_ina_file = input_ama_ina_file
        self.input_watershed_file = input_watershed_file
        self.input_state_file = input_state_file
        self.input_cdl_file = input_cdl_file
        self.ssebop_link = ssebop_link
        self.input_gw_boundary_reproj_file = None
        self.input_ama_ina_reproj_file = None
        self.input_state_reproj_file = None
        self.input_watershed_reproj_file = None
        self.final_gw_dir = None
        self.actual_gw_dir = None
        self.ref_raster = None
        self.raster_reproj_dir = None
        self.reclass_reproj_file = None
        self.raster_mask_dir = None
        self.land_use_dir_list = None
        self.rf_data_dir = None
        self.pred_data_dir = None
        self.lu_mask_dir = None
        self.ssebop_zip_dir = self.input_dir + 'SSEBop_Data/'
        self.ssebop_out_dir = self.ssebop_zip_dir + 'SSEBop_Files/'
        self.ssebop_reproj_dir = self.ssebop_out_dir + 'SSEBop_Reproj/'
        self.ssebop_year_list = None
        self.ssebop_month_list = None
        makedirs([self.output_dir, self.output_gw_raster_dir, self.output_shp_dir])

    def download_ssebop_data(self, year_list, month_list, already_downloaded=False):
        """
        Download, extract, and preprocess SSEBop data
        :param year_list: List of years %yyyy format
        :param month_list: List of months in %m format
        :param already_downloaded: Set True to disable downloading
        :return: None
        """

        self.ssebop_year_list = year_list
        self.ssebop_month_list = month_list
        if not already_downloaded:
            makedirs([self.ssebop_zip_dir, self.ssebop_out_dir])
            pssebop.download_ssebop_data(self.ssebop_link, year_list, month_list, self.ssebop_zip_dir)
            pssebop.extract_data(self.ssebop_zip_dir, self.ssebop_out_dir)
        else:
            print('SSEBop data already downloaded and extracted...')

    def preprocess_gw_csv(self, input_gw_csv_dir, fill_attr='AF Pumped', filter_attr=None,
                          filter_attr_value='OUTSIDE OF AMA OR INA', already_preprocessed=False, **kwargs):
        """
        Preprocess the well registry file to add GW pumping from each CSV file. That is, add an attribute present in the
        GW csv file to the Well Registry shape files (yearwise) based on matching ids given in kwargs.
        By default, the GW withdrawal is added. The csv ids must include: csv_well_id, csv_mov_id, csv_water_id,
        movement_type, water_type, The shp id must include shp_well_id. For the Arizona datasets, csv_well_id='Well Id',
        csv_mov_id='Movement Type', csv_water_id='Water Type', movement_type='WITHDRAWAL', water_type='GROUNDWATER', and
        shp_well_id='REGISTRY_I' by default. For changing, pass appropriate kwargs.
        :param input_gw_csv_dir: Input GW csv directory
        :param fill_attr: Attribute present in the CSV file to add to Well Registry.
        :param filter_attr: Remove specific wells based on this attribute. Set None to disable filtering.
        :param filter_attr_value: Value for filter_attr
        :param already_preprocessed: Set True to disable preprocessing
        :return: None
        """

        if not already_preprocessed:
            input_gw_csv_dir = make_proper_dir_name(input_gw_csv_dir)
            vops.add_attribute_well_reg_multiple(input_well_reg_file=self.input_gw_boundary_file,
                                                 input_gw_csv_dir=input_gw_csv_dir, out_gw_shp_dir=self.output_shp_dir,
                                                 fill_attr=fill_attr, filter_attr=filter_attr,
                                                 filter_attr_value=filter_attr_value, **kwargs)

    def extract_shp_from_gdb(self, input_gdb_dir, year_list, attr_name='AF_USED', already_extracted=False):
        """
        Extract shapefiles from geodatabase (GDB)
        :param input_gdb_dir: Input GDB directory
        :param year_list: List of years to extract
        :param attr_name: Attribute name for shapefile
        :param already_extracted: Set True to disable extraction
        :return: None
        """

        if not already_extracted:
            print('Extracting GW data from GDB...')
            vops.extract_gdb_data(input_gdb_dir, attr_name=attr_name, year_list=year_list, outdir=self.output_shp_dir)
        else:
            print("GW shapefiles already extracted")

    def reproject_shapefiles(self, already_reprojected=False):
        """
        Reproject GMD/Well Registry and state shapefiles
        :param already_reprojected: Set True to disable reprojection
        :return: None
        """

        gw_boundary_reproj_dir = make_proper_dir_name(self.file_dir + 'gw_boundary/reproj')
        gw_ama_ina_reproj_dir = make_proper_dir_name(self.file_dir + 'gw_ama_ina/reproj')
        watershed_reproj_dir = make_proper_dir_name(self.file_dir + 'watershed/reproj')
        state_reproj_dir = make_proper_dir_name(self.file_dir + 'state/reproj')
        self.input_gw_boundary_reproj_file = gw_boundary_reproj_dir + 'input_boundary_reproj.shp'
        if self.input_ama_ina_file:
            self.input_ama_ina_reproj_file = gw_ama_ina_reproj_dir + 'input_ama_ina_reproj.shp'
        if self.input_watershed_file:
            self.input_watershed_reproj_file = watershed_reproj_dir + 'input_watershed_reproj.shp'
        self.input_state_reproj_file = state_reproj_dir + 'input_state_reproj.shp'
        if not already_reprojected:
            print('Reprojecting Boundary/State/AMA_INA/Watershed shapefiles...')
            makedirs([gw_boundary_reproj_dir, state_reproj_dir])
            ref_shp = glob(self.output_shp_dir + '*.shp')[0]
            vops.reproject_vector(self.input_gw_boundary_file, outfile_path=self.input_gw_boundary_reproj_file,
                                  ref_file=ref_shp, raster=False)
            if self.input_ama_ina_file:
                makedirs([gw_ama_ina_reproj_dir])
                vops.reproject_vector(self.input_ama_ina_file, outfile_path=self.input_ama_ina_reproj_file,
                                      ref_file=ref_shp, raster=False)
            if self.input_watershed_file:
                makedirs([watershed_reproj_dir])
                vops.reproject_vector(self.input_watershed_file, outfile_path=self.input_watershed_reproj_file,
                                      ref_file=ref_shp, raster=False)
            vops.reproject_vector(self.input_state_file, outfile_path=self.input_state_reproj_file, ref_file=ref_shp,
                                  raster=False)
        else:
            print('Boundary/State/AMA_INA shapefiles are already reprojected')

    def clip_gw_shpfiles(self, new_clip_file=None, already_clipped=False, extent_clip=True):
        """
        Clip GW shapefiles based on GMD extent
        :param new_clip_file: Input clip file for clipping GW shapefiles (e.g, it could be a watershed shapefile),
        required only if you don't want to clip using GMD extent. Should be in the same projection system
        :param already_clipped: Set False to re-clip shapefiles
        :param extent_clip: Set False to clip by cutline, if shapefile consists of multiple polygons, then this won't
        work
        :return: None
        """

        clip_file = self.input_gw_boundary_reproj_file
        if new_clip_file:
            clip_file = new_clip_file
        clip_shp_dir = make_proper_dir_name(self.output_shp_dir + 'Clipped')
        if not already_clipped:
            print('Clipping GW shapefiles...')
            makedirs([clip_shp_dir])
            vops.clip_vectors(self.output_shp_dir, clip_file=clip_file, outdir=clip_shp_dir, gdal_path=self.gdal_path,
                              extent_clip=extent_clip)
        else:
            print('GW Shapefiles already clipped')
        self.output_shp_dir = clip_shp_dir

    def crop_gw_rasters(self, raster_mask=None, ext_mask=True, use_ama_ina=False, already_cropped=False):
        """
        Crop GW rasters based on a mask, should be called after GW rasters have been created.
        :param raster_mask: Raster mask (shapefile) for cropping raster, required only if crop_rasters=True
        :param ext_mask: Set True to crop by cutline, if shapefile consists of multiple polygons, then this won't
        work and appropriate AMA/INA should be set
        :param use_ama_ina: Use AMA/INA shapefile for cropping (Set True for Arizona).
        :param already_cropped: Set True to disable cropping
        :return: None
        """

        cropped_dir = make_proper_dir_name(self.output_gw_raster_dir + 'Cropped')
        if not already_cropped:
            makedirs([cropped_dir])
            multi_poly = False
            if not raster_mask:
                raster_mask = self.input_state_reproj_file
            if use_ama_ina:
                raster_mask = self.input_ama_ina_reproj_file
                multi_poly = True
            rops.crop_rasters(self.output_gw_raster_dir, outdir=cropped_dir, input_mask_file=raster_mask,
                              ext_mask=ext_mask, gdal_path=self.gdal_path, multi_poly=multi_poly)
        else:
            print('GW rasters already cropped')
        self.final_gw_dir = cropped_dir

    def create_gw_rasters(self, xres=5000., yres=5000., max_gw=1000., value_field=None, value_field_pos=0,
                          convert_units=True, already_created=True):
        """
        Create GW rasters from shapefiles
        :param xres: X-Resolution (map unit)
        :param yres: Y-Resolution (map unit)
        :param max_gw: Maximum GW pumping in mm. Any value higher than this will be set to no data
        :param value_field: Name of the value attribute. Set None to use value_field_pos
        :param value_field_pos: Value field position (zero indexing)
        :param convert_units: If true, converts GW pumping values in acreft to mm
        :param already_created: Set False to re-compute GW pumping rasters
        :return: None
        """

        fixed_dir = make_proper_dir_name(self.output_gw_raster_dir + 'Fixed')
        converted_dir = make_proper_dir_name(self.output_gw_raster_dir + 'Converted')
        if not already_created:
            print('Converting SHP to TIF...')
            makedirs([fixed_dir])
            vops.shps2rasters(self.output_shp_dir, self.output_gw_raster_dir, xres=xres, yres=yres, smoothing=0,
                              value_field=value_field, value_field_pos=value_field_pos, gdal_path=self.gdal_path,
                              gridding=False)
            if convert_units:
                max_gw *= xres * yres / (1.233 * 1e+6)
            rops.fix_large_values(self.output_gw_raster_dir, max_threshold=max_gw, outdir=fixed_dir)
            if convert_units:
                print('Changing GW units from acreft to mm')
                makedirs([converted_dir])
                rops.convert_gw_data(fixed_dir, converted_dir)
        else:
            print('GW  pumping rasters already created')
        if convert_units:
            self.final_gw_dir = converted_dir
        else:
            self.final_gw_dir = fixed_dir
        self.actual_gw_dir = self.final_gw_dir

    def reclassify_cdl(self, reclass_dict, pattern='*.tif', already_reclassified=False):
        """
        Reclassify raster
        :param reclass_dict: Dictionary where key values are tuples representing the interval for reclassification, the
        dictionary values represent the new class
        :param pattern: File pattern required for reprojection
        :param already_reclassified: Set True to disable reclassification
        :return: None
        """

        reclass_dir = make_proper_dir_name(self.file_dir + 'Reclass')
        self.reclass_reproj_file = reclass_dir + 'reclass_reproj.tif'
        self.ref_raster = glob(self.actual_gw_dir + pattern)[0]
        if not already_reclassified:
            print('Reclassifying CDL 2015 data...')
            makedirs([reclass_dir])
            reclass_file = reclass_dir + 'reclass.tif'
            rops.reclassify_raster(self.input_cdl_file, reclass_dict, reclass_file)
            rops.reproject_raster(reclass_file, self.reclass_reproj_file, from_raster=self.ref_raster,
                                  gdal_path=self.gdal_path)
            rops.filter_nans(self.reclass_reproj_file, ref_file=self.ref_raster, outfile_path=self.reclass_reproj_file)
        else:
            print('Already reclassified')

    def reproject_rasters(self, pattern='*.tif', already_reprojected=False):
        """
        Reproject rasters based on GW as reference raster
        :param pattern: File pattern to look for
        :param already_reprojected: Set True to disable raster reprojection
        :return: None
        """

        self.raster_reproj_dir = make_proper_dir_name(self.file_dir + 'Reproj_Rasters')
        if not already_reprojected:
            print('Reprojecting rasters...')
            makedirs([self.raster_reproj_dir])
            rops.reproject_rasters(self.input_ts_dir, ref_raster=self.ref_raster, outdir=self.raster_reproj_dir,
                                   pattern=pattern, gdal_path=self.gdal_path)
            if self.ssebop_link:
                makedirs([self.ssebop_reproj_dir])
                rops.reproject_rasters(self.ssebop_out_dir, ref_raster=self.ref_raster, outdir=self.ssebop_reproj_dir,
                                       pattern=pattern, gdal_path=self.gdal_path)
                pssebop.generate_cummulative_ssebop(self.ssebop_reproj_dir, year_list=self.ssebop_year_list,
                                                    month_list=self.ssebop_month_list, out_dir=self.raster_reproj_dir)
        else:
            print('All rasters already reprojected')

    def create_land_use_rasters(self, class_values=(1, 2, 3), class_labels=('AGRI', 'SW', 'URBAN'),
                                smoothing_factors=(3, 5, 3), already_created=False):
        """
        Create land use rasters from the reclassified raster
        :param class_values: List of land use class values to consider for creating separate rasters
        :param class_labels: List of class_labels ordered according to land_uses
        :param smoothing_factors: Smoothing factor (sigma value for Gaussian filter) to use while smoothing
        :param already_created: Set True to disable land use raster generation
        :return: None
        """

        self.land_use_dir_list = [make_proper_dir_name(self.file_dir + class_label) for class_label in class_labels]
        if not already_created:
            for idx, (class_value, class_label) in enumerate(zip(class_values, class_labels)):
                print('Extracting land use raster for', class_label, '...')
                raster_dir = self.land_use_dir_list[idx]
                intermediate_raster_dir = make_proper_dir_name(raster_dir + 'Intermediate')
                makedirs([intermediate_raster_dir])
                raster_file = intermediate_raster_dir + class_label + '_actual.tif'
                rops.apply_raster_filter2(self.reclass_reproj_file, outfile_path=raster_file, val=class_value)
                raster_masked = intermediate_raster_dir + class_label + '_masked.tif'
                rops.filter_nans(raster_file, self.ref_raster, outfile_path=raster_masked)
                raster_flt = raster_dir + class_label + '_flt.tif'
                rops.apply_gaussian_filter(raster_masked, ref_file=self.ref_raster, outfile_path=raster_flt,
                                           sigma=smoothing_factors[idx], normalize=True, ignore_nan=False)
        else:
            print('Land use rasters already created')

    def create_water_stress_index_rasters(self, pattern_list=('P*.tif', 'SSEBop*.tif', 'AGRI*.tif', 'URBAN*.tif'),
                                          rep_landuse=True, already_created=False):
        """
        Create water stress index rasters based on P, ET, and landuse
        :param pattern_list: Raster pattern list ordered by P, ET (or SSEBop), AGRI, and URBAN
        :param rep_landuse: Set True to replicate landuse raster file paths (should be True if same AGRI and URBAN are
        used for all years)
        :param already_created: Set True to disable water stress raster creation
        :return: None
        """

        ws_out_dir = make_proper_dir_name(self.file_dir + 'WS_Rasters')
        makedirs([ws_out_dir])
        if not already_created:
            input_raster_dir_list = [self.raster_reproj_dir] * 2 + [self.land_use_dir_list[0],
                                                                    self.land_use_dir_list[2]]
            rops.compute_water_stress_index_rasters(self.input_watershed_reproj_file, pattern_list=pattern_list,
                                                    input_raster_dir_list=input_raster_dir_list,
                                                    rep_landuse=rep_landuse, output_dir=ws_out_dir,
                                                    gdal_path=self.gdal_path)
            rops.reproject_rasters(ws_out_dir, ref_raster=self.ref_raster, outdir=self.raster_reproj_dir,
                                   pattern='*.tif', gdal_path=self.gdal_path)
        else:
            print('Water stress rasters already created')

    def mask_rasters(self, pattern='*.tif', already_masked=False):
        """
        Mask rasters based on reference GW raster
        :param pattern: File pattern to look for
        :param already_masked: Set True to disable raster masking
        :return: None
        """

        self.ref_raster = glob(self.final_gw_dir + pattern)[0]
        self.raster_mask_dir = make_proper_dir_name(self.file_dir + 'Masked_Rasters')
        self.lu_mask_dir = make_proper_dir_name(self.raster_mask_dir + 'Masked_LU')
        if not already_masked:
            print('Masking rasters...')
            makedirs([self.raster_mask_dir, self.lu_mask_dir])
            rops.mask_rasters(self.raster_reproj_dir, ref_raster=self.ref_raster, outdir=self.raster_mask_dir,
                              pattern=pattern)
            for lu_dir in self.land_use_dir_list:
                rops.mask_rasters(lu_dir, ref_raster=self.ref_raster, outdir=self.lu_mask_dir, pattern=pattern)
        else:
            print('All rasters already masked')

    def create_dataframe(self, year_list, column_names=None, ordering=False, load_df=False, exclude_vars=(),
                         exclude_years=(2019, ), pattern='*.tif', verbose=True):
        """
        Create dataframe from preprocessed files
        :param year_list: List of years for which the dataframe will be created
        :param column_names: Dataframe column names, these must be df headers
        :param ordering: Set True to order dataframe column names
        :param load_df: Set true to load existing dataframe
        :param exclude_vars: Exclude these variables from the dataframe
        :param exclude_years: List of years to exclude from dataframe
        :param pattern: File pattern
        :param verbose: Get extra information if set to True
        :return: Pandas dataframe object
        """

        self.rf_data_dir = make_proper_dir_name(self.file_dir + 'RF_Data')
        self.pred_data_dir = make_proper_dir_name(self.file_dir + 'Pred_Data')
        df_file = self.output_dir + 'raster_df.csv'
        if load_df:
            print('Getting dataframe...')
            return pd.read_csv(df_file)
        else:
            print('Copying files...')
            makedirs([self.rf_data_dir, self.pred_data_dir])
            input_dir_list = [self.final_gw_dir] + [self.raster_mask_dir]
            pattern_list = [pattern] * len(input_dir_list)
            copy_files(input_dir_list, target_dir=self.rf_data_dir, year_list=year_list, pattern_list=pattern_list,
                       verbose=verbose)
            copy_files([self.lu_mask_dir], target_dir=self.rf_data_dir, year_list=year_list,
                       pattern_list=[pattern], rep=True, verbose=verbose)

            input_dir_list = [self.actual_gw_dir] + [self.raster_reproj_dir]
            pattern_list = [pattern] * len(input_dir_list)
            copy_files(input_dir_list, target_dir=self.pred_data_dir, year_list=year_list, pattern_list=pattern_list,
                       verbose=verbose)
            pattern_list = [pattern] * len(self.land_use_dir_list)
            copy_files(self.land_use_dir_list, target_dir=self.pred_data_dir, year_list=year_list,
                       pattern_list=pattern_list, rep=True, verbose=verbose)
            print('Creating dataframe...')
            df = rfr.create_dataframe(self.rf_data_dir, out_df=df_file, column_names=column_names, make_year_col=True,
                                      exclude_vars=exclude_vars, exclude_years=exclude_years, ordering=ordering)
            return df

    def tune_parameters(self, df, pred_attr, drop_attrs=()):
        """
        Tune random forest hyperparameters
        :param df: Input pandas dataframe object
        :param pred_attr: Target attribute
        :param drop_attrs: List of attributes to drop from the df
        :return: None
        """

        n_features = len(df.columns) - len(drop_attrs) - 1
        test_cases = [2014, 2012, range(2012, 2017)]
        est_range = range(100, 601, 100)
        f_range = range(1, n_features + 1)
        ts = []
        for n in range(1, len(test_cases) + 1):
            ts.append('T' + str(n))
        for ne in est_range:
            for nf in f_range:
                for y, t in zip(test_cases, ts):
                    if not isinstance(y, range):
                        ty = (y,)
                    else:
                        ty = y
                    rfr.rf_regressor(df, self.output_dir, n_estimators=ne, random_state=0, pred_attr=pred_attr,
                                     drop_attrs=drop_attrs, test_year=ty, shuffle=False, plot_graphs=False,
                                     split_yearly=True, bootstrap=True, max_features=nf, test_case=t)

    def build_model(self, df, n_estimators=100, random_state=0, bootstrap=True, max_features=3, test_size=None,
                    pred_attr='GW', shuffle=False, plot_graphs=False, plot_3d=False, drop_attrs=(), test_year=(2012,),
                    split_yearly=True, load_model=False):
        """
        Build random forest model
        :param df: Input pandas dataframe object
        :param pred_attr: Target attribute
        :param drop_attrs: List of attributes to drop from the df
        :param n_estimators: RF hyperparameter
        :param random_state: RF hyperparameter
        :param bootstrap: RF hyperparameter
        :param max_features: RF hyperparameter
        :param test_size: Required only if split_yearly=False
        :param pred_attr: Prediction attribute name in the dataframe
        :param shuffle: Set False to stop data shuffling
        :param plot_graphs: Plot Actual vs Prediction graph
        :param plot_3d: Plot pairwise 3D partial dependence plots
        :param drop_attrs: Drop these specified attributes
        :param test_year: Build test data from only this year(s).
        :param split_yearly: Split train test data based on years
        :param load_model: Load an earlier pre-trained RF model
        :return: Fitted RandomForestRegressor object
        """

        print('Building RF Model...')
        plot_dir = make_proper_dir_name(self.output_dir + 'Partial_Plots/PDP_Data')
        makedirs([plot_dir])
        rf_model = rfr.rf_regressor(df, self.output_dir, n_estimators=n_estimators, random_state=random_state,
                                    pred_attr=pred_attr, drop_attrs=drop_attrs, test_year=test_year, shuffle=shuffle,
                                    plot_graphs=plot_graphs, plot_3d=plot_3d, split_yearly=split_yearly,
                                    bootstrap=bootstrap, plot_dir=plot_dir, max_features=max_features,
                                    load_model=load_model, test_size=test_size)
        return rf_model

    def get_predictions(self, rf_model, pred_years, column_names=None, ordering=False, pred_attr='GW',
                        only_pred=False, exclude_vars=(), exclude_years=(2019,), drop_attrs=()):
        """
        Get prediction results and/or rasters
        :param rf_model: Fitted RandomForestRegressor model
        :param pred_years: Predict for these years
        :param column_names: Dataframe column names, these must be df headers
        :param ordering: Set True to order dataframe column names
        :param pred_attr: Prediction attribute name in the dataframe
        :param only_pred: Set True to disable prediction raster generation
        :param exclude_vars: Exclude these variables from the model prediction analysis
        :param exclude_years: List of years to exclude from dataframe
        :param drop_attrs: Drop these specified attributes
        :return: Predicted raster directory path
        """

        print('Predicting...')
        pred_out_dir = make_proper_dir_name(self.output_dir + 'Predicted_Rasters')
        makedirs([pred_out_dir])
        rfr.predict_rasters(rf_model, pred_years=pred_years, drop_attrs=drop_attrs, out_dir=pred_out_dir,
                            actual_raster_dir=self.rf_data_dir, pred_attr=pred_attr, only_pred=only_pred,
                            exclude_vars=exclude_vars, exclude_years=exclude_years, column_names=column_names,
                            ordering=ordering)
        return pred_out_dir


def run_gw_ks(analyze_only=False, load_files=True, load_rf_model=False, use_gmds=True):
    """
    Main function for running the project for Kansas, some variables require to be hardcoded
    :param analyze_only: Set True to just produce analysis results, all required files must be present
    :param load_files: Set True to load existing files, needed only if analyze_only=False
    :param load_rf_model: Set True to load existing Random Forest model, needed only if analyze_only=False
    :param use_gmds: Set False to use entire GW raster for analysis
    :return: None
    """

    gee_data = ['Apr_Sept/', 'Apr_Aug/', 'Annual/']
    input_dir = '../Inputs/Data/Kansas_GW/'
    file_dir = '../Inputs/Files_KS_' + gee_data[0]
    output_dir = '../Outputs/Output_KS_' + gee_data[0]
    input_ts_dir = input_dir + 'GEE_Data_' + gee_data[0]
    output_shp_dir = file_dir + 'GW_Shapefiles/'
    output_gw_raster_dir = file_dir + 'GW_Rasters/'
    input_gmd_file = input_dir + 'gmds/ks_gmds.shp'
    input_cdl_file = input_dir + 'CDL/CDL_KS_2015.tif'
    input_gdb_dir = input_dir + 'ks_pd_data_updated2018.gdb'
    input_state_file = input_dir + 'Kansas/kansas.shp'
    gdal_path = 'C:/OSGeo4W64/'
    gw_dir = file_dir + 'RF_Data/'
    pred_gw_dir = output_dir + 'Predicted_Rasters/'
    grace_csv = input_dir + 'GRACE/TWS_GRACE.csv'
    ks_class_dict = {(0, 59.5): 1,
                     (66.5, 77.5): 1,
                     (203.5, 255): 1,
                     (110.5, 111.5): 2,
                     (111.5, 112.5): 0,
                     (120.5, 124.5): 3,
                     (59.5, 61.5): 0,
                     (130.5, 195.5): 0
                     }
    drop_attrs = ('YEAR',)
    pred_attr = 'GW'
    if not analyze_only:
        gw = HydroML(input_dir, file_dir, output_dir, input_ts_dir, output_shp_dir, output_gw_raster_dir,
                     input_state_file, input_cdl_file, gdal_path, input_gw_boundary_file=input_gmd_file)
        gw.extract_shp_from_gdb(input_gdb_dir, year_list=range(2002, 2019), already_extracted=load_files)
        gw.reproject_shapefiles(already_reprojected=load_files)
        gw.create_gw_rasters(already_created=load_files)
        gw.reclassify_cdl(ks_class_dict, already_reclassified=load_files)
        gw.reproject_rasters(already_reprojected=load_files)
        gw.create_land_use_rasters(already_created=load_files)
        gw.mask_rasters(already_masked=load_files)
        df = gw.create_dataframe(year_list=range(2002, 2020), load_df=load_files)
        rf_model = gw.build_model(df, test_year=range(2011, 2019), drop_attrs=drop_attrs, pred_attr=pred_attr,
                                  load_model=load_rf_model, max_features=5, plot_graphs=False)
        pred_gw_dir = gw.get_predictions(rf_model=rf_model, pred_years=range(2002, 2020), drop_attrs=drop_attrs,
                                         pred_attr=pred_attr, only_pred=False)
    ma.run_analysis(gw_dir, pred_gw_dir, grace_csv, use_gmds=use_gmds, input_gmd_file=input_gmd_file,
                    out_dir=output_dir)


def run_gw_az(analyze_only=False, load_files=True, load_rf_model=False):
    """
    Main function for running the project for Arizona, some variables require to be hardcoded
    :param analyze_only: Set True to just produce analysis results, all required files must be present
    :param load_files: Set True to load existing files, needed only if analyze_only=False
    :param load_rf_model: Set True to load existing Random Forest model, needed only if analyze_only=False
    :return: None
    """

    gee_data = ['Apr_Sept/', 'Apr_Aug/', 'Annual/']
    input_dir = '../Inputs/Data/Arizona_GW/'
    file_dir = '../Inputs/Files_AZ_' + gee_data[0]
    output_dir = '../Outputs/Output_AZ_' + gee_data[0]
    input_ts_dir = input_dir + 'GEE_Data_' + gee_data[0]
    output_shp_dir = file_dir + 'GW_Shapefiles/'
    output_gw_raster_dir = file_dir + 'GW_Rasters/'
    input_well_reg_file = input_dir + 'Well_Registry/WellRegistry.shp'
    input_ama_ina_file = input_dir + 'Boundary/AMA_and_INA.shp'
    input_cdl_file = input_dir + 'CDL/CDL_AZ_2015.tif'
    input_watershed_file = input_dir + 'Watersheds/Surface_Watershed.shp'
    input_gw_csv_dir = input_dir + 'GW_Data/'
    input_state_file = input_dir + 'Arizona/Arizona.shp'
    gdal_path = 'C:/OSGeo4W64/'
    gw_dir = file_dir + 'RF_Data/'
    pred_gw_dir = output_dir + 'Predicted_Rasters/'
    grace_csv = input_dir + 'GRACE/TWS_GRACE.csv'
    ssebop_link = 'https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/uswem/web/conus/eta/modis_eta/monthly/' \
                  'downloads/'
    ssebop_year_list = range(2002, 2020)
    ssebop_month_list = range(4, 10)
    az_class_dict = {(0, 59.5): 1,
                     (66.5, 77.5): 1,
                     (203.5, 255): 1,
                     (110.5, 111.5): 2,
                     (111.5, 112.5): 0,
                     (120.5, 124.5): 3,
                     (59.5, 61.5): 0,
                     (130.5, 195.5): 0
                     }
    drop_attrs = ('YEAR',)
    exclude_vars = ('ET',)
    pred_attr = 'GW'
    fill_attr = 'AF Pumped'
    filter_attr = None
    if not analyze_only:
        gw = HydroML(input_dir, file_dir, output_dir, input_ts_dir, output_shp_dir, output_gw_raster_dir,
                     input_state_file, input_cdl_file, gdal_path, input_gw_boundary_file=input_well_reg_file,
                     input_ama_ina_file=input_ama_ina_file, input_watershed_file=input_watershed_file,
                     ssebop_link=ssebop_link)
        gw.download_ssebop_data(year_list=ssebop_year_list, month_list=ssebop_month_list, already_downloaded=True)
        gw.preprocess_gw_csv(input_gw_csv_dir, fill_attr=fill_attr, filter_attr=filter_attr,
                             already_preprocessed=load_files)
        gw.reproject_shapefiles(already_reprojected=load_files)
        gw.create_gw_rasters(already_created=load_files, value_field=fill_attr, xres=1000, yres=1000, max_gw=1e+5)
        gw.crop_gw_rasters(use_ama_ina=True, already_cropped=load_files)
        gw.reclassify_cdl(az_class_dict, already_reclassified=load_files)
        gw.reproject_rasters(already_reprojected=load_files)
        gw.create_land_use_rasters(already_created=load_files, smoothing_factors=(3, 5, 3))
        gw.create_water_stress_index_rasters(already_created=load_files)
        gw.mask_rasters(already_masked=load_files)
        df = gw.create_dataframe(year_list=range(2002, 2019), exclude_vars=exclude_vars, exclude_years=(),
                                 load_df=False)
        rf_model = gw.build_model(df, test_year=range(2011, 2019), drop_attrs=drop_attrs, pred_attr=pred_attr,
                                  load_model=load_rf_model, max_features=7, plot_graphs=False)
        pred_gw_dir = gw.get_predictions(rf_model=rf_model, pred_years=range(2002, 2019), drop_attrs=drop_attrs,
                                         pred_attr=pred_attr, exclude_vars=exclude_vars, exclude_years=(),
                                         only_pred=False)
    ma.run_analysis(gw_dir, pred_gw_dir, grace_csv, use_gmds=False, input_gmd_file=None, out_dir=output_dir)


# run_gw_ks(analyze_only=False, load_files=False, load_rf_model=False, use_gmds=True)
run_gw_az(analyze_only=False, load_files=True, load_rf_model=False)
