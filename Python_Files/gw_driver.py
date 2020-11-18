# Author: Sayantan Majumdar
# Email: smxnv@mst.edu

import pandas as pd
from Python_Files.hydrolibs import rasterops as rops
from Python_Files.hydrolibs import vectorops as vops
from Python_Files.hydrolibs import data_download as dd
from Python_Files.hydrolibs.sysops import makedirs, make_proper_dir_name, copy_files
from Python_Files.hydrolibs import random_forest_regressor as rfr
from Python_Files.hydrolibs import model_analysis as ma
from glob import glob


class HydroML:

    def __init__(self, input_dir, file_dir, output_dir, output_shp_dir, output_gw_raster_dir,
                 input_state_file, gdal_path, input_ts_dir=None, input_subsidence_dir=None, input_cdl_file=None,
                 input_gw_boundary_file=None, input_ama_ina_file=None, input_watershed_file=None, ssebop_link=None):
        """
        Constructor for initializing class variables
        :param input_dir: Input data directory
        :param file_dir: Directory for storing intermediate files
        :param output_dir: Output directory
        :param output_shp_dir: Output shapefile directory
        :param output_gw_raster_dir: Output GW raster directory
        :param input_state_file: Input state shapefile (must be in WGS84)
        :param input_cdl_file: Input NASS CDL file path, set None to automatically download
        :param gdal_path: GDAL directory path, in Windows replace with OSGeo4W directory path, e.g. '/usr/bin/gdal/' on
        Linux or Mac and 'C:/OSGeo4W64/' on Windows
        :param input_ts_dir: Input directory containing the time series data, set None to automatically download data
        :param input_gw_boundary_file: Input GMD shapefile for Kansas or Well Registry shapefile for Arizona
        :param input_ama_ina_file: The file path to the AMA/INA shapefile required for Arizona. Set None for Kansas.
        :param input_watershed_file: The file path to the Arizona surface watershed shapefile. Set None for Kansas.
        :param ssebop_link: SSEBop data download link. SSEBop data are not downloaded if its set to None.
        """

        self.input_dir = make_proper_dir_name(input_dir)
        self.file_dir = make_proper_dir_name(file_dir)
        self.output_dir = make_proper_dir_name(output_dir)
        self.output_shp_dir = make_proper_dir_name(output_shp_dir)
        self.output_gw_raster_dir = make_proper_dir_name(output_gw_raster_dir)
        self.gdal_path = make_proper_dir_name(gdal_path)
        self.input_ts_dir = make_proper_dir_name(input_ts_dir)
        self.input_subsidence_dir = make_proper_dir_name(input_subsidence_dir)
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
        self.crop_coeff_dir = None
        self.crop_coeff_reproj_dir = None
        self.crop_coeff_mask_dir = None
        self.reclass_reproj_file = None
        self.raster_mask_dir = None
        self.land_use_dir_list = None
        self.rf_data_dir = None
        self.pred_data_dir = None
        self.lu_mask_dir = None
        self.ssebop_file_dir = None
        self.ssebop_reproj_dir = None
        self.ws_ssebop_file_dir = None
        self.ws_ssebop_reproj_dir = None
        self.data_year_list = None
        self.data_start_month = None
        self.data_end_month = None
        self.ws_year_list = None
        self.ws_start_month = None
        self.ws_end_month = None
        self.ws_data_dir = None
        self.ws_data_reproj_dir = None
        self.converted_subsidence_dir = None
        self.pred_out_dir = None
        self.subsidence_pred_gw_dir = None
        makedirs([self.output_dir, self.output_gw_raster_dir, self.output_shp_dir])

    def download_data(self, year_list, start_month, end_month, cdl_year=2015, already_downloaded=False,
                      already_extracted=False):
        """
        Download, extract, and preprocess SSEBop data
        :param year_list: List of years %yyyy format
        :param start_month: Start month in %m format
        :param end_month: End month in %m format
        :param cdl_year: CDL year
        :param already_downloaded: Set True to disable downloading
        :param already_extracted: Set True to disable extraction
        :return: None
        """

        self.data_year_list = year_list
        self.data_start_month = start_month
        self.data_end_month = end_month
        gee_data_flag = False
        if self.input_ts_dir is None:
            self.input_ts_dir = self.input_dir + 'Downloaded_Data/'
            gee_data_flag = True
        gee_zip_dir = self.input_ts_dir + 'GEE_Data/'
        cdl_flag = False
        cdl_dir = self.input_ts_dir + 'CDL/'
        if self.input_cdl_file is None:
            self.input_cdl_file = cdl_dir + 'CDL_' + str(cdl_year) + '.tif'
            cdl_flag = True
        ssebop_zip_dir = self.input_ts_dir + 'SSEBop_Data/'
        self.ssebop_file_dir = ssebop_zip_dir + 'SSEBop_Files/'
        if not already_downloaded:
            if cdl_flag:
                makedirs([cdl_dir])
                dd.download_cropland_data(self.input_state_file, year=cdl_year, outfile=self.input_cdl_file)
            if gee_data_flag:
                makedirs([gee_zip_dir])
                dd.download_gee_data(year_list, start_month=start_month, end_month=end_month,
                                     aoi_shp_file=self.input_state_file, outdir=gee_zip_dir)
            makedirs([ssebop_zip_dir])
            dd.download_ssebop_data(self.ssebop_link, year_list, start_month, end_month, ssebop_zip_dir)
        if gee_data_flag:
            self.input_ts_dir = gee_zip_dir + 'GEE_Files/'
        if not already_extracted:
            if gee_data_flag:
                makedirs([self.input_ts_dir])
                dd.extract_data(gee_zip_dir, out_dir=self.input_ts_dir, rename_extracted_files=True)
            makedirs([self.ssebop_file_dir])
            dd.extract_data(ssebop_zip_dir, self.ssebop_file_dir)
        print('CDL, GEE, and SSEBop data downloaded and extracted...')

    def download_ws_data(self, year_list, start_month, end_month, already_downloaded=False, already_extracted=False):
        """
        Download SSEBop and P data for water stress index computation
        :param year_list: List of years %yyyy format
        :param start_month: Start month in %m format
        :param end_month: End month in %m format
        :param already_downloaded: Set True to disable downloading
        :param already_extracted: Set True to disable extraction
        :return: None
        """

        self.ws_year_list = year_list
        self.ws_start_month = start_month
        self.ws_end_month = end_month
        self.ws_data_dir = self.input_dir + 'WS_Data/'
        ws_gee_dir = self.ws_data_dir + 'WS_GEE/'
        ws_ssebop_dir = self.ws_data_dir + 'WS_SSEBop/'
        self.ws_ssebop_file_dir = ws_ssebop_dir + 'WS_SSEBop_Files/'
        if not already_downloaded:
            makedirs([ws_gee_dir, ws_ssebop_dir])
            dd.download_ssebop_data(self.ssebop_link, year_list, start_month, end_month, ws_ssebop_dir)
            dd.download_gee_data(year_list, start_month=start_month, end_month=end_month,
                                 aoi_shp_file=self.input_state_file, outdir=ws_gee_dir)

        if not already_extracted:
            makedirs([self.ws_ssebop_file_dir])
            dd.extract_data(ws_ssebop_dir, out_dir=self.ws_ssebop_file_dir)
            dd.extract_data(ws_gee_dir, out_dir=self.ws_data_dir, rename_extracted_files=True)
        print('Data for WS metric downloaded...')

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
            rops.crop_rasters(self.final_gw_dir, outdir=cropped_dir, input_mask_file=raster_mask, ext_mask=ext_mask,
                              gdal_path=self.gdal_path, multi_poly=multi_poly)
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

    def create_crop_coeff_raster(self, already_created=False):
        """
        Create crop coefficient raster based on the NASS CDL file
        :param already_created: Set True to disable raster creation
        :return: None
        """

        self.crop_coeff_dir = make_proper_dir_name(self.file_dir + 'Crop_Coeff')
        crop_coeff_file = self.crop_coeff_dir + 'Crop_Coeff.tif'
        if not already_created:
            print('Creating crop coefficient raster...')
            makedirs([self.crop_coeff_dir])
            rops.create_crop_coeff_raster(self.input_cdl_file, output_file=crop_coeff_file)

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

    def organize_subsidence_rasters(self, decorrelated_value=-10000, verbose=False, already_organized=False):
        """
        Organize ADWR subsidence rasters and then create resampled subsidence rasters
        :param decorrelated_value: Decorrelated pixel value for subsidence rasters, these would be set to no data
        :param verbose: Set True to get additional info
        :param already_organized: Set True to disable organizing subsidence rasters
        :return: None
        """

        self.converted_subsidence_dir = self.file_dir + 'Converted_Subsidence_Rasters/'
        if not already_organized:
            print('Organizing subsidence rasters...')
            makedirs([self.converted_subsidence_dir])
            rops.organize_subsidence_data(self.input_subsidence_dir, output_dir=self.converted_subsidence_dir,
                                          ref_raster=self.ref_raster, gdal_path=self.gdal_path,
                                          decorrelated_value=decorrelated_value, verbose=verbose)
        print('Organized and created subsidence rasters...')

    def reproject_rasters(self, pattern='*.tif', already_reprojected=False):
        """
        Reproject rasters based on GW as reference raster
        :param pattern: File pattern to look for
        :param already_reprojected: Set True to disable raster reprojection
        :return: None
        """

        self.raster_reproj_dir = self.file_dir + 'Reproj_Rasters/'
        self.ssebop_reproj_dir = self.ssebop_file_dir + 'SSEBop_Reproj/'
        self.ws_data_reproj_dir = self.file_dir + 'WS_Reproj_Rasters/'
        self.ws_ssebop_reproj_dir = self.file_dir + 'WS_SSEBop_Reproj_Rasters/'
        self.crop_coeff_reproj_dir = self.crop_coeff_dir + 'Crop_Coeff_Reproj/'

        if not already_reprojected:
            print('Reprojecting rasters...')
            makedirs([self.raster_reproj_dir, self.crop_coeff_reproj_dir])
            rops.reproject_rasters(self.input_ts_dir, ref_raster=self.ref_raster, outdir=self.raster_reproj_dir,
                                   pattern=pattern, gdal_path=self.gdal_path)
            rops.reproject_rasters(self.crop_coeff_dir, ref_raster=self.ref_raster,
                                   outdir=self.crop_coeff_reproj_dir, pattern=pattern, gdal_path=self.gdal_path)
            if self.ssebop_link:
                makedirs([self.ssebop_reproj_dir, self.ws_ssebop_reproj_dir, self.ws_data_reproj_dir])
                rops.reproject_rasters(self.ssebop_file_dir, ref_raster=self.ref_raster, outdir=self.ssebop_reproj_dir,
                                       pattern=pattern, gdal_path=self.gdal_path)
                rops.generate_cummulative_ssebop(self.ssebop_reproj_dir, year_list=self.data_year_list,
                                                 start_month=self.data_start_month, end_month=self.data_end_month,
                                                 out_dir=self.raster_reproj_dir)
                if self.ws_year_list is not None:
                    rops.reproject_rasters(self.ws_ssebop_file_dir, ref_raster=self.ref_raster,
                                           outdir=self.ws_ssebop_reproj_dir, pattern=pattern, gdal_path=self.gdal_path)
                    rops.generate_cummulative_ssebop(self.ws_ssebop_reproj_dir, year_list=self.ws_year_list,
                                                     start_month=self.ws_start_month, end_month=self.ws_end_month,
                                                     out_dir=self.ws_data_reproj_dir)
                    rops.reproject_rasters(self.ws_data_dir, ref_raster=self.ref_raster, outdir=self.ws_data_reproj_dir,
                                           pattern=pattern, gdal_path=self.gdal_path)
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

    def update_crop_coeff_raster(self, pattern='*.tif', already_updated=False):
        """
        Update crop coefficient raster according to AGRI
        :param pattern: AGRI raster file pattern
        :param already_updated: Set True to disable updation
        :return: None
        """

        if not already_updated:
            print('Updating crop coefficient raster based on AGRI...')
            crop_coeff_file = glob(self.crop_coeff_reproj_dir + pattern)[0]
            rops.update_crop_coeff_raster(crop_coeff_file, self.reclass_reproj_file)
        print('Crop coefficients updated!')

    def create_water_stress_index_rasters(self, pattern_list=('P*.tif', 'SSEBop*.tif', 'AGRI*.tif', 'URBAN*.tif'),
                                          rep_landuse=True, already_created=False, normalize=False):
        """
        Create water stress index rasters based on P, ET, and landuse
        :param pattern_list: Raster pattern list ordered by P, ET (or SSEBop), AGRI, and URBAN
        :param rep_landuse: Set True to replicate landuse raster file paths (should be True if same AGRI and URBAN are
        used for all years)
        :param already_created: Set True to disable water stress raster creation
        :param normalize: Set True to normalize water stress index
        :return: None
        """

        ws_out_dir = make_proper_dir_name(self.file_dir + 'WS_Rasters')
        makedirs([ws_out_dir])
        if not already_created:
            input_raster_dir_list = [self.ws_data_reproj_dir] * 2 + [self.land_use_dir_list[0],
                                                                     self.land_use_dir_list[2]]
            rops.compute_water_stress_index_rasters(self.input_watershed_reproj_file, pattern_list=pattern_list,
                                                    input_raster_dir_list=input_raster_dir_list,
                                                    rep_landuse=rep_landuse, output_dir=ws_out_dir,
                                                    gdal_path=self.gdal_path, normalize=normalize)
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
        self.crop_coeff_mask_dir = make_proper_dir_name(self.raster_mask_dir + 'Masked_Crop_Coeff')
        if not already_masked:
            print('Masking rasters...')
            makedirs([self.raster_mask_dir, self.lu_mask_dir, self.crop_coeff_mask_dir])
            rops.mask_rasters(self.raster_reproj_dir, ref_raster=self.ref_raster, outdir=self.raster_mask_dir,
                              pattern=pattern)
            rops.mask_rasters(self.crop_coeff_reproj_dir, ref_raster=self.ref_raster, outdir=self.crop_coeff_mask_dir,
                              pattern=pattern)
            for lu_dir in self.land_use_dir_list:
                rops.mask_rasters(lu_dir, ref_raster=self.ref_raster, outdir=self.lu_mask_dir, pattern=pattern)
        else:
            print('All rasters already masked')

    def create_dataframe(self, year_list, column_names=None, ordering=False, load_df=False, exclude_vars=(),
                         exclude_years=(2019, ), pattern='*.tif', verbose=True, remove_na=True):
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
        :param remove_na: Set False to disable NA removal
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
            copy_files([self.crop_coeff_mask_dir], target_dir=self.rf_data_dir, year_list=year_list,
                       pattern_list=[pattern], rep=True, verbose=verbose)
            copy_files([self.lu_mask_dir], target_dir=self.rf_data_dir, year_list=year_list,
                       pattern_list=[pattern], rep=True, verbose=verbose)

            input_dir_list = [self.actual_gw_dir] + [self.raster_reproj_dir]
            pattern_list = [pattern] * len(input_dir_list)
            copy_files(input_dir_list, target_dir=self.pred_data_dir, year_list=year_list, pattern_list=pattern_list,
                       verbose=verbose)
            pattern_list = [pattern] * len(self.land_use_dir_list)
            copy_files(self.land_use_dir_list, target_dir=self.pred_data_dir, year_list=year_list,
                       pattern_list=pattern_list, rep=True, verbose=verbose)
            copy_files([self.crop_coeff_reproj_dir], target_dir=self.pred_data_dir, year_list=year_list,
                       pattern_list=[pattern], rep=True, verbose=verbose)
            print('Creating dataframe...')
            df = rfr.create_dataframe(self.rf_data_dir, out_df=df_file, column_names=column_names, make_year_col=True,
                                      exclude_vars=exclude_vars, exclude_years=exclude_years, ordering=ordering,
                                      remove_na=remove_na)
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
                    split_yearly=True, load_model=False, calc_perm_imp=False):
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
        :param calc_perm_imp: Set True to get permutation importances on train and test data
        :return: Fitted RandomForestRegressor object
        """

        print('Building RF Model...')
        plot_dir = make_proper_dir_name(self.output_dir + 'Partial_Plots/PDP_Data')
        makedirs([plot_dir])
        rf_model = rfr.rf_regressor(df, self.output_dir, n_estimators=n_estimators, random_state=random_state,
                                    pred_attr=pred_attr, drop_attrs=drop_attrs, test_year=test_year, shuffle=shuffle,
                                    plot_graphs=plot_graphs, plot_3d=plot_3d, split_yearly=split_yearly,
                                    bootstrap=bootstrap, plot_dir=plot_dir, max_features=max_features,
                                    load_model=load_model, test_size=test_size, calc_perm_imp=calc_perm_imp)
        return rf_model

    def get_predictions(self, rf_model, pred_years, column_names=None, ordering=False, pred_attr='GW',
                        only_pred=False, exclude_vars=(), exclude_years=(2019,), drop_attrs=(), use_full_extent=False):
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
        :param use_full_extent: Set True to predict over entire region
        :return: Actual and Predicted raster directory paths
        """

        print('Predicting...')
        self.pred_out_dir = make_proper_dir_name(self.output_dir + 'Predicted_Rasters')
        makedirs([self.pred_out_dir])
        actual_raster_dir = self.rf_data_dir
        if use_full_extent:
            actual_raster_dir = self.pred_data_dir
        rfr.predict_rasters(rf_model, pred_years=pred_years, drop_attrs=drop_attrs, out_dir=self.pred_out_dir,
                            actual_raster_dir=actual_raster_dir, pred_attr=pred_attr, only_pred=only_pred,
                            exclude_vars=exclude_vars, exclude_years=exclude_years, column_names=column_names,
                            ordering=ordering)
        return actual_raster_dir, self.pred_out_dir

    def create_subsidence_pred_gw_rasters(self, scale_to_cm=True, verbose=False, already_created=False):
        """
        Create total predicted GW withdrawal rasters based on subsidence years
        :param scale_to_cm: Set False to disable scaling GW and subsidence data to cm, default GW is in mm and
        subsidence is in m
        :param verbose: Set True to get additional info
        :param already_created: Set True to disable creating these rasters
        :return: None
        """

        self.subsidence_pred_gw_dir = make_proper_dir_name(self.output_dir + 'Subsidence_Analysis')
        if not already_created:
            makedirs([self.subsidence_pred_gw_dir])
            rops.create_subsidence_pred_gw_rasters(self.pred_out_dir, self.converted_subsidence_dir,
                                                   self.subsidence_pred_gw_dir, scale_to_cm=scale_to_cm,
                                                   verbose=verbose)
        print('Subsidence and total predicted GW rasters created!')


def run_gw_ks(analyze_only=False, load_files=True, load_rf_model=False, use_gmds=True, show_box_plots=False,
              load_df=False, show_train_test_box_plots=False, build_ml_model=True):
    """
    Main function for running the project, some variables require to be hardcoded
    :param analyze_only: Set True to just produce analysis results, all required files must be present
    :param load_files: Set True to load existing files, needed only if analyze_only=False
    :param load_rf_model: Set True to load existing Random Forest model, needed only if analyze_only=False
    :param use_gmds: Set False to use entire GW raster for analysis
    :param show_box_plots: Set True to display box plots of features
    :param load_df: Set True to load existing dataframe from CSV
    :param show_train_test_box_plots: Set True to show box plots of predictors for train and test data separately
    :param build_ml_model: Set False to disable model building. If True, MOD16 will be used instead of SSEBop as it is a
    better predictor for Kansas
    :return: Returns HydroML object and Pandas dataframe
    """

    gee_data = ['Apr_Sept/', 'Apr_Aug/', 'Annual/']
    input_dir = '../Inputs/Data/Kansas_GW/'
    file_dir = '../Inputs/Files_KS_' + gee_data[0]
    output_dir = '../Outputs/Output_KS_' + gee_data[0]
    output_shp_dir = file_dir + 'GW_Shapefiles/'
    output_gw_raster_dir = file_dir + 'GW_Rasters/'
    input_gmd_file = input_dir + 'gmds/ks_gmds.shp'
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
    exclude_vars = ('ET',)
    if build_ml_model:
        exclude_vars = ('SSEBop',)
    drop_attrs = ('YEAR',)
    pred_attr = 'GW'
    ssebop_link = 'https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/uswem/web/conus/eta/modis_eta/monthly/' \
                  'downloads/'
    data_year_list = range(2002, 2020)
    data_start_month = 4
    data_end_month = 9
    df = pd.DataFrame()
    gw = None
    if not analyze_only:
        gw = HydroML(input_dir, file_dir, output_dir, output_shp_dir, output_gw_raster_dir, input_state_file,
                     gdal_path, input_gw_boundary_file=input_gmd_file, ssebop_link=ssebop_link)
        gw.download_data(year_list=data_year_list, start_month=data_start_month, end_month=data_end_month,
                         already_downloaded=load_files, already_extracted=load_files)
        gw.extract_shp_from_gdb(input_gdb_dir, year_list=range(2002, 2019), already_extracted=load_files)
        gw.reproject_shapefiles(already_reprojected=load_files)
        gw.create_gw_rasters(already_created=load_files)
        gw.reclassify_cdl(ks_class_dict, already_reclassified=load_files)
        gw.create_crop_coeff_raster(already_created=load_files)
        gw.reproject_rasters(already_reprojected=load_files)
        gw.create_land_use_rasters(already_created=load_files)
        gw.update_crop_coeff_raster(already_updated=load_files)
        gw.mask_rasters(already_masked=load_files)
        remove_na = True
        if not build_ml_model:
            remove_na = False
        df = gw.create_dataframe(year_list=range(2002, 2020), exclude_vars=exclude_vars, load_df=load_df,
                                 remove_na=remove_na)
        if build_ml_model:
            max_features = len(df.columns.values.tolist()) - len(drop_attrs) - 1
            rf_model = gw.build_model(df, n_estimators=500, test_year=range(2011, 2019), drop_attrs=drop_attrs,
                                      pred_attr=pred_attr, load_model=load_rf_model, max_features=max_features,
                                      plot_graphs=False)
            actual_gw_dir, pred_gw_dir = gw.get_predictions(rf_model=rf_model, pred_years=range(2002, 2020),
                                                            drop_attrs=drop_attrs[:1] + exclude_vars,
                                                            pred_attr=pred_attr, only_pred=False)
    if build_ml_model:
        input_gmd_file = file_dir + 'gmds/reproj/input_gmd_reproj.shp'
        ma.run_analysis(gw_dir, pred_gw_dir, grace_csv, use_gmds=use_gmds, out_dir=output_dir,
                        input_gmd_file=input_gmd_file)
        if show_box_plots:
            ma.generate_feature_box_plots(output_dir + '/raster_df.csv')
        if show_train_test_box_plots:
            ma.generate_feature_box_plots(output_dir + 'X_Train.csv')
            ma.generate_feature_box_plots(output_dir + 'X_Test.csv')
    return gw, df


def run_gw_az(analyze_only=False, load_files=True, load_rf_model=False, load_df=False, build_ml_model=True,
              subsidence_analysis=False):
    """
    Main function for running the project for Arizona, some variables require to be hardcoded
    :param analyze_only: Set True to just produce analysis results, all required files must be present
    :param load_files: Set True to load existing files, needed only if analyze_only=False
    :param load_rf_model: Set True to load existing Random Forest model, needed only if analyze_only=False
    :param load_df: Set True to load existing dataframe from CSV
    :param build_ml_model: Set False to disable model building. If True, watershed stress index rasters will be computed
    as it is a useful predictor in Arizona
    :param subsidence_analysis: Set True to analyze total subsidence and total groundwater withdrawals in a
    specified period, build_ml_model must be True
    :return: Returns HydroML object and Pandas dataframe if build_model=False, None otherwise
    """

    gee_data = ['Apr_Sept/', 'Apr_Aug/', 'Annual/']
    input_dir = '../Inputs/Data/Arizona_GW/'
    input_subsidence_dir = input_dir + 'Subsidence/Subsidence_Rasters/'
    file_dir = '../Inputs/Files_AZ_' + gee_data[0]
    output_dir = '../Outputs/Output_AZ_' + gee_data[0]
    output_shp_dir = file_dir + 'GW_Shapefiles/'
    output_gw_raster_dir = file_dir + 'GW_Rasters/'
    input_well_reg_file = input_dir + 'Well_Registry/WellRegistry.shp'
    input_ama_ina_file = input_dir + 'Boundary/AMA_and_INA.shp'
    input_watershed_file = input_dir + 'Watersheds/Surface_Watershed.shp'
    input_gw_csv_dir = input_dir + 'GW_Data/'
    input_state_file = input_dir + 'Arizona/Arizona.shp'
    gdal_path = 'C:/OSGeo4W64/'
    actual_gw_dir = file_dir + 'RF_Data/'
    pred_gw_dir = output_dir + 'Predicted_Rasters/'
    subsidence_gw_dir = output_dir + 'Subsidence_Analysis/'
    grace_csv = input_dir + 'GRACE/TWS_GRACE.csv'
    ssebop_link = 'https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/uswem/web/conus/eta/modis_eta/monthly/' \
                  'downloads/'
    data_year_list = range(2002, 2020)
    data_start_month = 4
    data_end_month = 9
    ws_start_month = 10
    ws_end_month = 5
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
    exclude_vars = ('ET', 'WS_PA', 'WS_PA_EA', 'WS_PT', 'WS_PT_ET')
    if build_ml_model:
        exclude_vars = ('ET', 'WS_PT', 'WS_PT_ET')
    pred_attr = 'GW'
    fill_attr = 'AF Pumped'
    filter_attr = None
    df = pd.DataFrame()
    gw = None
    if not analyze_only:
        gw = HydroML(input_dir, file_dir, output_dir, output_shp_dir, output_gw_raster_dir,
                     input_state_file, gdal_path, input_subsidence_dir=input_subsidence_dir,
                     input_gw_boundary_file=input_well_reg_file, input_ama_ina_file=input_ama_ina_file,
                     input_watershed_file=input_watershed_file, ssebop_link=ssebop_link)
        gw.download_data(year_list=data_year_list, start_month=data_start_month, end_month=data_end_month,
                         already_downloaded=load_files, already_extracted=load_files)
        remove_na = False
        if build_ml_model:
            remove_na = True
            gw.download_ws_data(year_list=data_year_list, start_month=ws_start_month, end_month=ws_end_month,
                                already_downloaded=load_files, already_extracted=load_files)
        gw.preprocess_gw_csv(input_gw_csv_dir, fill_attr=fill_attr, filter_attr=filter_attr,
                             already_preprocessed=load_files)
        gw.reproject_shapefiles(already_reprojected=load_files)
        gw.create_gw_rasters(already_created=load_files, value_field=fill_attr, xres=5000, yres=5000, max_gw=2e+4)
        gw.crop_gw_rasters(use_ama_ina=True, already_cropped=load_files)
        gw.reclassify_cdl(az_class_dict, already_reclassified=load_files)
        gw.create_crop_coeff_raster(already_created=load_files)
        gw.reproject_rasters(already_reprojected=load_files)
        gw.create_land_use_rasters(already_created=load_files, smoothing_factors=(3, 5, 3))
        gw.update_crop_coeff_raster(already_updated=load_files)
        if build_ml_model:
            gw.create_water_stress_index_rasters(already_created=load_files, normalize=False)
            if subsidence_analysis:
                gw.organize_subsidence_rasters(already_organized=load_files)
        gw.mask_rasters(already_masked=load_files)
        df = gw.create_dataframe(year_list=range(2002, 2020), exclude_vars=exclude_vars, exclude_years=(2019,),
                                 load_df=load_df, remove_na=remove_na)
        if build_ml_model:
            max_features = len(df.columns.values.tolist()) - len(drop_attrs) - 1
            rf_model = gw.build_model(df, test_year=range(2011, 2019), drop_attrs=drop_attrs, pred_attr=pred_attr,
                                      load_model=load_rf_model, max_features=max_features, plot_graphs=False)
            use_full_extent = False
            if subsidence_analysis:
                use_full_extent = True
            actual_gw_dir, pred_gw_dir = gw.get_predictions(rf_model=rf_model, pred_years=range(2002, 2020),
                                                            drop_attrs=drop_attrs, pred_attr=pred_attr,
                                                            exclude_vars=exclude_vars, exclude_years=(),
                                                            only_pred=False, use_full_extent=use_full_extent)
            if subsidence_analysis:
                gw.create_subsidence_pred_gw_rasters(already_created=load_files)

    if build_ml_model:
        ma.run_analysis(actual_gw_dir, pred_gw_dir, grace_csv, use_gmds=False, input_gmd_file=None, out_dir=output_dir,
                        forecast_years=(2019,))
        ma.subsidence_analysis(subsidence_gw_dir)
    return gw, df


def run_gw(build_individual_model=False, run_only_az=True):
    """
    Main caller function for Kansas and Arizona. This will build the ML model using both Kansas and Arizona data.
    :param build_individual_model: Set True to build individual models for Kansas and Arizona and analyze results.
    :param run_only_az: Set False to run both Kansas and Arizona models. If True, build_individual_model must be True
    :return: None
    """

    analyze_only = False
    load_files = True
    load_rf_model = False
    load_df = False
    subsidence_analysis = False
    gw_ks, ks_df = None, None
    if not run_only_az:
        gw_ks, ks_df = run_gw_ks(analyze_only=analyze_only, load_files=load_files, load_rf_model=load_rf_model,
                                 use_gmds=False, build_ml_model=build_individual_model, load_df=load_df)
    gw_az, az_df = run_gw_az(analyze_only=analyze_only, load_files=load_files, load_rf_model=load_rf_model,
                             build_ml_model=build_individual_model, subsidence_analysis=subsidence_analysis,
                             load_df=load_df)
    if not build_individual_model:
        final_gw_df = ks_df.append(az_df)
        output_dir = '../Outputs/All_Data/'
        final_gw_df.to_csv(output_dir + 'GW_DF_KS_AZ.csv', index=False)
        final_gw_df = final_gw_df.dropna(axis=0)
        drop_attrs = ('YEAR',)
        exclude_vars = ('ET',)
        pred_attr = 'GW'
        max_features = len(final_gw_df.columns.values.tolist()) - len(drop_attrs) - 1
        gw = HydroML(None, None, output_dir, None, None, None, None)
        rf_model = gw.build_model(final_gw_df, test_year=range(2011, 2019), drop_attrs=drop_attrs, pred_attr=pred_attr,
                                  load_model=False, max_features=max_features, plot_graphs=False)
        actual_gw_dir, pred_gw_dir = gw_ks.get_predictions(rf_model=rf_model, pred_years=range(2002, 2020),
                                                           drop_attrs=drop_attrs, exclude_vars=exclude_vars,
                                                           pred_attr=pred_attr, only_pred=False)
        grace_csv = '../Inputs/Data/Kansas_GW/GRACE/TWS_GRACE.csv'
        ma.run_analysis(actual_gw_dir, pred_gw_dir, grace_csv, use_gmds=False, input_gmd_file=None, out_dir=output_dir,
                        forecast_years=(2019,))
        exclude_vars = ('ET', 'WS_PA', 'WS_PA_EA', 'WS_PT', 'WS_PT_ET')
        actual_gw_dir, pred_gw_dir = gw_az.get_predictions(rf_model=rf_model, pred_years=range(2002, 2020),
                                                           drop_attrs=drop_attrs, pred_attr=pred_attr,
                                                           exclude_vars=exclude_vars, exclude_years=(),
                                                           only_pred=False, use_full_extent=False)
        grace_csv = '../Inputs/Data/Arizona_GW/GRACE/TWS_GRACE.csv'
        ma.run_analysis(actual_gw_dir, pred_gw_dir, grace_csv, use_gmds=False, input_gmd_file=None, out_dir=output_dir,
                        forecast_years=(2019,))


run_gw(build_individual_model=True, run_only_az=True)
