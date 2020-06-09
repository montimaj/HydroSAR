# Author: Sayantan Majumdar
# Email: smxnv@mst.edu

import wget
import zipfile
import numpy as np
from glob import glob
from Python_Files.hydrolibs import rasterops as rops


def download_ssebop_data(sse_link, year_list, month_list, outdir):
    """
    Download SSEBop Data
    :param sse_link: Main SSEBop link without file name
    :param year_list: List of years
    :param month_list: List of months
    :param outdir: Download directory
    :return: None
    """

    for year in year_list:
        print('Downloading SSEBop for', year, '...')
        for month in month_list:
            month_str = str(month)
            if 1 <= month <= 9:
                month_str = '0' + month_str
            url = sse_link + 'm' + str(year) + month_str + '.zip'
            local_file_name = outdir + 'SSEBop_' + str(year) + month_str + '.zip'
            wget.download(url, local_file_name)


def extract_data(zip_dir, out_dir):
    """
    Extract SSEBop Data
    :param zip_dir: SSEBop Zip directory
    :param out_dir: Output directory to write SSEBop images
    :return: None
    """

    print('Extracting SSEBop files...')
    for zip_file in glob(zip_dir + '*.zip'):
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(out_dir)


def generate_cummulative_ssebop(ssebop_dir, year_list, month_list, out_dir):
    """
    Generate cummulative SSEBop data
    :param ssebop_dir: SSEBop directory
    :param year_list: List of years
    :param month_list: List of months
    :param out_dir: Output directory
    :return: None
    """

    for year in year_list:
        print('Generating cummulative SSEBop for', year, '...')
        ssebop_raster_arr_list = []
        ssebop_raster_file_list = []
        for month in month_list:
            month_str = str(month)
            if 1 <= month <= 9:
                month_str = '0' + month_str
            pattern = '*' + str(year) + month_str + '*.tif'
            ssebop_file = glob(ssebop_dir + pattern)[0]
            ssebop_raster_arr, ssebop_raster_file = rops.read_raster_as_arr(ssebop_file)
            ssebop_raster_arr_list.append(ssebop_raster_arr)
            ssebop_raster_file_list.append(ssebop_raster_file)
        sum_arr_ssebop = ssebop_raster_arr_list[0]
        for ssebop_raster_arr in ssebop_raster_arr_list[1:]:
            sum_arr_ssebop += ssebop_raster_arr
        sum_arr_ssebop[np.isnan(sum_arr_ssebop)] = rops.NO_DATA_VALUE
        ssebop_raster_file = ssebop_raster_file_list[0]
        out_ssebop = out_dir + 'SSEBop_' + str(year) + '.tif'
        rops.write_raster(sum_arr_ssebop, ssebop_raster_file, transform=ssebop_raster_file.transform,
                          outfile_path=out_ssebop)
