# Author: Sayantan Majumdar
# Email: smxnv@mst.edu

import pandas as pd
from Python_Files.hydrolibs.sysops import makedirs
from glob import glob


def create_merged_results(input_dir, remove_cols=('Test', 'F_IMP')):
    """
    Create a merged results csv from multiple CSVs
    :param input_dir: Input directory contaning CSV directories
    :param remove_cols: Remove columns in this list from the final data frame
    :return: None
    """

    pattern_list = ['T', 'S', 'ST']
    merged_dir = input_dir + 'Results_SA/'
    makedirs([merged_dir])
    for pattern in pattern_list:
        csv_dirs = sorted(glob(input_dir + '*_{}'.format(pattern)))
        merged_df = pd.DataFrame()
        for idx, csv_dir in enumerate(csv_dirs):
            print('Merging', csv_dir)
            df = pd.read_csv(csv_dir + '/RF_Results.csv')
            test_na = ~df.Test.isna()
            if test_na.all():
                df = df.shift(1, axis=1)
            df = df.drop(columns=list(remove_cols))
            df['Scale'] = idx + 1
            df['Window'] = list(range(1, 11))
            merged_df = pd.concat([merged_df, df])
        merged_df.to_csv('{}Merged_{}.csv'.format(merged_dir, pattern), index=False)


if __name__ == '__main__':
    create_merged_results('../../Outputs/')
