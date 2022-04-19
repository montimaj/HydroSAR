# Author: Sayantan Majumdar
# Email: smxnv@mst.edu

import pandas as pd
from glob import glob


def create_merged_results(input_dir, remove_cols=('Test', 'F_IMP')):
    """
    Create a merged results csv from multiple CSVs
    :param input_dir: Input directory contaning CSVs
    :param remove_cols: Remove columns in this list from the final data frame
    :return: Merged Pandas data frame
    """

    csv_files = sorted(glob(input_dir + 'RF*.csv'))
    df = pd.read_csv(csv_files[0])
    for idx, csv in enumerate(csv_files):
        if idx == 0:
            df['Scale'] = [1] * df.shape[0]
        else:
            new_df = pd.read_csv(csv)
            new_df['Scale'] = [idx + 1] * new_df.shape[0]
            df = df.append(new_df)
    df['Window'] = list(range(1, 11)) * 5
    df = df.drop(columns=list(remove_cols))
    df.to_csv(input_dir + 'merged.csv', index=False)
    return df


create_merged_results('D:/HydroMST/Paper2/Results_New/Scale/Temporal/')
create_merged_results('D:/HydroMST/Paper2/Results_New/Scale/Spatial/')
create_merged_results('D:/HydroMST/Paper2/Results_New/Scale/ST/')
