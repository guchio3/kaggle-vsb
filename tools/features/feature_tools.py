import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append('../utils/')
from general_utils import sel_log, dec_timer


@dec_timer
def split_df(base_df, target_df, split_name,
             target_name, nthread, logger=None):
    '''
    policy
    ------------
    * split df based on split_id, and set split_id as index
        because of efficiency.

    '''
    sel_log(
        f'now splitting a df to {nthread} dfs using {split_name} ...',
        logger)
    split_ids = base_df[split_name].unique()
    splitted_ids = np.array_split(split_ids, nthread)
    target_ids = [base_df.set_index(split_name)
                  .loc[splitted_id][target_name]
                  for splitted_id in splitted_ids]
    # Pay attention that this is col-wise splitting bacause of the
    #   data structure of this competition.
    dfs = [target_df[target_id.astype(str)] for target_id in target_ids]
    return dfs


@dec_timer
def load_features(features, base_dir, logger=None):
    loaded_features = []
    for feature in tqdm(features):
        load_filename = base_dir + feature + '.pkl.gz'
        sel_log(f'now loading {feature}', None)
        loaded_feature = pd.read_pickle(load_filename, compression='gzip')
        loaded_features.append(loaded_feature)

    features_df = pd.concat(loaded_features, axis=1)
    return features_df


@dec_timer
def save_features(features_df, base_dir, logger=None):
    for feature in tqdm(features_df.columns):
        save_filename = base_dir + feature + '.pkl.gz'
        if os.path.exists(save_filename):
            sel_log(f'already exists at {save_filename} !', None)
        else:
            sel_log(f'saving to {save_filename} ...', logger)
            features_df[feature].to_pickle(save_filename, compression='gzip')
