import os
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..utils.general_utils import dec_timer, sel_log


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


@dec_timer
def _mk_features(load_func, feature_func, nthread, exp_ids, test=False,
                 series_df=None, meta_df=None, logger=None):
    # Load dfs
    # Does not load if the exp_ids are not the targets.
    series_df, meta_df = load_func(exp_ids, test, series_df, meta_df, logger)
    # Finish before feature engineering if the exp_ids are not the targets.
    if series_df is None:
        return None, None

    # Test meta is only 20338, so i use splitting only for series.
    series_dfs = split_df(
        meta_df,
        series_df,
        'id_measurement',
        'signal_id',
        nthread,
        logger=logger)

    with Pool(nthread) as p:
        sel_log(f'feature engineering ...', None)
        # Using partial enable to use constant argument for the iteration.
        iter_func = partial(feature_func, exp_ids=exp_ids)
        features_list = p.map(iter_func, series_dfs)
        p.close()
        p.join()
        features_df = pd.concat(features_list, axis=0)

    # Merge w/ meta.
    # This time, i don't remove the original features because
    #   this is the base feature function.
    sel_log(f'merging features ...', None)
    features_df = meta_df.merge(features_df, on='signal_id', how='left')

    # Save the features
    if test:
        base_dir = './inputs/test/'
    else:
        base_dir = './inputs/train/'
    sel_log(f'saving features ...', logger)
    save_features(features_df, base_dir, logger)

    return series_df, meta_df
