import sys
from functools import partial
from multiprocessing import Pool

import pandas as pd
import pyarrow.parquet as pq

from feature_tools import save_features, split_df

sys.path.append('../utils/')
from general_utils import dec_timer, sel_log


def e001_basic(df, exp_ids):
    features = pd.DataFrame()
    features['max'] = df.max(axis=0)
    features['min'] = df.min(axis=0)
    features['std'] = df.std(axis=0)
    features = features.add_prefix('e001_base_').reset_index(drop=True)
    return features


def _base_features(df, exp_ids):
    _features = []
    _features.append(pd.DataFrame(df.columns,
                                  columns=['signal_id'],
                                  dtype=int))
    if 'e001' in exp_ids:
        _features.append(e001_basic(df, exp_ids))
    features = pd.concat(_features, axis=1)
    return features


@dec_timer
def mk_base_features(nthread, exp_ids, test=False, series_df=None,
                     meta_df=None, logger=None):
    if test:
        series_path = './inputs/origin/test.parquet'
        meta_path = './inputs/origin/metadata_test.csv'
        base_dir = './inputs/test/'
    else:
        series_path = './inputs/origin/train.parquet'
        meta_path = './inputs/origin/metadata_train.csv'
        base_dir = './inputs/train/'

    # Load dfs if not input.
    if not series_df:
        sel_log(f'loading {series_path} ...', None)
        series_df = pq.read_pandas(series_path).to_pandas()
    if not meta_df:
        sel_log(f'loading {meta_path} ...', None)
        meta_df = pd.read_csv(meta_path)

    # Test is only 20338, so i use splitting only for series.
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
        iter_func = partial(_base_features, exp_ids=exp_ids)
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
    sel_log(f'saving features ...', logger)
    save_features(features_df, base_dir, logger)

    return series_df, meta_df
