import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ..utils.general_utils import dec_timer, sel_log


def e002_hp_dn_basic(df):
    df = df.astype('float64')
    features = pd.DataFrame()
    features['max'] = df.max(axis=0)
    features['min'] = df.min(axis=0)
    features['mean'] = df.mean(axis=0)
    features['std'] = df.std(axis=0)
    features = features.add_prefix('e002_hp_dn_basic_').reset_index(drop=True)
    return features


def e006_hp_dn_percentiles(df):
    df = df.astype('float64')
    features = pd.DataFrame()
    features['perc1'] = np.percentile(df, 1, axis=0)
    features['perc5'] = np.percentile(df, 5, axis=0)
    features['perc10'] = np.percentile(df, 10, axis=0)
    features['perc90'] = np.percentile(df, 90, axis=0)
    features['perc95'] = np.percentile(df, 95, axis=0)
    features['perc99'] = np.percentile(df, 99, axis=0)
    features['perc99m1'] = features['perc99'] - features['perc1']
    features['perc95m5'] = features['perc95'] - features['perc5']
    features['perc90m10'] = features['perc90'] - features['perc10']
    features['perc1p99'] = features['perc1'] / features['perc99']
    features['perc5p95'] = features['perc5'] / features['perc95']
    features['perc10p90'] = features['perc10'] / features['perc90']
    features = features.add_prefix(
        'e006_hp_dn_percentiles_').reset_index(drop=True)
    return features


def _hp_dn_features(df, exp_ids):
    _features = []
    _features.append(pd.DataFrame(df.columns,
                                  columns=['signal_id'],
                                  dtype=int))
    if 'e002' in exp_ids:
        _features.append(e002_hp_dn_basic(df))
    features = pd.concat(_features, axis=1)
    return features


@dec_timer
def _load_hp_dn_features_src(exp_ids, test, series_df, meta_df, logger):
    target_ids = [
        'e002',
        'e006',
    ]
    if len(set(target_ids) & set(exp_ids)) < 1:
        sel_log(f'''
                ======== {__name__} ========
                Stop feature making because even 1 element in exp_ids
                    {exp_ids}
                does not in target_ids
                    {target_ids}''', logger)
        return None, None

    if test:
        series_path = './inputs/prep/test_hp_dn.pkl.gz'
        meta_path = './inputs/origin/metadata_test.csv'
    else:
        series_path = './inputs/prep/train_hp_dn.pkl.gz'
        meta_path = './inputs/origin/metadata_train.csv'

    # Load dfs if not input.
    if series_df is None:
        sel_log(f'loading {series_path} ...', None)
        series_df = pd.read_pickle(series_path, compression='gzip')
    if meta_df is None:
        sel_log(f'loading {meta_path} ...', None)
        meta_df = pd.read_csv(meta_path)

    return series_df, meta_df
