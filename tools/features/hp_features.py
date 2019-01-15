import pandas as pd
import pyarrow.parquet as pq

from ..utils.general_utils import dec_timer, sel_log


def e002_hp_basic(df):
    df = df.astype('float64')
    features = pd.DataFrame()
    features['max'] = df.max(axis=0)
    features['min'] = df.min(axis=0)
    features['mean'] = df.mean(axis=0)
    features['std'] = df.std(axis=0)
    features = features.add_prefix('e002_hp_basic_').reset_index(drop=True)
    return features


def _hp_features(df, exp_ids):
    _features = []
    _features.append(pd.DataFrame(df.columns,
                                  columns=['signal_id'],
                                  dtype=int))
    if 'e002' in exp_ids:
        _features.append(e002_hp_basic(df))
    features = pd.concat(_features, axis=1)
    return features


@dec_timer
def _load_hp_features_src(exp_ids, test, series_df, meta_df, logger):
    target_ids = [
        'e002',
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
        series_path = './inputs/prep/test_hp.pkl.gz'
        meta_path = './inputs/origin/metadata_test.csv'
    else:
        series_path = './inputs/prep/train_hp.pkl.gz'
        meta_path = './inputs/origin/metadata_train.csv'

    # Load dfs if not input.
    if series_df is None:
        sel_log(f'loading {series_path} ...', None)
        series_df = pd.read_pickle(series_path, compression='gzip')
    if meta_df is None:
        sel_log(f'loading {meta_path} ...', None)
        meta_df = pd.read_csv(meta_path)

    return series_df, meta_df
