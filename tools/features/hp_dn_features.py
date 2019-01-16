import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy

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


def e006_hp_dn_abs_percentiles(df):
    df = df.abs().astype('float64')
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


def e007_hp_dn_abs_thresh_overs(df):
    df = df.abs().astype('float64')
    features = pd.DataFrame()
    # threshold 1
    th1_df = df[df > 1]
    features['th1_count'] = th1_df.count()
    features['th1_std'] = [
        np.std(th1_df[col].dropna()) for col in th1_df.columns]
    features['th1_time_std'] = [
        np.std(th1_df[col].dropna().index) for col in th1_df.columns]
    features['th1_skew'] = [scipy.stats.skew(
        th1_df[col].dropna()) for col in th1_df.columns]
    features['th1_time_skew'] = [scipy.stats.skew(
        th1_df[col].dropna().index) for col in th1_df.columns]
    features['th1_kurt'] = [scipy.stats.kurtosis(
        th1_df[col].dropna()) for col in th1_df.columns]
    features['th1_time_kurt'] = [scipy.stats.kurtosis(
        th1_df[col].dropna().index) for col in th1_df.columns]
    # threshold 3
    th3_df = df[df > 3]
    features['th3_count'] = th3_df.count()
    features['th3_std'] = [
        np.std(th3_df[col].dropna()) for col in th3_df.columns]
    features['th3_time_std'] = [
        np.std(th3_df[col].dropna().index) for col in th3_df.columns]
    features['th3_skew'] = [scipy.stats.skew(
        th3_df[col].dropna()) for col in th3_df.columns]
    features['th3_time_skew'] = [scipy.stats.skew(
        th3_df[col].dropna().index) for col in th3_df.columns]
    features['th3_kurt'] = [scipy.stats.kurtosis(
        th3_df[col].dropna()) for col in th3_df.columns]
    features['th3_time_kurt'] = [scipy.stats.kurtosis(
        th3_df[col].dropna().index) for col in th3_df.columns]
    # threshold 5
    th5_df = df[df > 5]
    features['th5_count'] = th5_df.count()
    features['th5_std'] = [
        np.std(th5_df[col].dropna()) for col in th5_df.columns]
    features['th5_time_std'] = [
        np.std(th5_df[col].dropna().index) for col in th5_df.columns]
    features['th5_skew'] = [scipy.stats.skew(
        th5_df[col].dropna()) for col in th5_df.columns]
    features['th5_time_skew'] = [scipy.stats.skew(
        th5_df[col].dropna().index) for col in th5_df.columns]
    features['th5_kurt'] = [scipy.stats.kurtosis(
        th5_df[col].dropna()) for col in th5_df.columns]
    features['th5_time_kurt'] = [scipy.stats.kurtosis(
        th5_df[col].dropna().index) for col in th5_df.columns]
    # threshold 10
    th10_df = df[df > 10]
    features['th10_count'] = th5_df.count()
    features['th10_std'] = [
        np.std(th10_df[col].dropna()) for col in th10_df.columns]
    features['th10_time_std'] = [
        np.std(th10_df[col].dropna().index) for col in th10_df.columns]
    features['th10_skew'] = [scipy.stats.skew(
        th10_df[col].dropna()) for col in th10_df.columns]
    features['th10_time_skew'] = [scipy.stats.skew(
        th10_df[col].dropna().index) for col in th10_df.columns]
    features['th10_kurt'] = [scipy.stats.kurtosis(
        th10_df[col].dropna()) for col in th10_df.columns]
    features['th10_time_kurt'] = [scipy.stats.kurtosis(
        th10_df[col].dropna().index) for col in th10_df.columns]
    features = features.add_prefix(
        'e007_hp_dn_abs_thresh_overs_').reset_index(drop=True)
    return features


def e008_hp_dn_scaled(df):
    df = df.astype('float64')
    features = pd.DataFrame()

    # Scaled in each phase
    phase_scaled_df = df / df.abs().max(axis=0)
    features['phase_scaled_mean'] = phase_scaled_df.mean(axis=0)
    features['phase_scaled_std'] = phase_scaled_df.std(axis=0)
    features['phase_scaled_skew'] = [scipy.stats.skew(
        phase_scaled_df[col]) for col in phase_scaled_df.columns]
    features['phase_scaled_kurt'] = [scipy.stats.kurtosis(
        phase_scaled_df[col]) for col in phase_scaled_df.columns]

    # Scaled in the 3 phases
    group_scaled_df = df / df.abs().max(axis=0).max()
    features['group_scaled_mean'] = group_scaled_df.mean(axis=0)
    features['group_scaled_std'] = group_scaled_df.std(axis=0)
    features['group_scaled_skew'] = [scipy.stats.skew(
        phase_scaled_df[col]) for col in group_scaled_df.columns]
    features['group_scaled_kurt'] = [scipy.stats.kurtosis(
        phase_scaled_df[col]) for col in group_scaled_df.columns]

    features = features.add_prefix(
        'e008_hp_dn_scaled_').reset_index(drop=True)
    return features


def e009_hp_dn_3phase_time_diffs(df):
    df = df.abs().astype('float64')
    features = pd.DataFrame()
    # threshold 1
    th1_df = df[df > 1]
    features['th1_time_diff_std'] = [np.std(np.diff(
        th1_df[col].dropna().index)) for col in th1_df.columns]
    features['th1_time_diff_skew'] = [scipy.stats.skew(np.diff(
        th1_df[col].dropna().index)) for col in th1_df.columns]
    features['th1_time_diff_kurt'] = [scipy.stats.kurtosis(np.diff(
        th1_df[col].dropna().index)) for col in th1_df.columns]
    # threshold 3
    th3_df = df[df > 3]
    features['th3_time_diff_std'] = [np.std(np.diff(
        th3_df[col].dropna().index)) for col in th3_df.columns]
    features['th3_time_diff_skew'] = [scipy.stats.skew(np.diff(
        th3_df[col].dropna().index)) for col in th3_df.columns]
    features['th3_time_diff_kurt'] = [scipy.stats.kurtosis(np.diff(
        th3_df[col].dropna().index)) for col in th3_df.columns]
    # threshold 5
    th5_df = df[df > 5]
    features['th5_time_diff_std'] = [np.std(np.diff(
        th5_df[col].dropna().index)) for col in th5_df.columns]
    features['th5_time_diff_skew'] = [scipy.stats.skew(np.diff(
        th5_df[col].dropna().index)) for col in th5_df.columns]
    features['th5_time_diff_kurt'] = [scipy.stats.kurtosis(np.diff(
        th5_df[col].dropna().index)) for col in th5_df.columns]
    # threshold 10
    th10_df = df[df > 10]
    features['th10_time_diff_std'] = [np.std(np.diff(
        th10_df[col].dropna().index)) for col in th10_df.columns]
    features['th10_time_diff_skew'] = [scipy.stats.skew(np.diff(
        th10_df[col].dropna().index)) for col in th10_df.columns]
    features['th10_time_diff_kurt'] = [scipy.stats.kurtosis(np.diff(
        th10_df[col].dropna().index)) for col in th10_df.columns]
    features = features.add_prefix(
        'e009_hp_dn_3phase_time_diffs_').reset_index(drop=True)
    return features


def _hp_dn_features(df, exp_ids):
    _features = []
    _features.append(pd.DataFrame(df.columns,
                                  columns=['signal_id'],
                                  dtype=int))
    if 'e002' in exp_ids:
        _features.append(e002_hp_dn_basic(df))
    if 'e006' in exp_ids:
        _features.append(e006_hp_dn_abs_percentiles(df))
    if 'e007' in exp_ids:
        _features.append(e007_hp_dn_abs_thresh_overs(df))
    if 'e008' in exp_ids:
        _features.append(e008_hp_dn_scaled(df))
    if 'e009' in exp_ids:
        _features.append(e009_hp_dn_3phase_time_diffs(df))
    features = pd.concat(_features, axis=1)
    return features


@dec_timer
def _load_hp_dn_features_src(exp_ids, test, series_df, meta_df, logger):
    target_ids = [
        'e002',
        'e006',
        'e007',
        'e008',
        'e009',
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
