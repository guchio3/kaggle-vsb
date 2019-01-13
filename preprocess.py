import gc
import os
import sys
from logging import getLogger
from multiprocessing import Pool

import pandas as pd

import numpy as np
import pyarrow.parquet as pq
from tools.features.preprocessing import (add_high_pass_filter,
                                          decode_signals_after_pool,
                                          denoise_signal)
from tools.utils.general_utils import logInit, dec_timer


@dec_timer
def preprocess(logger):
    '''
    policy
    ------------
    * Directly edit the src for each preprocessing because preprocess
        should be ran only once.

    '''
#    trn_series_path = './inputs/origin/train.parquet'
#    trn_meta_path = './inputs/origin/metadata_train.csv'
#    tst_series_path = './inputs/origin/test.parquet'
#    tst_meta_path = './inputs/origin/metadata_test.csv'

    with Pool(32) as p:
        trn_series_path = './inputs/origin/train.parquet'
        logger.info('Loading base data ...')
        trn_series_df = pq.read_pandas(trn_series_path).to_pandas()

        trn_hp_filename = './inputs/prep/train_hp.pkl.gz'
        logger.info(f'Processing for {trn_hp_filename} ...')
        trn_sigid_series_pairs = [[i, trn_series_df.loc[:, str(i)]]
                                  for i in range(trn_series_df.shape[1])]
        del trn_series_df
        gc.collect()
        trn_pooled_hp_signals = p.map(
            add_high_pass_filter, trn_sigid_series_pairs)
        p.close()
        p.join()
        trn_decoded_signals_df = decode_signals_after_pool(
            trn_pooled_hp_signals)
        logger.info(f'Saving to {trn_hp_filename} ...')
        trn_decoded_signals_df.to_pickle(
            trn_hp_filename, compression='gzip')
        del trn_decoded_signals_df
        gc.collect()


if __name__ == '__main__':
    logger = getLogger(__name__)
    logger = logInit(logger, './logs/', 'preprocess.log')
    preprocess(logger)
