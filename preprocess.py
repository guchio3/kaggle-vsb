import gc
import os
import sys
from logging import getLogger
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

from tools.features.preprocessing import (add_high_pass_filter,
                                          decode_signals_after_pool,
                                          denoise_signal)
from tools.utils.general_utils import (dec_timer, logInit,
                                       send_line_notification)


@dec_timer
def preprocess(logger):
    '''
    policy
    ------------
    * Directly edit the src for each preprocessing because preprocess
        should be ran only once.

    '''
    # HPF for train
    if False:
        p = Pool(62)
        trn_series_path = './inputs/origin/train.parquet'
        logger.info('Loading base data ...')
        trn_series_df = pq.read_pandas(trn_series_path).to_pandas()

        trn_hp_filename = './inputs/prep/train_hp.pkl'
        logger.info(f'Processing for {trn_hp_filename} ...')
        trn_sigid_series_pairs = [[i, trn_series_df.loc[:, i]]
                                  for i in trn_series_df.columns]
        del trn_series_df
        gc.collect()
        trn_pooled_hp_signals = p.map(
            add_high_pass_filter, trn_sigid_series_pairs)
        p.close()
        p.join()
        del trn_sigid_series_pairs
        gc.collect()

        logger.info(f'Decoding after pool ...')
        trn_decoded_signals_df = decode_signals_after_pool(
            trn_pooled_hp_signals)
        logger.info(f'Saving to {trn_hp_filename} ...')
        trn_decoded_signals_df.to_pickle(
            trn_hp_filename)
        send_line_notification(f'Finish processing {trn_hp_filename} !')

    # HPF for test
    if False:
        p = Pool(62)
        tst_series_path = './inputs/origin/test.parquet'
        logger.info('Loading base data ...')
        tst_series_df = pq.read_pandas(tst_series_path).to_pandas()

        tst_hp_filename = './inputs/prep/test_hp.pkl'
        logger.info(f'Processing for {tst_hp_filename} ...')
        tst_sigid_series_pairs = [[i, tst_series_df.loc[:, i]]
                                  for i in tst_series_df.columns]
        del tst_series_df
        gc.collect()
        tst_pooled_hp_signals = p.map(
            add_high_pass_filter, tst_sigid_series_pairs)
        p.close()
        p.join()
        del tst_sigid_series_pairs
        gc.collect()

        logger.info(f'Decoding after pool ...')
        tst_decoded_signals_df = decode_signals_after_pool(
            tst_pooled_hp_signals)
        logger.info(f'Saving to {tst_hp_filename} ...')
        tst_decoded_signals_df.to_pickle(tst_hp_filename)
        send_line_notification(f'Finish processing {tst_hp_filename} !')

    # Denoising for HPF train
    if False:
        p = Pool(62)
        trn_hp_series_path = './inputs/prep/train_hp.pkl.gz'
        logger.info(f'Loading base data from {trn_hp_series_path} ...')
        trn_hp_series_df = pd.read_pickle(
            trn_hp_series_path, compression='gzip')

        trn_hp_dn_filename = './inputs/prep/train_hp_dn.pkl'
        logger.info(f'Processing for {trn_hp_dn_filename} ...')
        trn_hp_dn_sigid_series_pairs = [[i, trn_hp_series_df.loc[:, i]]
                                        for i in trn_hp_series_df.columns]
        del trn_hp_series_df
        gc.collect()
        trn_pooled_hp_dn_signals = p.map(
            denoise_signal, trn_hp_dn_sigid_series_pairs)
        p.close()
        p.join()
        del trn_hp_dn_sigid_series_pairs
        gc.collect()

        logger.info(f'Decoding after pool ...')
        trn_decoded_hp_dn_signals_df = decode_signals_after_pool(
            trn_pooled_hp_dn_signals)
        logger.info(f'Saving to {trn_hp_dn_filename} ...')
        trn_decoded_hp_dn_signals_df.to_pickle(trn_hp_dn_filename)
        send_line_notification(f'Finish processing {trn_hp_dn_filename} !')

    # Denoising for HPF test
    if True:
        p = Pool(62)
        tst_hp_series_path = './inputs/prep/test_hp.pkl.gz'
        logger.info(f'Loading base data from {tst_hp_series_path} ...')
        tst_hp_series_df = pd.read_pickle(
            tst_hp_series_path, compression='gzip')

        tst_hp_dn_filename = './inputs/prep/test_hp_dn.pkl'
        logger.info(f'Processing for {tst_hp_dn_filename} ...')
        tst_hp_dn_sigid_series_pairs = [[i, tst_hp_series_df.loc[:, i]]
                                        for i in tst_hp_series_df.columns]
        del tst_hp_series_df
        gc.collect()
        tst_pooled_hp_dn_signals = p.map(
            denoise_signal, tst_hp_dn_sigid_series_pairs)
        p.close()
        p.join()
        del tst_hp_dn_sigid_series_pairs
        gc.collect()

        logger.info(f'Decoding after pool ...')
        tst_decoded_hp_dn_signals_df = decode_signals_after_pool(
            tst_pooled_hp_dn_signals)
        logger.info(f'Saving to {tst_hp_dn_filename} ...')
        tst_decoded_hp_dn_signals_df.to_pickle(tst_hp_dn_filename)
        send_line_notification(f'Finish processing {tst_hp_dn_filename} !')


if __name__ == '__main__':
    logger = getLogger(__name__)
    logger = logInit(logger, './logs/', 'preprocess.log')
    preprocess(logger)