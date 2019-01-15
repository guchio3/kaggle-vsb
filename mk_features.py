import gc
import sys
from logging import getLogger

from tools.features.base_features import (_base_features,
                                          _load_base_features_src)
from tools.features.feature_tools import _mk_features
from tools.features.hp_dn_3phase_features import (_hp_dn_3phase_basic_features,
                                                  _load_hp_dn_3phase_features_src)
from tools.features.hp_dn_features import (_hp_dn_basic_features,
                                           _load_hp_dn_features_src)
from tools.features.hp_features import (_hp_basic_features,
                                        _load_hp_features_src)
from tools.utils.general_utils import (dec_timer, logInit, parse_args,
                                       send_line_notification)


@dec_timer
def mk_features(args, logger):
    meta_df = None
    series_df = None
    hp_series_df = None
    hp_dn_series_df = None
    # base features
    series_df, meta_df = _mk_features(
        _load_base_features_src, _base_features,
        args.nthread, args.exp_ids, args.test,
        series_df, meta_df, logger=logger)
    # HPF features
    hp_series_df, meta_df = _mk_features(
        _load_hp_features_src, _hp_basic_features,
        args.nthread, args.exp_ids, args.test,
        hp_series_df, meta_df, logger=logger)
    # HPF and Denoising (DN) features
    hp_dn_series_df, meta_df = _mk_features(
        _load_hp_dn_features_src, _hp_dn_basic_features,
        args.nthread, args.exp_ids, args.test,
        hp_dn_series_df, meta_df, logger=logger)
    # 3phase HPF and DN features
    hp_dn_series_df, meta_df = _mk_features(
        _load_hp_dn_3phase_features_src, _hp_dn_3phase_basic_features,
        args.nthread, args.exp_ids, args.test,
        hp_dn_series_df, meta_df, logger=logger)
    gc.collect()


if __name__ == '__main__':
    logger = getLogger(__name__)
    logger = logInit(logger, './logs/', 'mk_features.log')
    args = parse_args(logger)

    logger.info('')
    logger.info('')
    logger.info(
        f'============ EXP {args.exp_ids[0]}, START MAKING FEATURES =============')
    mk_features(args, logger)
    send_line_notification(f'Finished: {" ".join(sys.argv)}')
