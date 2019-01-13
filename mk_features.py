import sys
import gc
from logging import getLogger

sys.path.append('./tools/utils/')
from general_utils import parse_args, logInit, sel_log, dec_timer

sys.path.append('./tools/features/')
from base_features import _base_features, _load_base_features_src
from feature_tools import _mk_features


@dec_timer
def mk_features(args, logger):
    series_df, meta_df = None, None
    # base features
    series_df, meta_df = _mk_features(
            _load_base_features_src, _base_features,
            args.nthread, args.exp_ids, args.test,
            series_df, meta_df, logger=logger)
    gc.collect()


if __name__ == '__main__':
    logger = getLogger(__name__)
    logger = logInit(logger, './logs/', 'mk_features.log')
    args = parse_args(logger)

    logger.info('')
    logger.info('')
    logger.info(f'============ EXP {args.exp_ids[0]}-{args.sub_id}, START MAKING FEATURES =============')
    mk_features(args, logger)
