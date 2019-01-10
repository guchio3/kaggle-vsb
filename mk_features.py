import sys
import gc
from logging import getLogger

sys.path.append('./tools/utils/')
from general_utils import parse_args, logInit, sel_log, dec_timer

sys.path.append('./tools/features/')
from base_features import mk_base_features


@dec_timer
def mk_features(args, logger):
    # series_df, meta_df = mk_base_features(
    series_df, meta_df = mk_base_features(
            args.nthread, args.exp_ids, args.test, logger=logger)
    gc.collect()


if __name__ == '__main__':
    logger = getLogger(__name__)
    logger = logInit(logger, './logs/', 'mk_features.log')
    args = parse_args(logger)

    logger.info('')
    logger.info('')
    logger.info(f'============ EXP {args.exp_ids}, START MAKING FEATURES =============')
    mk_features(args, logger)
