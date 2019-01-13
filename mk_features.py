import gc
from logging import getLogger

from .tools.features.base_features import (_base_features,
                                           _load_base_features_src)
from .tools.features.feature_tools import _mk_features
from .tools.utils.general_utils import dec_timer, logInit, parse_args


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
    logger.info(
        f'============ EXP {args.exp_ids[0]}-{args.sub_id}, START MAKING FEATURES =============')
    mk_features(args, logger)
