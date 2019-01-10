import sys
import time
import inspect
import argparse
import yaml
import functools
from tqdm import tqdm
from logging import Formatter, StreamHandler, FileHandler, DEBUG

import pandas as pd

from lightgbm.callback import _format_eval_result


# ==========================================
#  exp settings utils
# ==========================================
def parse_args(logger=None):
    '''
    Policy
    ------------
    * experiment id must be required

    '''
    parser = argparse.ArgumentParser(
        prog='XXX.py',
        usage='ex) python XXX.py -e e001 -n 31',
        description='short explanation of args',
        add_help=True,
    )
    parser.add_argument('-e', '--exp_ids',
                        help='experiment id',
                        type=str,
                        nargs='+',
                        required=True)
    parser.add_argument('-n', '--nthread',
                        help='number of avalable threads.',
                        type=int,
                        required=True)
    parser.add_argument('-t', '--test',
                        help='set when you run test',
                        action='store_true',
                        default=False)
    parser.add_argument('-c', '--use_cached_features',
                        help='set when you use cached features',
                        action='store_true',
                        default=False)
    parser.add_argument('-g', '--gen_cached_features',
                        help='set when you generate cached features',
                        action='store_true',
                        default=False)

    args = parser.parse_args()
    sel_log(f'args: {sorted(vars(args).items())}', logger)
    assert not (args.use_cached_features and args.gen_cached_features), \
        'U can not use use and generate chached features simultaneously !'
    return args


def load_configs(path, logger=None):
    '''
    Load config file written in yaml format.

    '''
    with open(path, 'r') as fin:
        configs = yaml.load(fin)
    sel_log(f'configs: {configs}', logger)
    return configs


# ==========================================
#  logging utils
# ==========================================
def _myself():
    '''
    get func name

    '''
    return inspect.stack()[1][3]


def logInit(logger, log_dir, log_filename):
    '''
    Init the logger.

    '''
    log_fmt = Formatter('%(asctime)s %(name)s \
            %(lineno)d [%(levelname)s] [%(funcName)s] %(message)s ')
    handler = StreamHandler()
    handler.setLevel('INFO')
    handler.setFormatter(log_fmt)
    logger.addHandler(handler)

    handler = FileHandler(log_dir + log_filename, 'a')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)
    return logger


def sel_log(message, logger, debug=False):
    '''
    Use logger if specified one, and use print otherwise.
    Also it's possible to specify to use debug mode (default: info mode).

    The func name is the shorter version of selective_logging.

    '''
    if logger:
        if debug:
            logger.debug(message)
        else:
            logger.info(message)
    else:
        print(message)


def log_evaluation(logger, period=1, show_stdv=True, level=DEBUG):
    def _callback(env):
        if period > 0 and \
                env.evaluation_result_list and \
                (env.iteration + 1) % period == 0:
            result = '\t'.join(
                    [_format_eval_result(x, show_stdv)
                        for x in env.evaluation_result_list])
            logger.log(level, '[{}]\t{}'.format(env.iteration+1, result))
    _callback.order = 10
    return _callback


def dec_timer(func):
    '''
    Decorator which measures the processing time of the func.

    '''
    # wraps func enable to hold the func name
    @functools.wraps(func)
    def _timer(*args, **kwargs):
        t0 = time.time()
        start_str = f'[{func.__name__}] start'
        if 'logger' in kwargs:
            logger = kwargs['logger']
        else:
            logger = None
        sel_log(start_str, logger)

        # run the func
        res = func(*args, **kwargs)

        end_str = f'[{func.__name__}] done in {time.time() - t0:.1f} s'
        sel_log(end_str, logger)
        return res

    return _timer
