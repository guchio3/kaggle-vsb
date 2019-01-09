import sys
import datetime
import pickle
from itertools import tee
from tqdm import tqdm
from logging import getLogger

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit

sys.path.append('./tools/utils')
# from metrics import MCC
from general_utils import parse_args, load_configs, logInit, sel_log, log_evaluation, dec_timer

sys.path.append('./tools/features')
from feature_tools import load_features

sys.path.append('../guchio_utils')
import my_lightgbm as mlgb


@dec_timer
def train(args, logger):
    '''
    policy
    ------------
    * use original functions only if there's no pre-coded functions
        in useful libraries such as sklearn.

    todos
    ------------
    * load features
    * train the model
    * save the followings
        * logs
        * oofs
        * importances
        * trained models
        * submissions (if test mode)

    '''
    # Prepare for training
    train_base_dir = './inputs/train/'
    configs = load_configs('./config.yml', logger)

    # Load train data
    sel_log('loading data ...', None)
    target = pd.read_pickle(
            train_base_dir + 'target.pkl.gz', compression='gzip')
    id_measurement = pd.read_pickle(
            train_base_dir + 'target.pkl.gz', compression='gzip')
    features_df = load_features(configs['features'], train_base_dir, logger)

    # Split using group k-fold w/ shuffling
    gss = GroupShuffleSplit(configs['train']['fold_num'], random_state=71)
    folds = gss.get_n_splits(features_df, target, id_measurement)
    folds, pred_folds = tee(folds)

    # Make training dataset
    train_set = mlgb.Dataset(features_df.values, target.values)

    # Set params
    PARAMS = configs['lgbm_params']
    PARAMS['nthread'] = args.nthread

    # CV
    sel_log('start training ...', None)
    hist, cv_model = mlgb.cv(
                        params=PARAMS,
                        folds=folds,
                        train_set=train_set,
                        verbose_eval=100,
#                        early_stopping_rounds=,
#                        feval=,
                        callbacks=[log_evaluation(logger, period=10)],
                     )

    # Prediction
    sel_log('predicting ...', logger)
    oofs = []
    scores = []
    fold_importances = {}
    for i, trn_idx, val_idx in tqdm(enumerate(pred_folds)):
        booster = cv_model.boosters[i]

        # Get oof
        y_pred = booster.predict(features_df.values[val_idx])
        y_true = target.values[val_idx]
        oofs.append(y_pred)
        
        # Calc best MCC
        best_mcc = calc_best_mcc(y_ture, y_pred)
        scores.append(best_mcc)

        # Save importance info
        fold_importance_df = pd.DataFrame()
        fold_importance_df['split'] = booster.feature_importance('split')
        fold_importance_df['gain'] = booster.feature_importance('gain')
        fold_importances[i] = fold_importances

    filename_base = f''

    # Save oofs
    with open('./oofs/' + filename_base + '_oofs.pkl', 'w') as fout:
        pickle.dump(oofs, fout)

    # Save importances
#    importance_df = get_i
#    with open('./figs/importances/' + filename_base + '_importances.pkl', 'w') as fout:
#        pickle.dump(importnce_df, fout)

    # Save trained models
    with open('./trained_models/' + filename_base + '_models.pkl', 'w') as fout:
        pickle.dump(cv_model, fout)



#    test_base_dir = './inputs/test/'


if __name__ == '__main__':
    logger = getLogger(__name__)
    logger = logInit(logger, './logs/', 'lgb_train.log')
    logger.info('')
    logger.info('')
    logger.info('============ START TRAINING =============')
    args = parse_args(logger)
    train(args, logger)
