import datetime
import pickle
import sys
import warnings
from itertools import tee
from logging import getLogger

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

import tools.models.my_lightgbm as mlgb
from tools.features.feature_tools import load_features
from tools.utils.general_utils import (dec_timer, load_configs, log_evaluation,
                                       logInit, parse_args, sel_log)
from tools.utils.metrics import calc_best_MCC, calc_MCC, lgb_MCC
from tools.utils.samplings import resampling
from tools.utils.visualizations import save_importance

warnings.simplefilter(action='ignore', category=FutureWarning)


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
    # -- Prepare for training
    exp_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    train_base_dir = './inputs/train/'
    configs = load_configs('./config.yml', logger)

    # -- Load train data
    sel_log('loading training data ...', None)
    target = pd.read_pickle(
        train_base_dir + 'target.pkl.gz', compression='gzip')
    id_measurement = pd.read_pickle(
        train_base_dir + 'id_measurement.pkl.gz', compression='gzip')
    # Cache can be used only in train
    if args.use_cached_features:
        features_df = pd.read_pickle(
            './inputs/train/cached_featurse.pkl.gz', compression='gzip')
    else:
        features_df = load_features(
            configs['features'], train_base_dir, logger)
        # gen cache file if specified for the next time
        if args.gen_cached_features:
            features_df.to_pickle(
                './inputs/train/cached_featurse.pkl.gz', compression='gzip')

    # -- Data resampling
    # Stock original data for validation
    if configs['preprocess']['resampling']:
        target, id_measurement, features_df = resampling(
            target, id_measurement, features_df,
            configs['preprocess']['resampling_type'],
            configs['preprocess']['resampling_seed'], logger)
    sel_log(f'the shape features_df is {features_df.shape}', logger)

    # -- Split using group k-fold w/ shuffling
    # NOTE: this is not stratified, I wanna implement it in the future
    gss = GroupShuffleSplit(configs['train']['fold_num'], random_state=71)
    folds = gss.split(features_df, target, groups=id_measurement)
    folds, pred_folds = tee(folds)

    # -- Make training dataset
    train_set = mlgb.Dataset(features_df.values, target.values)

    # -- CV
    # Set params
    PARAMS = configs['lgbm_params']
    PARAMS['nthread'] = args.nthread

    sel_log('start training ...', None)
    hist, cv_model = mlgb.cv(
        params=PARAMS,
        folds=folds,
        train_set=train_set,
        verbose_eval=50,
        early_stopping_rounds=100,
        feval=lgb_MCC,
        callbacks=[log_evaluation(logger, period=50)],
    )

    # -- Prediction
    sel_log('predicting ...', logger)
    oofs = []
    y_trues = []
    scores = []
    fold_importance_dict = {}
    for i, idxes in tqdm(list(enumerate(pred_folds))):
        trn_idx, val_idx = idxes
        booster = cv_model.boosters[i]

        # Get and store oof and y_true
        y_pred = booster.predict(features_df.values[val_idx])
        y_true = target.values[val_idx]
        oofs.append(y_pred)
        y_trues.append(y_true)

        # Calc MCC using thresh of 0.5
        MCC = calc_MCC(y_true, y_pred, 0.5)
        scores.append(MCC)

        # Save importance info
        fold_importance_df = pd.DataFrame()
        fold_importance_df['split'] = booster.feature_importance('split')
        fold_importance_df['gain'] = booster.feature_importance('gain')
        fold_importance_dict[i] = fold_importance_df

    sel_log(f'MCC_mean: {np.mean(scores)}, MCC_std: {np.std(scores)}', logger)

    # Calc best MCC
    sel_log('calculating the best MCC ...', None)
    y_true = np.concatenate(y_trues, axis=0)
    y_pred = np.concatenate(oofs, axis=0)
    best_MCC, best_thresh = calc_best_MCC(y_true, y_pred, bins=30)
    sel_log(f'best_MCC: {best_MCC}, best_thresh: {best_thresh}', logger)

    # -- Post processings
    filename_base = f'{args.exp_ids[0]}_{exp_time}_{best_MCC:.4}_{best_thresh:.2}'

    # Save oofs
    with open('./oofs/' + filename_base + '_oofs.pkl', 'wb') as fout:
        pickle.dump(oofs, fout)

    # Save importances
    save_importance(configs['features'], fold_importance_dict,
                    './importances/' + filename_base + '_importances')

    # Save trained models
    with open(
            './trained_models/' + filename_base + '_models.pkl', 'wb') as fout:
        pickle.dump(cv_model, fout)

    # --- Make submission file
    if args.test:
        # -- Prepare for test
        test_base_dir = './inputs/test/'

        sel_log('loading test data ...', None)
        test_features_df = load_features(
            configs['features'], test_base_dir, logger)

        # -- Prediction
        sel_log('predicting ...', None)
        preds = []
        for booster in tqdm(cv_model.boosters):
            preds.append(booster.predict(test_features_df.values))

        # -- Make submission file
        sub_values = np.mean(preds, axis=0)
        target_values = (sub_values > best_thresh).astype(np.int32)

        sel_log(f'loading sample submission file ...', None)
        sub_df = pd.read_csv('./inputs/origin/sample_submission.csv')
        sub_df.target = target_values

        # print stats
        sel_log(f'prositive percentage: \
                {sub_df.target.sum()/sub_df.target.count()*100:.3}%',
                logger=logger)

        submission_filename = f'./submissions/{filename_base}_sub.csv.gz'
        sel_log(f'saving submission file to {submission_filename}', logger)
        sub_df.to_csv(submission_filename, compression='gzip', index=False)


if __name__ == '__main__':
    logger = getLogger(__name__)
    logger = logInit(logger, './logs/', 'lgb_train.log')
    args = parse_args(logger)

    logger.info('')
    logger.info('')
    logger.info(
        f'============ EXP {args.exp_ids[0]}-{args.sub_id}, START TRAINING =============')
    train(args, logger)
