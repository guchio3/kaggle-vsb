import datetime
import pickle
import sys
import warnings
from itertools import tee
from logging import getLogger

import lightgbm
import numpy as np
import pandas as pd
from sklearn.model_selection import (GroupKFold, GroupShuffleSplit,
                                     StratifiedKFold)
from tqdm import tqdm

import tools.models.my_lightgbm as mlgb
from tools.features.feature_tools import load_features, select_features
from tools.features.y_preds_features import y_preds_features
from tools.utils.general_utils import (dec_timer, load_configs, log_evaluation,
                                       logInit, parse_args, sel_log,
                                       send_line_notification)
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
    if configs['train']['feature_selection']:
        features_df = select_features(features_df,
                                      configs['train']['feature_select_path'],
                                      'gain_mean',
                                      configs['train']['feature_topk'])
    features = features_df.columns

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
    if configs['train']['fold_type'] == 'gkf':
        gkf = GroupKFold(configs['train']['fold_num'])
        folds = gkf.split(features_df, target, groups=id_measurement)
    elif configs['train']['fold_type'] == 'skf':
        skf = StratifiedKFold(configs['train']['fold_num'], random_state=71)
        folds = skf.split(features_df, target, groups=id_measurement)
    else:
        print(f"ERROR: wrong fold_type, {configs['train']['fold_type']}")
    # gss = GroupShuffleSplit(configs['train']['fold_num'], random_state=71)
    # folds = gss.split(features_df, target, groups=id_measurement)
    folds, pred_folds = tee(folds)
    if configs['train']['label_train']:
        folds, folds_2 = tee(folds)
        folds, pred_folds_2 = tee(folds)

    # -- Make training dataset
    train_set = mlgb.Dataset(features_df.values, target.values)

    # -- CV
    # Set params
    PARAMS = configs['lgbm_params']
    PARAMS['nthread'] = args.nthread

    sel_log('start training ...', None)
    hist, cv_model = mlgb.cv(
        params=PARAMS,
        num_boost_round=10000,
        folds=folds,
        train_set=train_set,
        verbose_eval=50,
        early_stopping_rounds=200,
        metrics='auc',
        # feval=lgb_MCC,
        callbacks=[log_evaluation(logger, period=50)],
    )

    # -- Prediction
    if configs['train']['single_model']:
        best_iter = cv_model.best_iteration
        single_train_set = lightgbm.Dataset(features_df.values, target.values)
        single_booster = lightgbm.train(
            params=PARAMS,
            num_boost_round=int(best_iter * 1.3),
            train_set=single_train_set,
            valid_sets=[single_train_set],
            verbose_eval=50,
            early_stopping_rounds=200,
            callbacks=[log_evaluation(logger, period=50)],
        )
        oofs = [single_booster.predict(features_df.values)]
        y_trues = [target]
        val_idxes = [features_df.index]
        scores = []
        y_true, y_pred = target, oofs[0]
        fold_importance_df = pd.DataFrame()
        fold_importance_df['split'] = single_booster.\
            feature_importance('split')
        fold_importance_df['gain'] = single_booster.\
            feature_importance('gain')
        fold_importance_dict = {0: fold_importance_df}
    else:
        sel_log('predicting using cv models ...', logger)
        oofs = []
        y_trues = []
        val_idxes = []
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
            val_idxes.append(val_idx)

            # Calc MCC using thresh of 0.5
            MCC = calc_MCC(y_true, y_pred, 0.5)
            scores.append(MCC)

            # Save importance info
            fold_importance_df = pd.DataFrame()
            fold_importance_df['split'] = booster.feature_importance('split')
            fold_importance_df['gain'] = booster.feature_importance('gain')
            fold_importance_dict[i] = fold_importance_df

#        y_true = np.concatenate(y_trues, axis=0)
#        y_pred = np.concatenate(oofs, axis=0)
        sel_log(
            f'MCC_mean: {np.mean(scores)}, MCC_std: {np.std(scores)}',
            logger)

    # Calc best MCC
    sel_log('calculating the best MCC ...', None)
    best_MCC, best_threshs = calc_best_MCC(y_trues, oofs, bins=3000)
    sel_log(f'best_MCC: {best_MCC}', logger)

    # -- Post processings
    filename_base = f'{args.exp_ids[0]}_{exp_time}_{best_MCC:.4}'

    # Save oofs
    with open('./oofs/' + filename_base + '_oofs.pkl', 'wb') as fout:
        pickle.dump([val_idxes, oofs, best_threshs], fout)

    # Save importances
    # save_importance(configs['features'], fold_importance_dict,
    save_importance(features, fold_importance_dict,
                    './importances/' + filename_base + '_importances')

    # Save trained models
    with open(
            './trained_models/' + filename_base + '_models.pkl', 'wb') as fout:
        pickle.dump(
            single_booster if configs['train']['single_model'] else cv_model,
            fout)

#    # -- Retrainig using the preds
#    if configs['train']['label_train']:
#        # -- Make training dataset
#        y_preds_df = y_preds_features(oofs, val_idxes)
#        features_df_2 = features_df
#        features_df_2 = pd.concat([features_df_2, y_preds_df], axis=1)
#        features_2 = features_df_2.columns
#        train_set_2 = mlgb.Dataset(features_df_2.values, target.values)
#
#        # -- CV
#        sel_log('RETRAINED -- start training ...', None)
#        hist_2, cv_model_2 = mlgb.cv(
#            params=PARAMS,
#            num_boost_round=10000,
#            folds=folds_2,
#            train_set=train_set_2,
#            verbose_eval=50,
#            early_stopping_rounds=200,
#            metrics='auc',
#            # feval=lgb_MCC,
#            callbacks=[log_evaluation(logger, period=50)],
#        )
#
#        # -- Prediction
#        sel_log('RETRAINED -- predicting ...', logger)
#        oofs_2 = []
#        y_trues_2 = []
#        val_idxes_2 = []
#        scores_2 = []
#        fold_importance_dict_2 = {}
#        for i, idxes in tqdm(list(enumerate(pred_folds_2))):
#            trn_idx, val_idx = idxes
#            booster = cv_model_2.boosters[i]
#
#            # Get and store oof and y_true
#            y_pred = booster.predict(features_df_2.values[val_idx])
#            y_true = target.values[val_idx]
#            oofs_2.append(y_pred)
#            y_trues_2.append(y_true)
#            val_idxes_2.append(val_idx)
#
#            # Calc MCC using thresh of 0.5
#            MCC = calc_MCC(y_true, y_pred, 0.5)
#            scores_2.append(MCC)
#
#            # Save importance info
#            fold_importance_df = pd.DataFrame()
#            fold_importance_df['split'] = booster.feature_importance('split')
#            fold_importance_df['gain'] = booster.feature_importance('gain')
#            fold_importance_dict_2[i] = fold_importance_df
#
#        sel_log(
#            f'RETRAINED -- MCC_mean: {np.mean(scores_2)}, MCC_std: {np.std(scores_2)}',
#            logger)
#
#        # Calc best MCC
#        sel_log('RETRAINED -- calculating the best MCC ...', None)
#        y_true_2 = np.concatenate(y_trues_2, axis=0)
#        y_pred_2 = np.concatenate(oofs_2, axis=0)
#        best_MCC_2, best_thresh_2 = calc_best_MCC(y_true_2, y_pred_2, bins=3000)
#        sel_log(
#            f'RETRAINED -- best_MCC: {best_MCC_2}, best_thresh: {best_thresh_2}',
#            logger)
#
#        # -- Post processings
#        filename_base = f'{args.exp_ids[0]}_{exp_time}_{best_MCC_2:.4}_{best_thresh_2:.3}'
#
#        # Save oofs
#        with open('./oofs/' + filename_base + '_oofs_retrained.pkl', 'wb') as fout:
#            pickle.dump([val_idxes_2, oofs_2], fout)
#
#        # Save importances
#        save_importance(features_2, fold_importance_dict_2,
#                        './importances/' + filename_base + '_importances_retrained')
#
#        # Save trained models
#        with open(
#                './trained_models/' + filename_base + '_models_retrained.pkl', 'wb') as fout:
#            pickle.dump(cv_model_2, fout)

    # --- Make submission file
    if args.test:
        # -- Prepare for test
        test_base_dir = './inputs/test/'

        sel_log('loading test data ...', None)
        test_features_df = load_features(
            configs['features'], test_base_dir, logger)

        # -- Prediction
        sel_log('predicting for test ...', None)
        preds = []
        for booster, best_thresh in tqdm(zip(cv_model.boosters, best_threshs)):
            pred = booster.predict(test_features_df.values)
            preds.append(pred * 0.5 / best_thresh)
            # preds.append(pred > best_thresh)
        sub_values = np.mean(preds, axis=0)
        target_values = (sub_values > 0.5).astype(np.int32)

#        if configs['train']['label_train']:
#            # -- Use retrained info
#            test_y_preds_df = y_preds_features(
#                [sub_values], [np.arange(len(sub_values))])
#            test_features_df = pd.concat(
#                [test_features_df, test_y_preds_df], axis=1)
#            sel_log('RETRAINED -- predicting ...', None)
#            preds = []
#            for booster in tqdm(cv_model_2.boosters):
#                preds.append(booster.predict(test_features_df.values))
#            sub_values = np.mean(preds, axis=0)
#            target_values = (sub_values > best_thresh_2).astype(np.int32)

        # -- Make submission file
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
        f'============ EXP {args.exp_ids[0]}, START TRAINING =============')
    train(args, logger)
    send_line_notification(f'Finished: {" ".join(sys.argv)}')
