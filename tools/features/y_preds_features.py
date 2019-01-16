import numpy as np
import pandas as pd


def place_0(x):
    return x.iloc[0]


def place_1(x):
    return x.iloc[1]


def place_2(x):
    return x.iloc[2]


def y_preds_features(oofs, val_idxes):
    # Make the base df
    y_preds_df = pd.DataFrame()
    y_preds_df['signal_id'] = np.concatenate(val_idxes).astype(int)
    y_preds_df['y_pred'] = np.concatenate(oofs).astype('float64')
    y_preds_df['id_measurement'] = (y_preds_df['signal_id'] // 3).astype(int)
    y_preds_df.sort_values('signal_id', inplace=True)
    y_preds_df.reset_index(drop=True, inplace=True)

    # Agg
    agg_y_preds_df = y_preds_df.groupby('id_measurement').agg({
        'y_pred': ['max', 'min', 'mean', 'std', place_0, place_1, place_2]
    })
    agg_y_preds_df.columns = [e[0] + '_' + e[1]
                              for e in agg_y_preds_df.columns]

    # Merge
    y_preds_df = y_preds_df.merge(
        agg_y_preds_df, on='id_measurement', how='left')\
        .drop(['signal_id', 'id_measurement'], axis=1)
    return y_preds_df
