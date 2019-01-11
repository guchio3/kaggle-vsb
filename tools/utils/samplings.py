import pandas as pd


def get_neg_ds_index(target, random_state=None):
    '''
    get negative donwnsamling index.

    '''
    res_index = target[target == 1].index
    res_index.append(target[target == 0].sample(
        len(res_index), random_state=random_state).index)
    return res_index
