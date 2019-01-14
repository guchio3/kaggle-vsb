import numpy as np
import pandas as pd
import pywt
import scipy
from scipy import signal
from scipy.signal import butter
from statsmodels.robust import mad
from tqdm import tqdm

# ==========================================
#  util tools for preprocessing
# ==========================================


def decode_signals_after_pool(pooled_signals):
    signal_ids, signals = [], []
    for pooled_signal in tqdm(pooled_signals):
        signal_ids.append(pooled_signal[0])
        signals.append(pooled_signal[1])
    # using np array for decoding is verrrrry faster
    decoded_signals_df = pd.DataFrame(np.array(signals).astype('float16')).T
    # the column should be int for sorting
    signal_ids = pd.Series(signal_ids).astype('int8')
    decoded_signals_df.columns = signal_ids
    decoded_signals_df.sort_index(axis=1, ascending=True, inplace=True)
    # change col type to str
    decoded_signals_df.columns = signal_ids.astype(str)
    return decoded_signals_df


# ==========================================
#  tools for signal conversion
# ==========================================
# Signal characteristics
# From @randxie
# https://github.com/randxie/Kaggle-VSB-Baseline/blob/master/src/utils/util_signal.py
SAMPLING_FREQ = 80000 / 0.02  # 80,000 data points taken over 20 ms


def add_high_pass_filter(id_signal_pair, low_freq=1000,
                         sample_fs=SAMPLING_FREQ):
    """
    From @randxie https://github.com/randxie/Kaggle-VSB-Baseline/
        blob/master/src/utils/util_signal.py
    Modified to work with scipy version 1.1.0
        which does not have the fs parameter
    """
    signal_id, x = id_signal_pair

    cutoff = 1000
    nyq = 0.5 * sample_fs
    normal_cutoff = cutoff / nyq

    # Fault pattern usually exists in high frequency band.
    # According to literature, the pattern is visible above 10^4 Hz.
    # scipy version 1.2.0
    # sos = butter(10, low_freq, btype='hp', fs=sample_fs, output='sos')

    # scipy version 1.1.0
    sos = butter(10, normal_cutoff, btype='hp', output='sos')
    filtered_sig = signal.sosfilt(sos, x)

    return signal_id, filtered_sig


def denoise_signal(id_signal_pair, wavelet='db4', level=1):
    """
    1. Adapted from waveletSmooth function found here:
    http://connor-johnson.com/2016/01/24/using-pywavelets-to-remove-high-frequency-noise/
    2. Threshold equation and using hard mode in threshold as mentioned
    in section '3.2 denoising based on optimized singular values' from paper:
    http://dspace.vsb.cz/bitstream/handle/10084/133114/VAN431_FEI_P1807_1801V001_2018.pdf
    """
    signal_id, x = id_signal_pair

    # Decompose to get the wavelet coefficients
    coeff = pywt.wavedec(x, wavelet, mode="per")

    # Calculate sigma for threshold as defined in
    # http://dspace.vsb.cz/bitstream/handle/10084/133114/VAN431_FEI_P1807_1801V001_2018.pdf
    sigma = (1 / 0.6745) * mad(coeff[-level])
    # sigma = mad( coeff[-level] )

    # Calculte the univeral threshold
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard')
                 for i in coeff[1:])

    # Reconstruct the signal using the thresholded coefficients
    return signal_id, pywt.waverec(coeff, wavelet, mode='per')
