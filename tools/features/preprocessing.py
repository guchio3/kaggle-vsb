import numpy as np
import pandas as pd
import pywt
import scipy
from scipy import signal
from scipy.signal import butter
from statsmodels.robust import mad

# ==========================================
#  tools for signal preprocessing
# ==========================================
# Signal characteristics
# From @randxie
# https://github.com/randxie/Kaggle-VSB-Baseline/blob/master/src/utils/util_signal.py
SAMPLING_FREQ = 80000 / 0.02  # 80,000 data points taken over 20 ms


def add_high_pass_filter(x, low_freq=1000, sample_fs=SAMPLING_FREQ):
    """
    From @randxie https://github.com/randxie/Kaggle-VSB-Baseline/
        blob/master/src/utils/util_signal.py
    Modified to work with scipy version 1.1.0
        which does not have the fs parameter
    """

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

    return filtered_sig


def denoise_signal(x, wavelet='db4', level=1):
    """
    1. Adapted from waveletSmooth function found here:
    http://connor-johnson.com/2016/01/24/using-pywavelets-to-remove-high-frequency-noise/
    2. Threshold equation and using hard mode in threshold as mentioned
    in section '3.2 denoising based on optimized singular values' from paper:
    http://dspace.vsb.cz/bitstream/handle/10084/133114/VAN431_FEI_P1807_1801V001_2018.pdf
    """

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
    return pywt.waverec(coeff, wavelet, mode='per')
