from cmath import exp, pi
import cv2
import numpy as np


def slowft(sig):
    sig_len = len(sig)
    return [sum([sig[n] * exp(-2j * pi * k * n / sig_len) for n in range(sig_len)]) for k in range(sig_len)]


def fastft(sig):
    n = len(sig)
    if n <= 1:
        return sig
    evens = fastft(sig[0::2])
    odds = fastft(sig[1::2])
    ws = np.exp(-2j * pi * np.arange(n // 2) / n)
    return np.append(evens + ws * odds, evens - ws * odds)


def ifastft(sig):
    """
    using conjugation method
    source: https://www.dsprelated.com/showarticle/800.php
    :param sig: frequency domain numpy array
    :return: time domain numpy array
    """
    return np.conjugate(fastft(np.conjugate(sig))) / len(sig)


def fastft2(sig):
    def axis_fft(a, axis):
        return np.apply_along_axis(fastft, axis, a)
    return np.apply_over_axes(axis_fft, sig, [0, 1])


def ifastft2(sig):
    def axis_ifft(a, axis):
        return np.apply_along_axis(ifastft, axis, a)
    return np.apply_over_axes(axis_ifft, sig, [0, 1])


def pow2_ceil(num: int) -> int:
    """
    :param num: integer to round up
    :return: num rounded up to the nearest power of 2
    """
    return int(np.exp2(np.ceil(np.log2(num))))


def pad_image(image):
    height, width = image.shape
    x_pad = pow2_ceil(width) - width
    y_pad = pow2_ceil(height) - height
    return cv2.copyMakeBorder(image, 0, y_pad, 0, x_pad, cv2.BORDER_CONSTANT, 0)
