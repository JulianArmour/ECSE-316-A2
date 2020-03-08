from cmath import exp, pi
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
    ws = [exp(-2j * pi * k / n) for k in range(n // 2)]
    return [even + w * odd for even, odd, w in zip(evens, odds, ws)] + \
           [even - w * odd for even, odd, w in zip(evens, odds, ws)]


def ifastft(sig):
    n = len(sig)
    if n <= 1:
        return sig
    evens = fastft(sig[0::2])
    odds = fastft(sig[1::2])
    ws = [exp(2j * pi * k / n) for k in range(n // 2)]
    return [(even + w * odd) / n for even, odd, w in zip(evens, odds, ws)] + \
           [(even - w * odd) / n for even, odd, w in zip(evens, odds, ws)]


def fastft2(sig):
    # perform transform on columns of sig, need to transpose to do so
    transform1 = [fastft(sigCol) for sigCol in np.transpose(sig)]
    # perform transform on rows of sig, transpose previous result
    return [fastft(sigRow) for sigRow in np.transpose(transform1)]


def ifastft2(sig):
    # perform transform on columns of sig, need to transpose to do so
    transform1 = [ifastft(sigCol) for sigCol in np.transpose(sig)]
    # perform transform on rows of sig, transpose previous result
    return [ifastft(sigRow) for sigRow in np.transpose(transform1)]

