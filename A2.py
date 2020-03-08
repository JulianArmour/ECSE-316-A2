from cmath import exp, pi


def slowft(signal):
    N = len(signal)
    return [sum([signal[n] * exp(-2j*pi*k*n/N) for n in range(N)]) for k in range(N)]


def fastft(signal):
    n = len(signal)
    if n <= 1:
        return signal
    evens = fastft(signal[0::2])
    odds = fastft(signal[1::2])
    ws = [exp(-2j * pi * k / n) for k in range(n // 2)]
    return [even + w * odd for even, odd, w in zip(evens, odds, ws)] + \
           [even - w * odd for even, odd, w in zip(evens, odds, ws)]


def fastift(signal):
    n = len(signal)
    if n <= 1:
        return signal
    evens = fastft(signal[0::2])
    odds = fastft(signal[1::2])
    ws = [exp(2j * pi * k / n) for k in range(n // 2)]
    return [(even + w * odd) / n for even, odd, w in zip(evens, odds, ws)] + \
           [(even - w * odd) / n for even, odd, w in zip(evens, odds, ws)]
