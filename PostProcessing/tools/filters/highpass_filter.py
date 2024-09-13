from scipy import signal


def _butter_highpass(cutoff, fs, order=5):
    return signal.butter(order, cutoff, fs=fs, btype="high", analog=True, output="ba")


def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = _butter_highpass(cutoff, fs, order)
    return signal.lfilter(b, a, data)


def _cheby1_highpass(cutoff, fs, order=2):
    return signal.cheby1(order, cutoff, fs=fs, btype="high", analog=True)


def cheby1_highpass_filter(data, cutoff, fs, order=2):
    b, a = _cheby1_highpass(cutoff, fs, order)
    return signal.lfilter(b, a, data)


def _cheby2_highpass(cutoff, fs, order=2):
    return signal.cheby1(order, cutoff, fs=fs, btype="high", analog=True)


def cheby2_highpass_filter(data, cutoff, fs, order=2):
    b, a = _cheby2_highpass(cutoff, fs, order)
    return signal.lfilter(b, a, data)

