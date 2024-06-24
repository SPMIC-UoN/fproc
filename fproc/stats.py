import logging

import numpy as np
import scipy

LOG = logging.getLogger(__name__)

def _sample(arr):
    return arr

def median(arr):
    return np.nanmedian(_sample(arr))

def skew(arr):
    return scipy.stats.skew(arr.flatten(), nan_policy='omit')

def kurtosis(arr):
    return scipy.stats.kurtosis(arr.flatten(), nan_policy='omit')

def mode(arr):
    """
    For FWHM, fit a Gaussian and return peak location of that
    """
    loc, _scale = scipy.stats.norm.fit(arr)
    return loc

def n(arr):
    return np.count_nonzero(~np.isnan(arr))

def lq(arr):
    return np.nanquantile(_sample(arr), 0.25)

def uq(arr):
    return np.nanquantile(_sample(arr), 0.75)

def iqr(arr):
    uq, lq = tuple(np.nanquantile(_sample(arr), [0.75, 0.25]))
    return uq-lq

def _get_iqdata(arr):
    if n(arr) == 0:
        return np.array([])
    uq, lq = tuple(np.nanquantile(_sample(arr), [0.75, 0.25]))
    arr2 = arr[arr < uq]
    if arr2.size == 0:
        arr2 = arr
    arr3 = arr2[arr2 > lq]
    if arr3.size == 0:
        arr3 = arr2
    return arr3

def iqn(arr):
    arr = _get_iqdata(arr)
    return np.count_nonzero(~np.isnan(arr))

def iqmean(arr):
    arr = _get_iqdata(arr)
    return np.nanmean(arr)

def iqstd(arr):
    arr = _get_iqdata(arr)
    return np.nanstd(arr)

def fwhm(arr):
    """
    For FWHM, fit a Gaussian and return fwhm of that
    """
    _loc, scale = scipy.stats.norm.fit(arr)
    return 2.355*scale

def mean(arr):
    return np.nanmean(arr)

def std(arr):
    return np.nanstd(arr)

def max(arr):
    return np.nanmax(arr)

def min(arr):
    return np.nanmin(arr)

STAT_IMPLS = {
    "mean" : mean,
    "std" : std,
    "median" : median,
    "min" : min,
    "max" : max,
    "lq" : lq,
    "uq" : uq,
    "iqr" : iqr,
    "mode" : mode,
    "fwhm" : fwhm,
    "skewness" : skew,
    "kurtosis" : kurtosis,
    "iqmean" : iqmean,
    "iqstd" : iqstd,
    "n" : n,
    "vol" : n,
    "ndata" : n,
    "voldata" : n,
    "iqn" : iqn,
    "iqvol" : iqn,
}

DEFAULT_STATS = ["mean", "median", "std", "min", "max"]

def run(data, **kwargs):
    
    data_limits = kwargs.get("data_limits", (None, None))

    stats = kwargs.get("stats", DEFAULT_STATS)
    if stats == "all":
        stats = list(STAT_IMPLS.keys())
    if not isinstance(stats, list):
        stats = [stats]
    for s in list(stats):
        if s not in STAT_IMPLS:
            LOG.warn("Unknown statistic: %s - ignoring" % s)
            stats.remove(s)

    if not isinstance(data_limits, (list, tuple)) or len(data_limits) != 2:
        LOG.warn("Invalid data limits: %s - ignoring", data_limits)
        data_limits = (None, None)
    return _get_stats(data, stats, data_limits, voxel_volume=kwargs.get("voxel_volume", 1.0))

def _get_stats(data, stats, data_limits=(None, None), voxel_volume=1.0):
    """
    Get statistics

    :param data: Numpy data to get stats from
    :param stats: List of names of statistics to extract - must be in STATS_IMPLS!
    :param roi: Restrict data to within this roi
    :param data_limits: If specified, min/max values of data to be considered for stats

    :return: Tuple of summary stats dictionary, roi labels
    """
    data_stats = {}
    for s in stats:
        data_stats[s] = 0

    if len(data) == 0:
        return data_stats

    restricted_data = _restrict_data(data, data_limits)
    for s in stats:
        if restricted_data.size == 0:
            value = 0
        elif s in ("n", "vol"):
            # Total volume / number of voxels independent of data restriction
            value = STAT_IMPLS[s](data)
        else:
            value = STAT_IMPLS[s](restricted_data)
        if s in ("vol", "voldata", "iqvol"):
            value *= voxel_volume
        data_stats[s] = value

    return data_stats

def _restrict_data(data, data_limits):
    dmin, dmax = data_limits
    if dmin is not None:
        data = data[data >= dmin]
    if dmax is not None:
        data = data[data <= dmax]
    return data
