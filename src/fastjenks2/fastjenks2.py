from typing import Union, List
import numpy as np
from collections import namedtuple

class FastJenks2:
    """
    fast split a one dim array by jenks optimazation (for k=2)
    This implementation checks all possible combinations, don't use it on big arrays
    """
    FastJenks2Data = namedtuple('FastJenks2Data', ['sdcm', 'sdam', 'gvf', 'group_array'])
    @staticmethod
    def jenks(a: Union[np.ndarray, List[float]]):
        a = np.array(a)
        assert len(a.shape) == 1 or (len(a.shape) == 2 and a.shape[1] == 1)
        factor = np.arange(1, a.shape[0] + 1)
        rfactor = factor[::-1]
        sa = np.sort(a)
        left = sa.cumsum()
        right = sa[::-1].cumsum()[::-1]
        left_mean = left / factor
        right_mean = right / rfactor
        mean_vals = np.dstack((left_mean[:-1], right_mean[1:])).flatten()
        mean_repetition = np.dstack((factor[:-1], rfactor[1:])).flatten()
        mean_matrix = np.repeat(mean_vals, mean_repetition).reshape(-1, sa.shape[0])
        squared_deviation = (mean_matrix - sa) ** 2
        sdcm_all = squared_deviation.sum(axis=1)
        sdcm_argmin = sdcm_all.argmin()
        sdcm = sdcm_all.min()
        sdcm_0_last_val = sa[sdcm_argmin]
        sdam = ((sa - sa.mean()) ** 2).sum()
        group_array = a > sdcm_0_last_val
        return FastJenks2.FastJenks2Data(sdcm, sdam, (sdam - sdcm) / sdam, group_array)
