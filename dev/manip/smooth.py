import warnings
import numpy as np
import scipy.signal as signal

from ..decorate import list_generalizer


@list_generalizer
def smooth(traj, kernel_width=100, n=500, order=3):
    if traj is None or traj.shape[0] <= 3:
        warnings.warn(f'could not smooth trajectory of size {traj.shape}')
        return None

    # noinspection PyBroadException
    try:
        r = np.zeros([n, traj.shape[1]])
        x = traj.index.values
        xx = np.linspace(np.min(x), np.max(x), num=n)

        for i in range(traj.shape[1]):
            r[:, i] = signal.savgol_filter(sp.interpolate.pchip(x, traj.values[:, i])(xx),
                                           kernel_width, order)
            r[:, i][r[:, i] < min_val] = min_val

        return pd.DataFrame(data=r, index=xx, columns=traj.columns)
    except:
        warnings.warn(f'could not smooth trajectory: {str(traj)[:100]}...')
        return None
