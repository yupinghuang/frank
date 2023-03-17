import numpy as np

def calc_noise(sefd:float, chan_width_hz: float, t_int_s: float) -> float:
    """Calculate the per visibility visibility rms for identical antennas.
    Args:
        sefd (float): System Equivalent Flux Density (SEFD) per antennas in Jy.
        chan_width_khz (float): Channel width in Hz.
        t_int_s (float): Accumulation time in seconds.
    Returns:
        float: noise standard devation per part visibility.
    """
    return sefd / np.sqrt(2 * chan_width_hz * t_int_s)