import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from africanus.coordinates import radec_to_lm
from numpy.random import Generator, default_rng
from typing import Optional
from scipy.interpolate import griddata, interp2d

def get_pointing_error(error_rms_deg, size, generator: Optional[Generator]=None):
    if not generator:
        generator = default_rng()
    sep_deg, pa_deg = get_pointing_error(0.00166667, 100)
    return sep_deg, pa_deg

# Make the per antenna beam grid (n_ant, n_freq, n_l, n_m) 
def make_beam():
    beam_2_0_data = np.loadtxt(
    '/fastpool/yuping/jonas-beams/Jonas_f=2.00.csv', delimiter=',',
    skiprows=1, dtype={'names': ('phi', 'theta', 'Xamp', 'Xph', 'Yamp', 'Yph'),
                       'formats': ['f'] * 6})
    beam_2_0 = beam_2_0_data['Xamp'] * np.exp(beam_2_0_data['Xph'] * 1j * np.pi/180)

    l, m = np.mgrid[-0.09:0.09:1001j, -0.09:0.09:1001j]
    ldata = np.sin(beam_2_0_data['theta'] * np.pi / 180) * np.cos(beam_2_0_data['phi'] * np.pi / 180)
    mdata = np.sin(beam_2_0_data['theta'] * np.pi / 180) * np.sin(beam_2_0_data['phi'] * np.pi / 180)
    beam_interp = griddata((ldata, mdata), beam_2_0, (l, m))
    sep_deg, pa_deg = get_pointing_error(1/60, 2048)
    pcenter = SkyCoord(ra = 0. * u.deg, dec=37.129833 * u.deg)
    target = pcenter.directional_offset_by(0 * u.deg, 4 * u.deg)
    pointings = target.directional_offset_by(pa_deg * u.deg, sep_deg * u.deg)
    lms = radec_to_lm(np.array([pointings.ra.to(u.radian).value, pointings.dec.to(u.radian).value]).T,
    lms_nominal = radec_to_lm(np.array([[target.ra.to(u.radian).value, target.dec.to(u.radian).value]]),
                  phase_centre=np.array([pcenter.ra.to(u.radian).value, pcenter.dec.to(u.radian).value]))
    phase_centre=np.array([pcenter.ra.to(u.radian).value, pcenter.dec.to(u.radian).value]))
    f = RegularGridInterpolator((l[:,0], m[0]), beam_interp)
    beams_2ghz = f(lms)
    all_gains = np.ones((2048, 8000, 2, 2), dtype=np.complex128)
    beam = np.ones((2048, 8000, 2, 2), dtype=np.complex128)
    for chan in range(8000):
        freq = (chan * 162.5e3) + 0.7e9
        all_gains[:, chan, :, :] = (f(lms * freq/2e9)/f(lms_nominal * freq/2e9))[:,None, None] * np.identity(2)
        beam[:, chan, :, :] = (f(lms * freq/2e9))[:,None, None] * np.identity(2)

def beam_gain(beam_grid: np.array[np.complex64], l: np.array[np.float], m: np.array[np.float]) -> np.array[np.complex64]:
    """
    Calculate the beam gain for a given time series of l and m."""
    return beam_grid[:, :, l, m]