import dftsource
import sys
from astropy.coordinates import SkyCoord
from africanus.coordinates import radec_to_lm
import numpy as np
import astropy.units as u

from dask.distributed import Client

from numpy.random import default_rng

from iono import h5parm_to_phases

def compute_dde_gain(h5parm_path):
    pcenter = SkyCoord(ra='00h00m0.0s', dec='+37d07m47.400s')
    src1 = SkyCoord(ra='00h00m0.0s', dec='+37d07m47.400s')
    src2 = SkyCoord(ra='00h00m0.0s', dec='+37d37m47.400s')
    def sc_to_rad(sc):
        return [sc.ra.to(u.radian).value, sc.dec.to(u.radian).value]
    _, phases = h5parm_to_phases(h5parm_path) 
    src_lm = radec_to_lm(
        np.array([sc_to_rad(src1), sc_to_rad(src2)]),
        phase_centre=np.array(sc_to_rad(pcenter)))
    src_lm = src_lm[[0]]
    phases = phases[...,:-1].swapaxes(1, 2)
    freq_arr = np.arange(64) * 134e3 + 0.7e9
    dde_gain = (np.exp(1j * phases[..., None] * (0.7e9/freq_arr))[..., None, None]
                * np.identity(2))
    return dde_gain

def compute_lms(sources, pcenter):
    def sc_to_rad(sc):
        return [sc.ra.to(u.radian).value, sc.dec.to(u.radian).value]
    src_lm = radec_to_lm(
        np.array([sc_to_rad(src) for src in sources]),
        phase_centre=np.array(sc_to_rad(pcenter)))
    return src_lm

if __name__ == '__main__':
    pcenter = SkyCoord(ra='00h00m0.0s', dec='+37d07m47.400s')
    src1 = SkyCoord(ra='00h00m0.0s', dec='+37d07m47.400s')
    src2 = SkyCoord(ra='00h00m0.0s', dec='+37d37m47.400s')
    src_lm = compute_lms([src1, src2], pcenter)
    src_lm = src_lm[[0]]
    print(src_lm)
    # shape 2,2048, 21
    dde_gain = compute_dde_gain('/safepool/yuping/sim_dsa2000W_light_dawn_30.0_1.5_1src.h5')
    np.save('/fastpool/data/W-64chan-30s-dd.npy', dde_gain)
    client = Client(processes=False, local_directory='/fastpool/yuping/tmp')
    dftsource.point_src_with_gain('/fastpool/data/W-64chan-30s.ms',
                                  src_lm,
                                  dd_gain=dde_gain, rms=9.88007, client=client)
