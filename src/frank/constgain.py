import dftsource
from astropy.coordinates import SkyCoord
from africanus.coordinates import radec_to_lm
import numpy as np
import astropy.units as u

from dask.distributed import Client

from numpy.random import default_rng

if __name__ == '__main__':
    pcenter = SkyCoord(ra='00h00m0.0s', dec='+37d07m47.400s')
    src2 = SkyCoord(ra='00h00m0.0s', dec='+37d37m47.400s')
    def sc_to_rad(sc):
        return [sc.ra.to(u.radian).value, sc.dec.to(u.radian).value]
    src_lm = radec_to_lm(
        np.array([sc_to_rad(pcenter), sc_to_rad(src2)]),
        phase_centre=np.array(sc_to_rad(pcenter)))
    # need shape 2, 20, 2048, 64, 2, 2
    phase_err1 = np.exp(1j * default_rng().normal(size=(2, 2048), scale=5) * np.pi / 180)
    dde_gain = (phase_err1[:,None,:,None,None,None] *
                np.ones(shape=(1, 1, 1, 64, 1, 1)) *
                np.ones(shape=(1, 20, 1, 1, 1, 1)) * 
                np.identity(2))
    np.save('/fastpool/data/W-64chan-30s-dd.npy', dde_gain)
    client = Client(processes=False, local_directory='/fastpool/yuping/tmp')
    dftsource.point_src_with_gain('/fastpool/data/W-64chan-30s.ms',
                                  src_lm,
                                  dd_gain=dde_gain, rms=9.88007)