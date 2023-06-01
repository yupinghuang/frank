import matplotlib.pyplot as plt
from dask.distributed import Client
import astropy.units as u
import astropy
from astropy.coordinates import SkyCoord
from astropy.coordinates import SkyOffsetFrame, ICRS, ITRS
import dftsource
import numpy as np
from typing import Tuple
from scipy.interpolate import griddata

from h5parm import DataPack
from enu import ENU

def find_center(ants, times_grid):
    ref_ant = ants[0]
    f = ENU(location=ref_ant.earth_location, obstime=times_grid[0])
    ants_enu = ITRS(*ants.cartesian.xyz, obstime=times_grid[0]).transform_to(f)
    mean_x = np.median(ants_enu.east.value)
    mean_y = np.median(ants_enu.north.value)
    print(mean_x, mean_y)
    ref_ind = np.argmin((ants_enu.east.value - mean_x)**2 + (ants_enu.north.value - mean_y)**2)
    print(ref_ind)
    return ref_ind

def h5parm_to_np(h5parm: str) -> Tuple[np.ndarray, astropy.time.Time, astropy.coordinates.ITRS]:
    with DataPack(h5parm, readonly=True) as dp:
        print("Axes order:", dp.axes_order)
        dp.current_solset = 'sol000'
        dp.select()
        phase, axes = dp.phase
        phase = phase[0,:,:,0,:]
        antenna_labels, antennas = dp.get_antennas(axes['ant'])
        timestamps, times = dp.get_times(axes['time'])
        freqs = dp.get_freqs(axes['freq'])
    return phase, times, freqs, antennas

def h5parm_to_phases(fn):
    phase_grid, times, freqs, antennas_grid = h5parm_to_np(fn)
    actual_ants = np.loadtxt('/home/yuping/frank/20210326-configs/20210226W.txt', usecols=(0,1))
    phase_center = SkyCoord(ra='00h00m0.0s', dec='+37d07m47.400s')
    ref_ind = find_center(antennas_grid, times)
    frame = ENU(location=antennas_grid[ref_ind].earth_location, obstime=times[0])
    antennas_grid_enu = ITRS(*antennas_grid.cartesian.xyz, obstime=times[0]).transform_to(frame)
    print(phase_grid.shape)
    phase_gain = griddata((antennas_grid_enu.east.value+(np.mean(actual_ants[:,0])-np.mean(antennas_grid_enu.east.value)),
                    antennas_grid_enu.north.value+(np.mean(actual_ants[:,1]) - np.mean(antennas_grid_enu.north.value))),
                    np.swapaxes(phase_grid, 0, 1),
                   actual_ants, method='linear')
    return phase_grid, phase_gain

if __name__=='__main__':
    phase_grid, phase_gains = h5parm_to_phases('/safepool/yuping/sim_dsa2000W_1000m_grid_dawn_30.0_1.5.h5')
    freq_arr = np.arange(64) * 134e3 + 0.7e9
    # need shape 20, 2048, 64, 2, 2
    """
    phase_gains2 = np.exp(-1j * (phase_gains[:,0,:-1].T)[:,:, None, None, None] *
                (0.7e9/freq_arr)[None, None, :, None, None]) * np.identity(2)
                """
    phase_gains2 = (np.arange(20)[:, None, None, None, None] *
                    np.ones(2048)[None, :, None, None, None] *
                    np.exp(1j * np.arange(64) * np.pi/180)[None, None, :, None, None]
                    * np.identity(2))
    phase_gains2[:,1238,:,0,0] = 1
    np.save('/fastpool/data/W-64chan-30s.npy', phase_gains2)
    client = Client(processes=False, local_directory='/fastpool/yuping/tmp')
    dftsource.point_src_with_gain(phase_gains2, '/fastpool/data/W-64chan-30s.ms', rms=1e-6)