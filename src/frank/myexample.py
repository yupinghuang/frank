from casacore.tables import table
import numpy as np
import logging

import idg.util as util
import idg

MAX_UV_M = 15030.
GRID_PADDING = 1.4
SPEED_OF_LIGHT = 299792458.0

logging.basicConfig(
    filename='myexample.log',
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

def compute_image_size(grid_size, freq):
    half_grid_size = grid_size / (2* GRID_PADDING)
    return half_grid_size / MAX_UV_M * (SPEED_OF_LIGHT / freq)    

def gridding(
        p, w_step, shift, cell_size, kernel_size, subgrid_size, frequencies, visibilities,
        uvw, baselines, grid, aterms, aterms_offsets, spheroidal):
    p.gridding(
        kernel_size, frequencies, visibilities, uvw, baselines,
        aterms, aterms_offsets, spheroidal)
    # p.get_grid(grid)
    logging.info('End gridding.')
    p.transform(idg.FourierDomainToImageDomain)
    p.get_final_grid(grid)
    print(grid.shape)
    np.save('/fastpool/data/grid_post_fft.npy', grid)


def main():
    nr_channels = 1
    nr_timesteps = 600

    with table('/fastpool/data/20210226M-2GHz-1chan-600int.ms') as t:
        t_ant = table(t.getkeyword("ANTENNA"))
        t_spw = table(t.getkeyword("SPECTRAL_WINDOW"))
        frequencies = np.asarray(t_spw[0]['CHAN_FREQ'], dtype=idg.frequenciestype)
        nr_stations = len(t_ant)
        nr_baselines = (nr_stations * (nr_stations - 1)) // 2
        # Number of aterms to generate
        ant1 = t.getcol('ANTENNA1', nrow=nr_baselines)
        ant2 = t.getcol('ANTENNA2', nrow=nr_baselines)
        uvw_orig = t.getcol('UVW', nrow=nr_baselines*nr_timesteps)

    print(f'max uvw length is ',
            np.max(np.sqrt(uvw_orig[:,0]**2 + uvw_orig[:,1]**2 + uvw_orig[:,2]**2)))
    nr_timeslots = 1
    nr_correlations = 4
    grid_size = 8192
    # image_size = 0.08
    image_size = compute_image_size(grid_size, frequencies.max())
    print('Image size is ', image_size)
    subgrid_size = 32
    kernel_size = 22
    cell_size = image_size / grid_size

    uvw_orig = uvw_orig.reshape((nr_timesteps, nr_baselines, 3))
    """
    vis = vis.reshape((nr_timesteps, nr_baselines, nr_channels, nr_correlations))


    vis = np.ascontiguousarray(np.swapaxes(vis, 0, 1).astype(idg.visibilitiestype))
    """
    vis = np.ones((nr_baselines, nr_timesteps, nr_channels, nr_correlations),
            dtype=idg.visibilitiestype)

    logging.info('Done with data generation.')
    baselines = np.zeros(shape=(nr_baselines), dtype=idg.baselinetype)
    baselines['station1'] = ant1[0]
    baselines['station2'] = ant2[0]

    uvw = np.zeros(shape=(nr_baselines, nr_timesteps), dtype=idg.uvwtype)
    uvw_orig = np.swapaxes(uvw_orig,0, 1)
    uvw['u'] = uvw_orig[:, :, 0 ]
    uvw['v'] = uvw_orig[:, :, 1]
    uvw['w'] = uvw_orig[:, :, 2]

    shift = np.zeros(3, dtype=np.float32)
    w_step = 0.0

    # p = idg.CUDA.Unified()
    p = idg.HybridCUDA.GenericOptimized()

    spheroidal     = util.get_example_spheroidal(subgrid_size)
    aterms         = util.get_identity_aterms(
                        nr_timeslots, nr_stations, subgrid_size, nr_correlations)
    aterms_offsets = util.get_example_aterms_offset(nr_timeslots, nr_timesteps)
    aterms_offsets[1] = 0
    grid = p.allocate_grid(nr_correlations, grid_size)
    ######################################################################
    # initialize data
    ######################################################################
    p.init_cache(subgrid_size, cell_size, w_step, shift)
    logging.info('Start gridding.')
    gridding(p, w_step, shift, cell_size, kernel_size, subgrid_size, frequencies, vis,
            uvw, baselines, grid, aterms, aterms_offsets, spheroidal)

if __name__=='__main__':
    main()
