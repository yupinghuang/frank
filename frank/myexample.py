from casacore.tables import table
import numpy as np

import idg.util as util
import idg

def gridding(
        p, w_step, shift, cell_size, kernel_size, subgrid_size, frequencies, visibilities,
        uvw, baselines, grid, aterms, aterms_offsets, spheroidal):
    p.gridding(
        kernel_size, frequencies, visibilities, uvw, baselines,
        aterms, aterms_offsets, spheroidal)
    p.get_grid(grid)
    p.transform(idg.FourierDomainToImageDomain)
    p.get_grid(grid)

nr_channels = 1
nr_timesteps = 1

with table('/fastpool/data/20210226M-2GHz-1chan-600int.ms') as t:
    t_ant = table(t.getkeyword("ANTENNA"))
    t_spw = table(t.getkeyword("SPECTRAL_WINDOW"))
    frequencies = np.asarray(t_spw[0]['CHAN_FREQ'], dtype=idg.frequenciestype)
    nr_stations = len(t_ant)
    nr_baselines = (nr_stations * (nr_stations - 1)) // 2
    # Number of aterms to generate
    vis = t.getcol('DATA', nrow=nr_baselines*nr_channels)
    ant1 = t.getcol('ANTENNA1', nrow=nr_baselines * nr_channels)
    ant2 = t.getcol('ANTENNA2', nrow=nr_baselines * nr_channels)
    uvw = t.getcol('UVW', nrow=nr_baselines*nr_channels)

vis[ant1 == ant2] = 0

nr_timeslots = 1
nr_correlations = 4
grid_size = 4096
image_size = 2048
subgrid_size = 32
kernel_size = 16
cell_size = image_size / grid_size

ant1 = ant1.reshape((nr_timesteps, nr_baselines))
ant2 = ant2.reshape((nr_timesteps, nr_baselines))
uvw_orig = uvw.reshape((nr_timesteps, nr_baselines, 3))
vis = vis.reshape((nr_timesteps, nr_baselines, nr_channels, nr_correlations))


vis = np.ascontiguousarray(np.swapaxes(vis, 0, 1).astype(idg.visibilitiestype))

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

proxy = idg.HybridCUDA.GenericOptimized()

spheroidal     = util.get_identity_spheroidal(subgrid_size)
aterms         = util.get_identity_aterms(
                    nr_timeslots, nr_stations, subgrid_size, nr_correlations)
aterms_offsets = util.get_example_aterms_offset(
                    nr_timeslots, nr_timesteps)

grid = proxy.allocate_grid(nr_correlations, grid_size)
proxy.init_cache(subgrid_size, cell_size, w_step, shift)

gridding(proxy, w_step, shift, cell_size, kernel_size, subgrid_size, frequencies, vis,
        uvw, baselines, grid, aterms, aterms_offsets, spheroidal)
