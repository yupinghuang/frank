from operator import getitem
import sys


from africanus.coordinates.dask import radec_to_lm
from africanus.rime.dask import phase_delay, predict_vis
from africanus.model.coherency.dask import convert
from astropy.coordinates import SkyCoord
import astropy.units as u
from dask.diagnostics import ProgressBar, Profiler
import dask
import dask.array as da
from dask.graph_manipulation import bind
from daskms import xds_from_ms, xds_from_table, xds_to_table
from casacore.tables import table

import numpy as np

def point_src_with_gain(gain, ms, rms=None):
    #pcenter = SkyCoord(ra = 0. * u.deg, dec=37.129833 * u.deg)
    #src_coord = SkyCoord(ra = 4.02 * u.deg, dec=40.129833 * u.deg)
    #src_lm = radec_to_lm(da.array([[src_coord.ra.to(u.radian).value, src_coord.dec.to(u.radian).value]]),
    #                     phase_centre=da.array([pcenter.ra.to(u.radian).value, pcenter.dec.to(u.radian).value]))
    src_lm = da.array([[0, 0]])
    writes = []
    chan_chunks = 64
    gain2 = da.from_array(gain.reshape(gain.shape[0], gain.shape[1], gain.shape[2], 4), chunks=(1,None,chan_chunks,None))
    time_i = 0
    # new plan: precompute time_inx, group by ANTENNA1.
    for xds in xds_from_ms(ms,
                           columns=["UVW", "ANTENNA1", "ANTENNA2", "TIME"],
                           group_cols=["FIELD_ID", "DATA_DESC_ID", "TIME"],
                           chunks={"row": 1024*2047, "chan": chan_chunks}):
        n_rows = xds.dims['row']
        print(n_rows)
        with table(f'{ms}/SPECTRAL_WINDOW', ack=False) as t:
            freq_arr = da.from_array(t.getcol('CHAN_FREQ')[0])
        src_coh = phase_delay(src_lm, xds.UVW.data, freq_arr)
        if time_i > 5:
            src_coh = bind(src_coh, writes[-5])
        # time_idx = xds.TIME.data.map_blocks(lambda a: np.unique(a, return_inverse=True)[1], dtype=np.int32)
        time_idx = da.ones(n_rows, dtype=np.int32, chunks=(None,)) + time_i
        # TODO fix this chunking nonsense https://blog.dask.org/2021/11/02/choosing-dask-chunk-sizes
        # See dask-ms doc, ROW and TIME chunks must align with vis.
        # gain2 = da.from_array(gain, chunks=(((1,) * (len(time_idx.chunks[0]) - 1) + (20,), # last dim is number of timestamps
        #                         (gain.shape[1],),
        #                         (gain.shape[2],), (gain.shape[3],), (gain.shape[4],))))
        source_coh = (src_coh[...,None] * da.array([1., 0., 0., 1.]))
        print(gain2.chunks)
        print('source_coh', source_coh.chunks)
        vis = predict_vis(time_index=time_idx,
                          antenna1=xds.ANTENNA1.data,
                          antenna2=xds.ANTENNA2.data,
                          die1_jones=gain2[[time_i],...],
                          source_coh=source_coh,
                          die2_jones=gain2[[time_i],...])
        if rms:
            re_noise = bind(da.random.normal(loc=0., scale=rms, size=vis.shape, chunks=vis.chunks), source_coh)
            imag_noise = bind(da.random.normal(loc=0., scale=rms, size=vis.shape, chunks=vis.chunks), source_coh)
            noise = re_noise + (1j * imag_noise)
            vis = vis + noise
        # Assign visibilities to DATA array on the dataset
        xds = xds.assign(DATA=(("row", "chan", "corr"), vis))
        time_i += 1
        write = xds_to_table(xds, ms, ['DATA'])
        writes.append(write)
    da.compute(writes)