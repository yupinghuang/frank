from operator import getitem


from africanus.coordinates.dask import radec_to_lm
from africanus.rime.dask import wsclean_predict, predict_vis
from astropy.coordinates import SkyCoord
import astropy.units as u
from dask.diagnostics import ProgressBar
import dask
import dask.array as da
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
    for xds in xds_from_ms(ms,
                           columns=["UVW", "ANTENNA1", "ANTENNA2", "TIME"],
                           group_cols=["FIELD_ID", "DATA_DESC_ID"],
                           chunks={"row": 2048}):
        n_rows = xds.dims['row']
        print(n_rows)
        with table(f'{ms}/SPECTRAL_WINDOW', ack=False) as t:
            freq_arr = da.from_array(t.getcol('CHAN_FREQ')[0])
        src_coh = wsclean_predict(xds['UVW'], src_lm, da.array(['POINT']), da.array([1.0]),
                        da.array([[0]]), da.array([True]), da.array([1.35e9]), da.array([[0,0,0]]), freq_arr)
        time_idx = xds.TIME.data.map_blocks(lambda a: np.unique(a, return_inverse=True)[1], dtype=np.int32)
        # TODO fix this chunking nonsense https://blog.dask.org/2021/11/02/choosing-dask-chunk-sizes
        # See dask-ms doc, ROW and TIME chunks must align with vis.
        gain2 = da.from_array(gain, chunks=(((0,) * (len(time_idx.chunks[0]) - 1) + (20,), # last dim is number of timestamps
                                 (gain.shape[1],),
                                 (gain.shape[2],), (gain.shape[3],), (gain.shape[4],))))
        vis = predict_vis(time_index=time_idx,
                          antenna1=xds.ANTENNA1.data,
                          antenna2=xds.ANTENNA2.data,
                          die1_jones=gain2,
                          source_coh=src_coh[None, :, :, :, None] * da.array([[1., 0.],[0., 1.]]),
                          die2_jones=gain2)
        if rms:
            vis = vis + (da.random.normal(loc=0., scale=rms, size=vis.shape, chunks=vis.chunks) +
                         1j * da.random.normal(loc=0., scale=rms, size=vis.shape, chunks=vis.chunks))
        vis = vis.reshape(vis.shape[:2] + (4,))
        # Assign visibilities to DATA array on the dataset
        xds = xds.assign(DATA=(("row", "chan", "corr"), vis))
        writes.append(xds_to_table(xds, ms, ['DATA']))
    with ProgressBar():
        dask.compute(writes)
