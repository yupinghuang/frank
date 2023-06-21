from operator import getitem
import sys


from africanus.coordinates.dask import radec_to_lm
from africanus.rime.dask import phase_delay, predict_vis
from africanus.model.coherency.dask import convert
from astropy.coordinates import SkyCoord
import astropy.units as u
from dask.diagnostics import ProgressBar, Profiler
import dask.array as da
from dask.graph_manipulation import bind
from daskms import xds_from_ms, xds_from_table, xds_to_table
from casacore.tables import table
from africanus.rime.phase import phase_delay as np_phase_delay

import numpy as np

def point_src_with_gain(ms, src_lm, di_gain=None, dd_gain=None, rms=None, client=None):
    #pcenter = SkyCoord(ra = 0. * u.deg, dec=37.129833 * u.deg)
    #src_coord = SkyCoord(ra = 4.02 * u.deg, dec=40.129833 * u.deg)
    #src_lm = radec_to_lm(da.array([[src_coord.ra.to(u.radian).value, src_coord.dec.to(u.radian).value]]),
    #                     phase_centre=da.array([pcenter.ra.to(u.radian).value, pcenter.dec.to(u.radian).value]))
    writes = []
    chan_chunks = 64
    gain2 = None
    dd_gain2 = None
    if di_gain is not None:
        gain2 = da.from_array(di_gain.reshape(di_gain.shape[0],
                                            di_gain.shape[1],
                                            di_gain.shape[2], 4),
                                            chunks=(1,None,chan_chunks,None))
    if dd_gain is not None:
        dd_gain2 = da.from_array(dd_gain.reshape(dd_gain.shape[0],
                                            dd_gain.shape[1],
                                            dd_gain.shape[2],
                                            dd_gain.shape[3], 4),
                                            chunks=(1,1,None,chan_chunks,None))
    src_lm = da.from_array(src_lm, chunks=(1,None))
    with table(f'{ms}/SPECTRAL_WINDOW', ack=False) as t:
        freq_arr = da.from_array(t.getcol('CHAN_FREQ')[0], chunks=(chan_chunks,))
    seq = np.random.SeedSequence()
    for xds in xds_from_ms(ms,
                           columns=["UVW", "ANTENNA1", "ANTENNA2", "TIME"],
                           group_cols=["FIELD_ID", "DATA_DESC_ID", "ANTENNA1"],
                           chunks={"row": 40940, "chan": chan_chunks}):
        n_rows = xds.dims["row"]
        xds = xds.chunk({"row": n_rows // 20})
        src_coh = phase_delay_diag(src_lm, xds.UVW.data, freq_arr)
        time_idx = xds.TIME.data.map_blocks(lambda a: np.unique(a, return_inverse=True)[1], dtype=np.int32)
        vis = predict_vis(time_index=time_idx,
                          antenna1=da.broadcast_to(xds.ANTENNA1, xds.ANTENNA2.shape),
                          antenna2=xds.ANTENNA2.data,
                          die1_jones=gain2,
                          source_coh=src_coh,
                          die2_jones=gain2,
                          dde1_jones=dd_gain2,
                          dde2_jones=dd_gain2)
        if rms:
            vis = da.map_blocks(add_noise, vis, rms, seq, dtype=np.complex64, meta=np.array((), dtype=np.complex64))
        # Assign visibilities to DATA array on the dataset
        xds = xds.assign(DATA=(("row", "chan", "corr"), vis))
        write = xds_to_table(xds, ms, ['DATA'])
        writes.append(write)
    np.random.default_rng().shuffle(writes)
    da.compute(writes)

def add_noise(blk, rms, seq):
    rng = np.random.default_rng(seq.spawn(1)[0].generate_state(1))
    return blk + rng.normal(loc=0., scale=rms, size=blk.shape) + 1j * rng.normal(loc=0., scale=rms, size=blk.shape)

def phase_delay_diag(lm, uvw, frequency, convention="casa"):
    """ Dask wrapper for phase_delay function """
    return da.core.blockwise(
        _phase_delay_wrap_diag,
        ("source", "row", "chan", "corr"),
        lm,
        ("source", "(l,m)"),
        uvw,
        ("row", "(u,v,w)"),
        frequency,
        ("chan",),
        convention=convention,
        new_axes={"corr": 4},
        dtype=np.complex64,
    )

def _phase_delay_wrap_diag(lm, uvw, frequency, convention):
    phases = np_phase_delay(lm[0], uvw[0], frequency, convention=convention)
    zeros = np.broadcast_to(0, phases.shape)
    ans = np.stack([phases, zeros, zeros, phases], axis=-1)
    phases = None
    zeros = None
    return ans