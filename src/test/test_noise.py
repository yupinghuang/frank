import numpy as np
import pytest

from frank import noise

def test_calc_noise():
    sefd = 5022
    chan_width_hz = 1.3e9 * 0.65
    t_int_s = 900

    assert noise.calc_noise(sefd, chan_width_hz, t_int_s) == pytest.approx(0.00407, rel=1e-3)