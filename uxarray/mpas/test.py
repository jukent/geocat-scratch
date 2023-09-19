from unittest import TestCase

import uxarray as ux
import xarray as xr
import numpy as np

from mpas_source import scrip_from_mpas_xr

class TestGrid(TestCase):
    mpas_path = "data/x1.40962.static.nc"
    mpas_ds = xr.open_dataset(mpas_path)

    expected_dims = ['grid_size', 'grid_corners', 'grid_rank']

    def test_scrip_from_mpas_xr(self):
        scrip_ds = scrip_from_mpas_xr(self.mpas_ds, useLandIceMask=False)

        # check for correct dimensions
        for dim in self.expected_dims:
            assert dim in scrip_ds.dims

    def test_uxarray_scrip_to_ugrid(self):
        scrip_ds = scrip_from_mpas_xr(self.mpas_ds, useLandIceMask=False)
        ugrid = ux.Grid(scrip_ds)

