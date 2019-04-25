import os
import pytest
import numpy as np
import numpy.testing as nt

import xgeo
from xgeo.crs import XCRS
import xarray as xr

here = os.path.dirname(__file__)
datapath = os.path.join(here, "data")

zones_shp = os.path.join(datapath, "zones.shp")
zones_geojson = os.path.join(datapath, "zones.geojson")


@pytest.fixture
def data_nc():
    return xr.open_dataset(os.path.join(datapath, "data.nc"))


@pytest.fixture
def data_tif():
    da = xr.open_rasterio(os.path.join(datapath, "data.tif"))
    return da.to_dataset(name="data")


@pytest.fixture
def proj_data_nc():
    return xr.open_dataset(os.path.join(datapath, "proj_data.nc"))


@pytest.fixture
def proj_data_tif():
    da = xr.open_rasterio(os.path.join(datapath, "proj_data.tif"))
    return da.to_dataset(name="data")


@pytest.fixture
def data_flipped_nc():
    return xr.open_dataset(os.path.join(datapath, "data_flipped.nc"))


@pytest.fixture
def proj_data_flipped_nc():
    return xr.open_dataset(os.path.join(datapath, "proj_data_flipped.nc"))


def test_reprojection(data_nc, data_tif, proj_data_nc, proj_data_tif, data_flipped_nc, proj_data_flipped_nc):
    def test_ds_equal(s, t):
        nt.assert_equal(s.geo.x_coords.values, t.geo.x_coords.values)
        nt.assert_equal(s.geo.y_coords.values, t.geo.y_coords.values)
        nt.assert_equal(s.data.values, t.data.values)

    def test_ds_alomst_equal(s, t):
        nt.assert_almost_equal(s.geo.x_coords.values, t.geo.x_coords.values)
        nt.assert_almost_equal(s.geo.y_coords.values, t.geo.y_coords.values)
        nt.assert_equal(s.data.values, t.data.values)

    netcdf_dsout = data_nc.geo.reproject(target_crs="EPSG:4326")
    gtiff_dsout = data_tif.geo.reproject(target_crs=4326)
    flipped_dsout = data_flipped_nc.geo.reproject(target_crs=4326)
    test_ds_equal(netcdf_dsout, gtiff_dsout)

    # Because of flipped y coordinate the data could be negligibly shifted
    test_ds_alomst_equal(netcdf_dsout, flipped_dsout)

    # Coordinates can have negligible amount of differences
    test_ds_alomst_equal(netcdf_dsout, proj_data_nc)
    test_ds_alomst_equal(gtiff_dsout, proj_data_tif)
    test_ds_alomst_equal(flipped_dsout, proj_data_flipped_nc)


def test_projection_system(data_nc, data_tif):
    netcdf_crs = XCRS.from_any(data_nc.geo.projection)
    gtiff_crs = XCRS.from_any(data_tif.geo.projection)
    assert gtiff_crs == netcdf_crs
    assert gtiff_crs == XCRS.from_epsg(32737)


def test_transform(data_nc, data_tif, data_flipped_nc):
    netcdf_transform = data_nc.geo.transform
    gtiff_transform = data_tif.geo.transform
    nt.assert_equal(netcdf_transform, gtiff_transform)

    nt.assert_almost_equal(data_nc.geo.bounds, data_flipped_nc.geo.bounds, decimal=5)

    # Setting transform
    data_nc.geo.transform = (1, 0, 0, 0, 1, 0)
    nt.assert_equal(data_nc.geo.x_coords.values, np.arange(0.5, data_nc.geo.x_size + 0.5))
    nt.assert_equal(data_nc.geo.y_coords.values, np.arange(0.5, data_nc.geo.y_size + 0.5))


def test_origin(data_nc):
    neds = data_nc.copy(deep=True)

    neds.geo.origin = 'bottom_right'
    nt.assert_equal(data_nc.geo.x_coords.values, neds.geo.x_coords.values[::-1])
    nt.assert_equal(data_nc.geo.y_coords.values, neds.geo.y_coords.values[::-1])
    nt.assert_equal(data_nc.data.values, neds.data.loc[{'x': data_nc.geo.x_coords,
                                                        'y': data_nc.geo.y_coords}].values)

    neds.geo.origin = 'top_left'
    nt.assert_equal(data_nc.geo.x_coords.values, neds.geo.x_coords.values)
    nt.assert_equal(data_nc.geo.y_coords.values, neds.geo.y_coords.values)
    nt.assert_equal(data_nc.data.values, neds.data.loc[{'x': data_nc.geo.x_coords,
                                                        'y': data_nc.geo.y_coords}])

    neds.geo.origin = 'bottom_left'
    nt.assert_equal(data_nc.geo.x_coords.values, neds.geo.x_coords.values)
    nt.assert_equal(data_nc.geo.y_coords.values, neds.geo.y_coords.values[::-1])
    nt.assert_equal(data_nc.data.values, neds.data.loc[{'x': data_nc.geo.x_coords,
                                                        'y': data_nc.geo.y_coords}])

    neds.geo.origin = 'top_right'
    nt.assert_equal(data_nc.geo.x_coords.values, neds.geo.x_coords.values[::-1])
    nt.assert_equal(data_nc.geo.y_coords.values, neds.geo.y_coords.values)
    nt.assert_equal(data_nc.data.values, neds.data.loc[{'x': data_nc.geo.x_coords,
                                                          'y': data_nc.geo.y_coords}])


def test_zonal_statistics(data_nc, data_tif):
    nt.assert_equal(data_nc.geo.zonal_stats(zones_shp, value_name='id').values,
                    data_tif.geo.zonal_stats(zones_geojson, value_name='id').values)
    nt.assert_equal(data_nc.geo.zonal_stats(zones_geojson, value_name='class').values,
                    data_tif.geo.zonal_stats(zones_shp, value_name='class').values)



