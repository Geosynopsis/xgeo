from pathlib import Path
import numpy.testing as nt
import xarray.testing as xt
import numpy as np
import pytest

import xarray as xr
import xgeo
from xgeo.crs import XCRS

here = Path(__file__).parent
datapath = here / "data"

zones_shp = datapath / "zones.shp"
zones_geojson = datapath / "zones.geojson"


@pytest.fixture
def netcdf_ds():
    return xr.open_dataset(datapath / "data.nc")


@pytest.fixture
def netcdf_qgis_ds():
    return xr.open_dataset(datapath / "netcdf_qgis.nc")


@pytest.fixture
def geotiff_da():
    return xr.open_rasterio(datapath / "data.tif")


@pytest.fixture
def projected_netcdf_ds():
    return xr.open_dataset(datapath / "proj_data.nc")


@pytest.fixture
def projected_netcdf_qgis_ds():
    return xr.open_dataset(datapath / "projected_netcdf_qgis.nc")


@pytest.fixture
def projected_geotiff_da():
    return xr.open_rasterio(datapath / "proj_data.tif")


@pytest.fixture
def netcdf_flipped_ds():
    return xr.open_dataset(datapath / "data_flipped.nc")


@pytest.fixture
def projected_netcdf_flipped_ds():
    return xr.open_dataset(datapath / "proj_data_flipped.nc")


@pytest.fixture
def zonal_stats_class_da():
    return xr.open_dataarray(datapath / "zonal_stats_class_da.nc")


@pytest.fixture
def zonal_stats_class_ds():
    return xr.open_dataset(datapath / "zonal_stats_class_ds.nc")


@pytest.fixture
def zonal_stats_id_da():
    return xr.open_dataarray(datapath / "zonal_stats_id_da.nc")


@pytest.fixture
def zonal_stats_id_ds():
    return xr.open_dataset(datapath / "zonal_stats_id_ds.nc")


@pytest.fixture
def subset_da():
    return xr.open_rasterio(datapath / "subset_data.tif")


@pytest.fixture(params=[
    ('netcdf_ds', "projected_netcdf_ds"),
    ('geotiff_da', "projected_geotiff_da"),
    ('netcdf_flipped_ds', "projected_netcdf_flipped_ds"),
    # ("netcdf_qgis_ds", "projected_netcdf_qgis_ds")
])
def projection_test_data(request):
    return map(request.getfixturevalue, request.param)


def test_reprojection(projection_test_data):
    original_data, projected_data = projection_test_data
    xrobj = original_data.geo.reproject(target_crs="EPSG:4326")

    # The reporjection could introduce some small offsets on the coordinates.
    # Therefore, we check if the coordinates are almost equal to the
    # coordinates of the reference data.
    xt.assert_allclose(
        xrobj.geo.x_coords,
        projected_data.geo.x_coords
    )
    xt.assert_allclose(
        xrobj.geo.y_coords,
        projected_data.geo.y_coords
    )
    if isinstance(xrobj, xr.DataArray):
        nt.assert_equal(
            xrobj.values,
            projected_data.values
        )
    else:
        for var_key in xrobj.data_vars:
            nt.assert_equal(
                xrobj[var_key].values,
                projected_data[var_key].values
            )


def test_projection_system(netcdf_ds, geotiff_da):
    netcdf_crs = XCRS.from_any(netcdf_ds.geo.projection)
    gtiff_crs = XCRS.from_any(geotiff_da.geo.projection)
    assert gtiff_crs == netcdf_crs
    assert gtiff_crs == XCRS.from_epsg(32737)


def test_transform(netcdf_ds, geotiff_da, netcdf_flipped_ds):
    netcdf_transform = netcdf_ds.geo.transform
    gtiff_transform = geotiff_da.geo.transform

    nt.assert_almost_equal(netcdf_transform, gtiff_transform)

    nt.assert_almost_equal(
        netcdf_ds.geo.bounds,
        netcdf_flipped_ds.geo.bounds, decimal=5
    )

    # Setting transform
    netcdf_ds.geo.transform = (1, 0, 0, 0, 1, 0)

    nt.assert_equal(
        netcdf_ds.geo.x_coords.values,
        np.arange(0.5, netcdf_ds.geo.x_size + 0.5)
    )

    nt.assert_equal(
        netcdf_ds.geo.y_coords.values,
        np.arange(0.5, netcdf_ds.geo.y_size + 0.5)
    )


@pytest.fixture(params=["netcdf_ds", "geotiff_da"])
def origin_test_data(request):
    return request.getfixturevalue(request.param)


@pytest.mark.parametrize("origin", [
    'bottom_right', 'top_left', 'bottom_left', 'top_right'
])
def test_origin(origin_test_data, origin):
    yo, xo = origin_test_data.geo.origin.split('_')
    tyo, txo = origin.split('_')

    neds = origin_test_data.copy(deep=True)
    neds.geo.origin = origin
    if xo == txo:
        nt.assert_equal(
            origin_test_data.geo.x_coords.values,
            neds.geo.x_coords.values
        )
    else:
        nt.assert_equal(
            origin_test_data.geo.x_coords.values,
            neds.geo.x_coords.values[::-1]
        )

    if yo == tyo:
        nt.assert_equal(
            origin_test_data.geo.y_coords.values,
            neds.geo.y_coords.values
        )
    else:
        nt.assert_equal(
            origin_test_data.geo.y_coords.values,
            neds.geo.y_coords.values[::-1]
        )


@pytest.fixture(params=[
    ('netcdf_ds', 'id', 'zonal_stats_id_ds'),
    ('netcdf_ds', 'class', 'zonal_stats_class_ds'),
    ('geotiff_da', 'id', 'zonal_stats_id_da'),
    ('geotiff_da', 'class', 'zonal_stats_class_da'),
])
def zonal_stat_test_data(request):
    def getvalue(p):
        try:
            return request.getfixturevalue(p)
        except BaseException:
            return p
    return map(getvalue, request.param)


def test_zonal_statistics(zonal_stat_test_data):
    data, value_name, reference = zonal_stat_test_data
    xt.assert_equal(
        data.geo.zonal_stats(zones_shp, value_name=value_name),
        reference
    )


def test_geotiff(netcdf_ds, geotiff_da):
    def assert_equal(result_file, reference):
        result = xr.open_rasterio(result_file)
        xt.assert_allclose(
            result.geo.x_coords,
            reference.geo.x_coords
        )
        xt.assert_allclose(
            result.geo.y_coords,
            reference.geo.y_coords
        )
        nt.assert_equal(
            result.values,
            reference.values
        )

    filepath = netcdf_ds.data.geo.to_geotiff(
        output_path=datapath,
        prefix='netcdf2gtiff',
        dims={'time': 0}
    )

    filepaths = netcdf_ds.geo.to_geotiff(
        output_path=datapath,
        prefix='netcdf2gtiff',
        dims={'time': 0}
    )

    assert_equal(filepath, geotiff_da)
    Path(filepath).unlink()
    for _, filepath in filepaths.items():
        assert_equal(filepath, geotiff_da)
        Path(filepath).unlink()


def test_subset(geotiff_da, subset_da):
    result = geotiff_da.geo.subset(zones_geojson, crop=True)
    nt.assert_allclose(
        result.geo.x_coords,
        subset_da.geo.x_coords
    )
    nt.assert_allclose(
        result.geo.y_coords,
        subset_da.geo.y_coords
    )
    nt.assert_equal(
        result.values,
        subset_da.values
    )
