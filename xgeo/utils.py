from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio.warp as warp
import xarray as xr

X_DIMS = ['x', 'lon', 'longitude', 'longitudes', 'lons', 'eastings', 'easting']
Y_DIMS = ['y', 'lat', 'latitude', 'latitudes', 'lats', 'northings', 'northing']
Z_DIMS = ['z', 'bands', 'band']
T_DIMS = ['time', 'times', 'xtime']

DEFAULT_DIMS = dict(
    time_dim='time',
    band_dim='band',
    x_dim='x',
    y_dim='y',
    z_dim='z'
)


def get_geodataframe(vector_file, projection, geometry_field="geometry"):
    """
    Gets GeoDataFrame for given vector file. It carries out following
    list of tasks.
        - Sets the geometry from given geometry_field
        - Reprojects the dataframe to the CRS system of the Dataset
        - Checks whether the Dataframe is empty and whether any geometry
        lies within the bound of the dataset.
    Parameters
    ----------
    vector_file: str or geopandas.GeoDataFrame
        Vector file to be read

    geometry_field: str
        Name of the geometry in the vector file if it doesn't default to
        "geometry"

    Returns
    -------
    vector_gdf: geopandas.GeoDataFrame
        GeoDataFrame for the given vector file
    """
    try:
        import fiona
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "`fiona` module is needed to use this function. Please \
                install it and try again"
        )

    # Read and reproject the vector file
    assert isinstance(vector_file, (str, Path)), "Invalid vector_file. The \
             `vector_file` should be path or string"

    with fiona.open(str(vector_file)) as vector:
        schema = deepcopy(vector.schema)
        properties = schema.pop('properties', {})
        columns = list(properties.keys()) + [geometry_field]
        data_types = dict(zip(
            columns,
            list(map(fiona.prop_type, properties.values())) + [str]
        ))

        geodata = pd.DataFrame(columns=columns).astype(dtype=data_types)
        # The additional attributes saved to the pandas dataframe are
        # overwritten after the manipulation. Inorder to persist those
        # attributes, they have to be declared on _metadata list of
        # dataframe. However, the operations that return the new object
        # like df.append returns the dataframe without these metadata. so
        # please make sure you copy those metadata before.
        geodata._metadata.extend(['bounds', 'crs'])

        for row in vector:
            data_row = {
                geometry_field: warp.transform_geom(
                    vector.crs,
                    projection,
                    row.get(geometry_field)
                )
            }
            data_row.update(row.get('properties'))
            geodata = geodata.append(data_row, ignore_index=True)

        geodata.bounds = warp.transform_bounds(
            vector.crs,
            projection,
            *vector.bounds
        )
        geodata.crs = vector.crs

    assert not geodata.empty, "Vector file doesn't contain any geometry"
    return geodata


def pixelwise_label_map(labeled_object):
    """Maps the key to each pixels. One use case can be to generate training
    data for pixelwise classification. For multivariable dataset it appends
    the datavariable in second axis. Therefore, if you don't want the value
    of variables in the mapping, pleas make sure that you remove them before
    calling this function.

    Parameters
    ----------
    labeled_object : dict
        Dictionary with class as key and dataset/dataarray in 
        (No of pixels, No of variables/feature) format.

    Returns
    -------
    (label, data)
        Label and data after mapping
    """
    label = None
    data = None
    for data_label, data_value in labeled_object.items():
        if isinstance(data_value, xr.DataArray):
            first_dim = data_value.dims[0]
            pixels = data_value.sizes.get(first_dim)
            if label is None:
                label = np.array([data_label]*pixels)
                data = data_value.values
            else:
                label = np.concatenate((label, [data_label]*pixels))
                data = np.concatenate((data, data_value.values), axis=0)
        elif isinstance(data_value, xr.Dataset):
            first_dim = list(data_value.dims.keys())[0]
            pixels = data_value.sizes.get(first_dim)
            if label is None:
                label = np.array([data_label]*pixels)
            else:
                label = np.concatenate((label, [data_label]*pixels))
            temp_data = None
            for _, var_value in data_value.items():
                if temp_data is None:
                    temp_data = var_value.values
                else:
                    if len(var_value.dims) == 1:
                        temp_data = np.concatenate(
                            (temp_data, var_value.values.reshape(-1, 1)),
                            axis=1
                        )
                    else:
                        temp_data = np.concatenate(
                            (temp_data, var_value.values),
                            axis=1
                        )
            if data is None:
                data = temp_data
            else:
                data = np.concatenate((data, temp_data), axis=0)
    return label, data
