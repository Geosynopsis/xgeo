import os
import pathlib
import re
from copy import deepcopy
import warnings

import numpy as np
import rasterio
import rasterio.features
import rasterio.warp
import xarray as xr
from xgeo.crs import XCRS
from xgeo.utils import (X_DIMS, Y_DIMS, Z_DIMS, T_DIMS, DEFAULT_DIMS)


@xr.register_dataset_accessor('geo')
class XGeoDatasetAccessor(object):
    """
    XGeoDatasetAccessor adds the geospatial functionalities to the xarray Dataset. The accessor makes use of the
    versatility of xarray together with the geospatial operations provided by rasterio together with many custom
    operations that are used in general day to day task in the geospatial world.
    """

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

        self._obj.attrs.update(
            transform=self._obj.attrs.get('transform', None),
            crs=self._obj.attrs.get('crs', None),
            bounds=self._obj.attrs.get('bounds', None),
            origin=self._obj.attrs.get('origin', None),
            resolutions=self._obj.attrs.get('resolutions', None)
        )
        if self._obj.data_vars and any(
                self._is_raster_data_array(data_val) for data_val in self._obj.data_vars.values()):
            # Initialize attributes:
            self.initialize_geo_attributes()

        # Validate and restructure the dataset
        self.validate_and_restructure()

    def initialize_geo_attributes(self):
        self._compute_resolutions()
        self._compute_transform()
        self._compute_origin()
        self._compute_bounds()
        self._find_projection()

    def _is_raster_data_array(self, value: xr.DataArray):
        """
        Checks whether the given DataArray is a raster. The raster objects fulfills following criteria:
            - It should be more or equal to two dimensional data array
            - It should have x and y dimensions

        Parameters
        ----------
        value: xarray.DataArray
            The DataArray to be checked

        Returns
        -------
        is_raster_array: bool
            True if Data Array is raster else False
        """

        if isinstance(value, xr.DataArray) and self.x_dim in value.dims and self.y_dim in value.dims:
            return True
        return False

    def validate_and_restructure(self):
        """
        Validates and restructures the dataset to make full utilization of GeoDataset.
            - Validates if x and y dimensions exists
            - Validates if band and time dimension exists. If they don't exist, it adds those dimensions to the raster
                DataArrays

        Returns
        -------
        dsout: xarray.Dataset
            A copy of original dataset restructured to have all raster DataArray in 4 dimensional format. It allows
            the library to be consistent over its operations.

        """
        for dim in ['x_dim', 'y_dim']:
            assert getattr(self, dim) is not None

        assert any([self._is_raster_data_array(data_var) for data_var in self._obj.data_vars.values()]), \
            "There are no raster DataArray in the Dataset."

        for dim in {'band_dim', 'time_dim'}:
            try:
                getattr(self, dim)
            except AttributeError:
                warnings.warn(
                    "There is no {0} dimension in the DataArray. It will be added to the dataarray.".format(dim)
                )
                for data_var, data_values in self._obj.data_vars.items():
                    # Expand the dimension if the DataArray is a raster.
                    if self._is_raster_data_array(data_values):
                        self._obj[data_var] = data_values.expand_dims(DEFAULT_DIMS.get(dim))
                self._obj = self._obj.assign_coords(**{DEFAULT_DIMS.get(dim): [0]})

    def _compute_resolutions(self):
        """
        Calculates the resolutions according to the current coordinates of the Dataset and adds them into the Dataset
        attribute named resolutions. The resolutions is a tuple as (x resolution, y resolution)
        """
        assert self.x_coords is not None and self.y_coords is not None
        x_resolutions = self.x_coords.diff(self.x_dim)
        y_resolutions = self.y_coords.diff(self.y_dim)
        assert not (
            not x_resolutions.any() or not x_resolutions.all() or not y_resolutions.any() or not y_resolutions.all()), \
            "The resolutions are inconsistent. The library isn't designed to handle inconsistent resolutions"

        self._obj.attrs.update({
            "resolutions": (x_resolutions.values.min(), y_resolutions.values.min())
        })

    @property
    def resolutions(self):
        """
        Gets the resolutions of the DataArrays in Dataset. If the resolutions don't exist, it calculates the
        resolutions from the current coordinates.

        Returns
        -------
        resolutions: (float, float)
            x and y resolutions of the DataArrays.
        """

        if self._obj.attrs.get('resolutions') is not None:
            self._compute_resolutions()
        return self._obj.attrs.get('resolutions')

    def _compute_transform(self):
        """
        Calculates the affine transform parameters from the current coordinates of the Dataset and adds them to the
        attribute of Dataset named transform.
        """
        x_res, y_res = self.resolutions
        x_origin = self.x_coords.values[0] - x_res / 2.0  # PixelAsArea Convention
        y_origin = self.y_coords.values[0] - y_res / 2.0  # PixelAsArea Convention
        transform = (x_res, 0, x_origin, 0, y_res, y_origin)

        self._obj.attrs.update(transform=transform)
        for data_value in self._obj.data_vars.values():
            if not self._is_raster_data_array(data_value):
                continue
            data_value.attrs.update(transform=transform)

    @property
    def transform(self):
        """
        Gets the geo-transform of the Dataset. If the transform isn't present, it calculate the transform from the
        current coordinates of Dataset.
        Returns
        -------
        transform: tuple
            Geo-transform (x resolution, 0, x origin, 0, y resolution, y origin)
        """
        if not self._obj.attrs.get("transform", None):
            self._compute_transform()
        return self._obj.attrs.get('transform')

    def _compute_coords_from_transform(self):
        """
        Computes x and y coordinates from the geo-transform and assigns this coordinates to the Dataset.
        """
        x_res, _, x_origin, _, y_res, y_origin = self.transform
        self._obj.coords.update({
            self.x_dim: x_origin + x_res / 2.0 + np.arange(0, self.x_size) * x_res,
            self.y_dim: y_origin + y_res / 2.0 + np.arange(0, self.y_size) * y_res
        })

    @transform.setter
    def transform(self, trans: tuple or list):
        """
        Sets the geo-transform to the Dataset and updates the x and y coordinates according to the provided
        geo-transform.

        Parameters
        ----------
        trans: list or tuple
            Geo-Transform (x resolution, 0, x origin, 0, y resolution, y origin)

        """
        assert type(trans) in [tuple, list, np.ndarray] and len(trans) == 6, \
            "`trans` should be either tuple or list with 6 numbers"

        self._obj.attrs.update(transform=tuple(trans))

        # Add transforms to all the raster DataArrays as well
        for data_values in self._obj.data_vars.values():
            if self._is_raster_data_array(value=data_values):
                data_values.attrs.update(transform=self._obj.attrs["transform"])

        # Update the coordinates according to the new transform
        self._compute_coords_from_transform()

    def _compute_origin(self):
        """
        Calculates the origin of Dataset in human readable format and adds it to the attribute of Dataset named
        origin.
        The origins could be any of following four:
            - top_left
            - bottom_left
            - top_right
            - bottom_right
        """
        x_origin = {True: 'left', False: 'right'}
        y_origin = {True: 'bottom', False: 'top'}
        x_res, y_res = self.resolutions
        self._obj.attrs.update(origin="{0}_{1}".format(y_origin.get(y_res >= 0), x_origin.get(x_res >= 0)))

    @property
    def origin(self):
        """
        Gets the origin of the Dataset in human readable format.

        Returns
        -------
        origin: str
            Origin of the Dataset.
        """

        if not self._obj.attrs.get('origin'):
            self._compute_origin()
        return self._obj.attrs.get('origin')

    def _update_on_origin(self, origin):
        """
        Updates the Dataset (coordinate systems, transforms, DataArrays etc.) according to the provided origin.
        Parameters
        ----------
        origin: str
            Origin to assign to the Dataset
        """
        yo, xo = self.origin.split('_')
        nyo, nxo = origin.split('_')
        y_coords = self.y_coords
        x_coords = self.x_coords
        if yo != nyo:
            y_coords = self.y_coords[::-1]
        if xo != nxo:
            x_coords = self.x_coords[::-1]
        for data_var, data_value in self._obj.data_vars.items():
            if not self._is_raster_data_array(data_value):
                continue
            if yo != nyo:
                data_value[:] = data_value.loc[{self.y_dim: y_coords}].values
            if xo != nxo:
                data_value[:] = data_value.loc[{self.x_dim: x_coords}].values
            self._obj[data_var] = data_value
        self._obj.coords.update({self.x_dim: x_coords, self.y_dim: y_coords})
        self.initialize_geo_attributes()

    @origin.setter
    def origin(self, value):
        """
        Sets the origin of the Dataset and updates the Dataset with respect to the new origin.

        Parameters
        ----------
        value: str
            Origin to be assigned to Dataset. It can be one of top_left, bottom_left, top_right, bottom_right

        """
        allowed_origins = ['top_left', 'bottom_left', 'top_right', 'bottom_right']
        if not isinstance(value, str) and value not in allowed_origins:
            raise IOError("Either provided value is not string or doesn't belong to one of {}".format(allowed_origins))
        self._update_on_origin(value)
        self._obj.attrs.update(origin=value)

    def _compute_bounds(self):
        # TODO: Validate this
        x_res, _, x_origin, _, y_res, y_origin = self.transform
        x_end = x_origin + self.x_size * x_res
        y_end = y_origin + self.y_size * y_res
        x_options = np.array([x_origin, x_end])
        y_options = np.array([y_origin, y_end])
        self._obj.attrs.update(bounds=(x_options.min(), y_options.min(), x_options.max(), y_options.max()))

    @property
    def bounds(self):
        """
        Gets the bounds of the data.

        Returns
        -------
        bounds: tuple
            Bounds of the data (left, bottom, right, top)
        """
        if not self._obj.attrs.get('bounds', None):
            self._compute_bounds()
        return self._obj.attrs.get('bounds')

    def _find_projection(self):
        """
        Finds the projection system of the Dataset. The method searches whether there exist any value to attribute
        `crs` in Dataset or any DataArray or if grid mapping as in netCDF exists for any DataArray. The method then
        converts the CRS to the proj4 string and adds to attribute of Dataset and DataArray named crs.
        """
        # Search for the projection.
        # 1. Search on Dataset level
        crs = self._obj.attrs.get("crs", None)
        if not crs:
            for data_array in self._obj.data_vars.values():
                if not self._is_raster_data_array(data_array):
                    continue

                # If the DataArray inside the DataSet has the crs, use it.
                crs = data_array.attrs.get("crs")
                if crs:
                    break

                # If the DataArray has grid mapping as in netCDF format. It uses this to determine the crs
                grid_mapping = data_array.attrs.pop('grid_mapping', None)
                if grid_mapping:
                    crs = self._obj.variables.get(grid_mapping).attrs
                    self._obj.drop(grid_mapping)
                    break

        # If crs is found assign it to Dataset and all DataArrays to maintain consistency
        assert crs is not None, "The projection information isn't present in the Dataset."
        self.projection = crs

    @property
    def projection(self):
        """
        Gets the projection/CRS system of the Dataset

        Returns
        -------
        projection: str
            Projection/CRS in proj4 string
        """
        try:
            if self._obj.attrs.get("crs", None) is None:
                self._find_projection()
        except Exception as aep:
            print(aep)
        return XCRS.from_any(self._obj.attrs.get("crs")).to_proj4()

    @projection.setter
    def projection(self, proj: str or int or dict):
        """
        Sets the projection system of the Dataset to the provided projection system. This doesn't reproject the
        Dataset to the assigned projection system. If your intention is to reproject, please use the reproject method.

        Parameters
        ----------
        proj: str or int or dict
            Projection system in any format supported by the rasterio eg. "EPSG:4326" or 4326
        """
        assert isinstance(proj, str) or isinstance(proj, int) or isinstance(proj, dict)
        self._obj.attrs.update(crs=XCRS.from_any(proj).to_proj4())
        # Add the crs information all raster DataArrays as well
        for data_values in self._obj.data_vars.values():
            if self._is_raster_data_array(value=data_values):
                data_values.attrs.update(crs=self._obj.attrs["crs"])

    @property
    def x_dim(self):
        """
        Gets name of X dimension
        Returns
        -------
        x_dim: str
            Name of the X dimension

        """
        for dim in self._obj.dims.keys():
            if dim in X_DIMS:
                return dim
        raise AttributeError("x dimension name isn't understood. Valid names are {}".format(X_DIMS))

    @property
    def x_size(self):
        """
        Gets the size of X dimension
        Returns
        -------
        xsize: int
            Size of X dimension
        """
        return self._obj.dims.get(self.x_dim)

    @property
    def x_coords(self):
        """
        Gets the X coordinates.
        Returns
        -------
        xcoords: xarray.DataArray
            X coordinates of the Dataset
        """
        return self._obj.coords.get(self.x_dim)

    @property
    def y_dim(self):
        """
        Gets name of Y dimension
        Returns
        -------
        y_dim: str
            Name of the y dimension

        """
        for dim in self._obj.dims.keys():
            if dim in Y_DIMS:
                return dim
        raise AttributeError("y dimension name isn't understood. Valid names are {}".format(Y_DIMS))

    @property
    def y_size(self):
        """
        Gets the size of Y dimension
        Returns
        -------
        ysize: int
            Size of Y dimension
        """
        return self._obj.dims.get(self.y_dim)

    @property
    def y_coords(self):
        """
        Gets the Y coordinates of the Dataset.

        Returns
        -------
        ycoords: xarray.DataArray
            Y Coordinates of the Dataset
        """
        return self._obj.coords.get(self.y_dim)

    @property
    def band_dim(self):
        """
        Gets name of band dimension

        Returns
        -------
        band_dim: str
            Name of the band dimension

        """
        for dim in self._obj.dims.keys():
            if dim in Z_DIMS:
                return dim
        raise AttributeError("band dimension name isn't understood. Valid names are {}".format(Z_DIMS))

    @property
    def band_size(self):
        """
        Gets the size of band dimension

        Returns
        -------
        bands: int
            Size of band dimension
        """
        try:
            return self._obj.dims[self.band_dim]
        except KeyError:
            self._obj = self._obj.expand_dims(DEFAULT_DIMS.get('band'))
            return self.band_size

    @property
    def band_coords(self):
        """
        Gets the band coordinates of the Dataset.

        Returns
        -------
        bandcoords: xarray.DataArray
            Band coordinates of the Dataset
        """
        return self._obj.coords.get(self.band_dim)

    @property
    def time_dim(self):
        """
        Gets name of time dimension

        Returns
        -------
        time_dim: str
            Name of the time dimension

        """
        for dim in self._obj.dims.keys():
            if dim in T_DIMS:
                return dim
        raise AttributeError("time dimension name isn't understood, Valid names are {}".format(T_DIMS))

    @property
    def time_size(self):
        """
        Gets the size of time dimension

        Returns
        -------
        times: int
            Size of time dimension
        """
        return self._obj.dims.get(self.time_dim)

    @property
    def time_coords(self):
        """
        Gets the time coordinates of the Dataset

        Returns
        -------
        timecoords: xarray.DataArray
            Time coordinates of the Dataset
        """
        return self._obj.coords.get(self.time_dim)

    @staticmethod
    def __validate_resampling(resampling: rasterio.enums.Resampling or str = None):
        """
        Validates if the resampling is valid <rasterio.enums.Resampling> or strings. If the resampling is the string,
        it fetches the corresponding <rasterio.enums.Resampling> object.

        Parameters
        ----------
        resampling: rasterio.warp.Resampling or str
            User provided resampling method

        Returns
        -------
        resampling: rasterio.warp.Resampling
            Validated resampling method

        """
        if resampling is not None:
            if isinstance(resampling, rasterio.enums.Resampling):
                return resampling
            elif isinstance(resampling, str):
                try:
                    return getattr(rasterio.enums.Resampling, resampling)
                except AttributeError:
                    raise IOError("Invalid resampling method")
            else:
                raise IOError("Resampling method can only be <rasterio.warp.Resampling> or string")
        else:
            return rasterio.enums.Resampling.nearest

    def reproject(self, data_var=None, target_crs=None, resolution=None, resampling=None, target_height=None,
                  target_width=None, source_nodata=None, target_nodata=None, memory_limit=0, threads=None):
        """
        Reprojects and resamples the Dataset.

        Parameters
        ----------
        data_var: str
            The raster DataArray to be reprojected. Defaults to all

        target_crs: int or string or dict
            Target projection/CRS system the DataSet should be reprojected to

        resolution: int or float (Optional)
            Target resolution

        resampling: rasterio.warp.Resampling or string
            Resampling method to be used. Default is 'nearest'

        target_height: int (Optional)
            Target height

        target_width: int (Optional)
            Target width

        source_nodata: int or float (Optional)
            Source NoData value

        target_nodata: int or float (Optional)
            Target NoData value

        memory_limit: int (Optional)
            Maximum memory the process should use. Defaults to 64MB

        threads: int (Optional)
            Number of threads the process should use. Defaults to number of CPU.

        Returns
        -------
        dsout: xarray.Dataset
            Dataset with the reprojected rasters.

        Examples
        --------
            >>> import xgeo  # In order to use the xgeo accessor
            >>> import xarray as xr
            >>> ds = xr.open_rasterio('test.tif')
            >>> ds = ds.to_dataset(name='data')
            >>> ds_reprojected = ds.geo.reproject(target_crs=4326)


        """

        # Validate the data_var exists in the Dataset
        assert data_var is None or self._obj.data_vars.get(data_var), "Selected `data_var` doesn't exist in DataSet"

        with rasterio.Env():
            # Create the transform parameters like affine transforms of source and destination, width and height of the
            # destination raster

            left, bottom, right, top = self.bounds
            src_transform = rasterio.Affine(*self.transform)

            dst_transform, width, height = rasterio.warp.calculate_default_transform(
                src_crs=XCRS.from_any(self.projection),
                dst_crs=XCRS.from_any(target_crs),
                width=self.x_size,
                height=self.y_size,
                left=left,
                right=right,
                bottom=bottom,
                top=top,
                resolution=resolution,
                dst_height=target_height,
                dst_width=target_width
            )

            # Create new GeoDataset, where all the transformed raster will be attached to, add coordinates and
            # dimensions matching the changes introduced by reprojection.
            dsout = xr.Dataset()

            # Copy time and band dimensions and coordinates as they are unaffected by re-projection.
            dsout.coords.update({
                self.time_dim: self.time_coords,
                self.band_dim: self.band_coords
            })

            dst_projection = XCRS.from_any(target_crs).to_string()

            # Re-project the raster DataArray from source to the destination dataset
            resampling = self.__validate_resampling(resampling)
            for data_key, data_array in self._obj.data_vars.items():
                if not self._is_raster_data_array(data_array):
                    # dsout[data_var] = data_values #TODO
                    continue

                # If the data_var is no the current raster DataArray, continue
                if data_var is not None and data_var != data_key:
                    continue

                # Add essential attributes to the raster DataArray
                attrs = deepcopy(data_array.attrs)
                attrs.update(dict(
                    crs=dst_projection,
                    transform=dst_transform[:6],
                    res=(dst_transform[0], dst_transform[4]),
                    nodatavals=target_nodata or 0
                ))
                dsout[data_key] = xr.DataArray(
                    np.ma.asanyarray(
                        np.empty(shape=(self.time_size, self.band_size, height, width), dtype=data_array.dtype)),
                    dims=(self.time_dim, self.band_dim, self.y_dim, self.x_dim), attrs=attrs
                )
                for current_time in self.time_coords:
                    rasterio.warp.reproject(
                        data_array.loc[{self.time_dim: current_time}].values,
                        dsout[data_key].loc[{self.time_dim: current_time}].values,
                        src_nodata=source_nodata,
                        dst_nodata=target_nodata,  # For some reason it doesn't support negative and nan
                        dst_crs=XCRS.from_any(target_crs),
                        src_crs=XCRS.from_any(self.projection),
                        dst_transform=dst_transform,
                        src_transform=src_transform,
                        num_threads=threads or os.cpu_count(),
                        resampling=resampling,
                        warp_mem_limit=memory_limit
                    )

            dsout.attrs = self._obj.attrs
            dsout.geo.transform = tuple(dst_transform[:6])  # We have to look more on that
            dsout.geo.projection = dst_projection
            return dsout

    def __get_geodataframe(self, vector_file, geometry_name="geometry"):
        """
        Gets GeoDataFrame for given vector file. It carries out following list of tasks.
            - Sets the geometry from given geometry_name
            - Reprojects the dataframe to the CRS system of the Dataset
            - Checks whether the Dataframe is empty and whether any geometry lies within the bound of the dataset.
        Parameters
        ----------
        vector_file: str or geopandas.GeoDataFrame
            Vector file to be read

        geometry_name: str
            Name of the geometry in the vector file if it doesn't default to "geometry"

        Returns
        -------
        vector_gdf: geopandas.GeoDataFrame
            GeoDataFrame for the given vector file
        """
        try:
            import geopandas
            import shapely.geometry
        except ModuleNotFoundError:
            raise ModuleNotFoundError("`fiona` module is needed to use this function. Please install it and try again")

        # Read and reproject the vector file
        assert isinstance(vector_file, str) or isinstance(vector_file, geopandas.GeoDataFrame), \
            "Invalid vector_file. The `vector_file` should either be path to file or geopandas.GeoDataFrame"

        if isinstance(vector_file, str):
            vf = geopandas.read_file(vector_file)
        else:
            vf = vector_file

        # Set geometry of the geodataframe
        vf = vf.set_geometry(geometry_name)

        # If the projection system exists. Reproject the vector to the projection system of the data.
        if self.projection:
            vf = vf.to_crs(XCRS.from_any(self.projection).to_dict())

        # Validate that the vector file isn't empty and at least one of the geometries in the vector file is
        # intersecting the raster bounds.
        assert not vf.empty, "Vector file doesn't contain any geometry"
        raster_bound = shapely.geometry.box(*self.bounds)
        assert any([raster_bound.intersects(feature) for feature in vf.geometry]), \
            "No geometry in vector file are intersects the image bound"
        return vf

    def sample(self, vector_file, geometry_name="geometry", value_name="id"):
        """
        Samples the pixel for the given regions. Each sample pixel have all the data values for each timestamp and
        each band.

        Parameters
        ----------
        vector_file: str
            Name of the vector file to be used for the sampling. The vector file can be any one supported by geopandas.
        geometry_name: str
            Name of the geometry in the vector file, if it doesn't default to 'geometry'"
        value_name: str
            Name of the value of each region. This value will be associated with each pixels.

        Returns
        -------
        samples: pandas.Dataframe
            Samples of pixels contained and touched by each regions in pandas.Dataframe.

        Examples
        --------
            >>> import xgeo  # In order to use the xgeo accessor
            >>> import xarray as xr
            >>> ds = xr.open_rasterio('test.tif')
            >>> ds = ds.to_dataset(name='data')
            >>> df_sample = ds.geo.sample(vector_file='test.shp', value_name="class")



        """
        try:
            import pandas as pd
        except ModuleNotFoundError:
            raise ModuleNotFoundError("`pandas` module should be installed to use this functionality")

        # Get geopandas.GeoDataFrame object for given vector file
        vf = self.__get_geodataframe(vector_file=vector_file, geometry_name=geometry_name)

        # Add mask to the Dataset matching the regions in the vector file
        self.add_mask(vector_file=vector_file, geometry_name=geometry_name, mask_name=value_name, value_name=value_name)

        # Collect all pixel and it values for each region.
        df_aggregate = []
        for bound in vf.bounds.values:
            # Subset the data as per the values and change it to pandas.Dataframe
            ds = self._obj.sel({self.x_dim: slice(bound[0], bound[2]), self.y_dim: slice(bound[3], bound[1])})
            df = ds.to_dataframe()

            # Select valid and non nan rows
            df = df.where(df[value_name].isin(vf[value_name])).dropna()

            # Reset the index to x, y, value name, time and band
            df = df.reset_index().set_index([value_name, self.x_dim, self.y_dim, self.time_dim, self.band_dim])
            df_aggregate.append(df)
        return pd.concat(df_aggregate)

    def polygonize(self):
        raise NotImplementedError

    def hillshade(self):
        raise NotImplementedError

    def slope(self):
        raise NotImplementedError

    def stats(self):
        """
        Calculates general statistics mean, standard deviation, max, min of for each band.

        Returns
        -------
        statistics: pandas.Dataframe
            DataFrame with  statistics
        """
        ds_out = xr.Dataset()
        for data_var, data_value in self._obj.data_vars.items():
            if not self._is_raster_data_array(data_value):
                continue
            ds_out["{}_mean".format(data_var)] = data_value.mean(dim=[self.x_dim, self.y_dim])
            ds_out["{}_std".format(data_var)] = data_value.std(dim=[self.x_dim, self.y_dim])
            ds_out["{}_min".format(data_var)] = data_value.min(dim=[self.x_dim, self.y_dim])
            ds_out["{}_max".format(data_var)] = data_value.max(dim=[self.x_dim, self.y_dim])
        return ds_out.to_dataframe()

    def __indices_to_bounds(self, indices):
        """
        Creates bounds (x minimum, y minimum, x maximum, y maximum) from the indices or coordinates in raster
        coordinate system.

        Parameters
        ----------
        indices: tuple/list
            Indices of the bounds (x mimimum index, y minimum index, x maximum index, y maximum index).

        Returns
        -------
        bounds: tuple/list
            Bounds (x minimum, y minimum, x maximum, y maximum)
        """
        x_min_ind, y_min_ind, x_max_ind, y_max_ind = indices
        bounds = (self.x_coords.values[x_min_ind],
                  self.y_coords.values[y_min_ind],
                  self.x_coords.values[x_max_ind],
                  self.y_coords.values[y_max_ind])
        return bounds

    def zonal_stats(self, vector_file, geometry_name="geometry", value_name="id"):
        """
        Calculates statistics for regions in the vector file.

        Parameters
        ----------
        vector_file: str or geopandas.GeoDataFrame
            Vector file with regions/zones for which statistics needs to be calculated

        geometry_name: str
            Name of the geometry column in vector file. Default is "geometry"

        value_name: str
            Name of the value column for each of which the statistics need to be calculated. Default is "id"

        Returns
        -------
        zonal_statistics: pandas.Dataframe
            DataFrame with Statistics

        """
        # Get geopandas.GeoDataframe object for given vector file.
        vf = self.__get_geodataframe(vector_file=vector_file, geometry_name=geometry_name)

        # Add mask with rasterized regions in the given vector file.
        self.add_mask(vector_file=vector_file, geometry_name=geometry_name, value_name=value_name)

        # Collect statistics for the regions
        ds_out = xr.Dataset()
        ds_out.coords.update({
            "stat": ["mean", "std", "min", "max"],
            # value_name: [],
            self.time_dim: self.time_coords,
            self.band_dim: self.band_coords,
        })
        for data_var, data_value in self._obj.data_vars.items():
            if not self._is_raster_data_array(data_value):
                continue

            data_total_stat = []
            value_coords = []
            for val in np.unique(vf.get(value_name)):
                value_coords.append(val)
                temp_val = data_value.where(self._obj.mask == val)
                data_total_stat.append(xr.concat([
                    temp_val.mean(dim=[self.x_dim, self.y_dim]),
                    temp_val.std(dim=[self.x_dim, self.y_dim]),
                    temp_val.min(dim=[self.x_dim, self.y_dim]),
                    temp_val.max(dim=[self.x_dim, self.y_dim])
                ], dim="stat"))

            ds_out.coords.update({
                value_name: value_coords
            })
            ds_out[data_var] = xr.concat(data_total_stat, dim=value_name)
        df = ds_out.to_dataframe()
        df = df.reset_index().set_index([value_name, self.time_dim, self.band_dim, "stat"])
        return df

    def subset(self, vector_file, geometry_name="geometry", crop=False, extent_only=False, invert=False):
        """
        Subset the Dataset with the vector file.
        Parameters
        ----------
        vector_file: str or geopandas.GeoDataFrame
            Path to the vector file. Any vector file supported by GDAL are supported.

        geometry_name: str
            Column name that describes the geometries in the vector file. Default value is "geometry"

        crop: bool
            If True, the output Dataset bounds is approximately equal to the total bounds of the geometry. The
            default value is False

        extent_only: bool
            If True, the output Dataset consists all the data that are within the total bounds of the geometry.
            Default value is True. If extent_only is True, the crop is by default True.

        invert: bool
            If True, the output GeoDataset contains values that are only outside of the geometries. Default value is
            False. This doesn't have effect if extent_only is True

        Returns
        -------
        ds_subset: xarray.Dataset
            Subset dataset

        """

        # Re-structure user input for special cases.
        # If extent_only is true, crop is always true.
        if extent_only:
            crop = True

        # Get GeoDataframe from given vector file
        vf = self.__get_geodataframe(vector_file=vector_file, geometry_name=geometry_name)

        if crop:
            ds_subset = self.slice_dataset(bounds=vf.total_bounds)

            # If extent_only the subset dataset doesn't need to be masked
            if extent_only:
                return ds_subset
        else:
            ds_subset = self._obj.copy()

        # Create a rasterized mask from the GeoDataframe and add it to the Dataset
        ds_subset.geo.add_mask(vector_file=vf, geometry_name=geometry_name)
        for data_var, data_value in ds_subset.data_vars.items():
            if not self._is_raster_data_array(data_value):
                continue
            mask_value = 1 if invert else 0
            ds_subset[data_var] = ds_subset[data_var].where(ds_subset.mask != mask_value)
        return ds_subset

    def add_mask(self, vector_file, geometry_name="geometry", value_name=None, mask_name='mask'):
        """
        Rasterizes the vector_file and add the mask as coordinate with name mask_name to the Dataset

        Parameters
        ----------
        vector_file: str or geopandas.Dataframe
            Vector file which need to be rasterized and added as mask

        geometry_name: str
            Name of geometry column in vector file if it doesn't default to "geometry"

        value_name: str
            Name of the value column, its value will be used to fill the raster. If None, all values in geometry is
            filled with 1

        mask_name: str
            Name of the mask index

        """
        vf = self.__get_geodataframe(vector_file=vector_file, geometry_name=geometry_name)
        with rasterio.Env():
            if value_name is not None:
                assert value_name in vf.columns, "`value_name` should be valid name. For defaults leave it None"
                assert vf.geometry.size == vf.get(value_name).size, \
                    "Rows in `value_name` column and geometries are different"
                geom_iterator = zip(vf.geometry, vf.get(value_name))
            else:
                geom_iterator = zip(vf.geometry, [1] * vf.geometry.size)

            mask = rasterio.features.rasterize(
                geom_iterator,
                transform=self.transform,
                out_shape=(self.y_size, self.x_size)
            )

            self._obj.coords.update({
                mask_name: ((self.y_dim, self.x_dim), mask)
            })

    def slice_dataset(self, indices=None, bounds=None):
        """
        Subsets Dataset either with indices or bounds.
        Parameters
        ----------
        indices: tuple/list
            Indices (row_x_min, col_y_min, row_x_max, col_y_max)
        bounds: tuple/list
            Bounds (x_min, y_min, x_max, y_max)

        Returns
        -------
        ds: xarray.Dataset
            Dataset with data in given bounds or indices
        """
        assert indices is not None or bounds is not None, \
            "Either one of parameters `indices` or `bounds` should be present"
        if indices is not None:
            bounds = self.__indices_to_bounds(indices)
        x_min, y_min, x_max, y_max = bounds
        ds_subset = self._obj.sel({
            self.x_dim: slice(x_min, x_max),
            self.y_dim: slice(y_max, y_min)
        })
        ds_subset.geo.initialize_geo_attributes()
        return ds_subset

    def to_geotiff(self, output_path='.', file_prefix='data', overviews=False, bigtiff=True, compress='lzw',
                   tiled=True, num_threads='ALL_CPUS'):
        """
        Creates one or multiple Geotiffs for the Dataset. If the Dataset has muliple raster arrays or raster arrays for
        multiple timestamps, separate geotiffs are created in following path:
            output_path/<file_prifix>_<variable_nane>_<timestamp>.tif
        Parameters
        ----------
        output_path: str
            Output directory for the files.
        file_prefix: str
            Prefix for the filename
        overviews: bool
            Creates image overviews if True.
        bigtiff: bool
            Creates BigTiff if True
        compress: str
            Compression algorithm to apply, default 'lzw'
        tiled: bool
            The tif is tiled if True
        num_threads: int or str
            The number of threads the process should use. Default is 'ALL_CPUS'

        """

        output_path = pathlib.Path(output_path)
        if not output_path.exists():
            output_path.mkdir(parents=True)
        for current_time in self.time_coords.values:
            for data_var, data_value in self._obj.data_vars.items():
                if not self._is_raster_data_array(data_value):
                    continue
                current_file_name = "{0}_{1}_{2}".format(file_prefix, data_var, current_time)
                current_file_name = re.sub(r'\W', '_', current_file_name)
                current_file_path = (output_path / current_file_name).with_suffix('.tif')
                open_attributes = dict(
                    driver='GTiff',
                    height=self.y_size,
                    width=self.x_size,
                    dtype=str(data_value.dtype),
                    count=self.band_size,
                    crs=self.projection,
                    transform=rasterio.Affine(*self.transform),
                    bigtiff="YES" if bigtiff else "NO",
                    copmress=compress,
                    tiled="YES" if tiled else "NO",
                    NUM_THREADS=num_threads
                )
                with rasterio.open(str(current_file_path), mode='w', **open_attributes) as ds_out:
                    attrs_out = self._obj.attrs
                    attrs_out.update(data_value.attrs)
                    for attr in ['crs', 'transform']:
                        attrs_out.pop(attr, None)
                    ds_out.update_tags(**attrs_out)
                    bands_out = np.arange(1, self.band_size + 1)
                    ds_out.write(data_value.sel(**{self.time_dim: current_time}).values, bands_out)
                    if overviews:
                        factors = [2, 4, 8, 16]
                        ds_out.build_overviews(factors, rasterio.enums.Resampling.average)
                        ds_out.update_tags(ns='rio_overview', resampling='average')
                    ds_out.close()
