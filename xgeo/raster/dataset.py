"""
XGeoDatasetAccessor adds the geospatial functionalities to the xarray
Dataset. The accessor makes use of the versatility of xarray together with
the geospatial operations provided by rasterio together with many custom
operations that are used in general day to day task in the geospatial world.
"""

import os
import pathlib
import re
import warnings
from copy import deepcopy

import numpy as np
import rasterio
import rasterio.enums as enums
import rasterio.features as features
import rasterio.warp as warp
import xarray as xr
from xgeo.crs import XCRS
from xgeo.utils import DEFAULT_DIMS, T_DIMS, X_DIMS, Y_DIMS, Z_DIMS
from xgeo.raster.base import XGeoBaseAccessor


@xr.register_dataset_accessor('geo')
class XGeoDatasetAccessor(XGeoBaseAccessor):

    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self.init_geoparams()
        self.projection = self.search_projection()

        # self._obj.attrs.update(
        #     transform=self._obj.attrs.get('transform', None),
        #     crs=self._obj.attrs.get('crs', None),
        #     bounds=self._obj.attrs.get('bounds', None),
        #     origin=self._obj.attrs.get('origin', None),
        #     resolutions=self._obj.attrs.get('resolutions', None)
        # )

        # if any(map(self.is_raster, self._obj.data_vars.values())):
        #     # Initialize attributes:
        #     self.init_geoparams()

        # # Validate and restructure the dataset
        # self.validate_and_restructure()

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
                    "There is no {0} dimension in the DataArray. It will be added to the dataarray.".format(
                        dim)
                )
                for data_var, data_values in self._obj.data_vars.items():
                    # Expand the dimension if the DataArray is a raster.
                    if self._is_raster_data_array(data_values):
                        self._obj[data_var] = data_values.expand_dims(
                            DEFAULT_DIMS.get(dim))
                self._obj = self._obj.assign_coords(
                    **{DEFAULT_DIMS.get(dim): [0]})

    # def _compute_resolutions(self):
    #     """
    #     Calculates the resolutions according to the current coordinates of the Dataset and adds them into the Dataset
    #     attribute named resolutions. The resolutions is a tuple as (x resolution, y resolution)
    #     """
    #     assert self.x_coords is not None and self.y_coords is not None
    #     x_resolutions = self.x_coords.diff(self.x_dim)
    #     y_resolutions = self.y_coords.diff(self.y_dim)
    #     assert not (
    #         not x_resolutions.any() or not x_resolutions.all() or not y_resolutions.any() or not y_resolutions.all()), \
    #         "The resolutions are inconsistent. The library isn't designed to handle inconsistent resolutions"

    #     self._obj.attrs.update({
    #         "resolutions": (x_resolutions.values.min(), y_resolutions.values.min())
    #     })

    # @property
    # def resolutions(self):
    #     """
    #     Gets the resolutions of the DataArrays in Dataset. If the resolutions don't exist, it calculates the
    #     resolutions from the current coordinates.

    #     Returns
    #     -------
    #     resolutions: (float, float)
    #         x and y resolutions of the DataArrays.
    #     """

    #     if self._obj.attrs.get('resolutions') is not None:
    #         self._compute_resolutions()
    #     return self._obj.attrs.get('resolutions')

    # def _compute_transform(self):
    #     """
    #     Calculates the affine transform parameters from the current coordinates of the Dataset and adds them to the
    #     attribute of Dataset named transform.
    #     """
    #     x_res, y_res = self.resolutions
    #     x_origin = self.x_coords.values[0] - \
    #         x_res / 2.0  # PixelAsArea Convention
    #     y_origin = self.y_coords.values[0] - \
    #         y_res / 2.0  # PixelAsArea Convention
    #     transform = (x_res, 0, x_origin, 0, y_res, y_origin)

    #     self._obj.attrs.update(transform=transform)
    #     for data_value in self._obj.data_vars.values():
    #         if not self._is_raster_data_array(data_value):
    #             continue
    #         data_value.attrs.update(transform=transform)

    # def _compute_coords_from_transform(self):
    #     """
    #     Computes x and y coordinates from the geo-transform and assigns this coordinates to the Dataset.
    #     """
    #     x_res, _, x_origin, _, y_res, y_origin = self.transform
    #     self._obj.coords.update({
    #         self.x_dim: x_origin + x_res / 2.0 + np.arange(0, self.x_size) * x_res,
    #         self.y_dim: y_origin + y_res / 2.0 +
    #         np.arange(0, self.y_size) * y_res
    #     })

    # def _compute_origin(self):
    #     """
    #     Calculates the origin of Dataset in human readable format and adds it to the attribute of Dataset named
    #     origin.
    #     The origins could be any of following four:
    #         - top_left
    #         - bottom_left
    #         - top_right
    #         - bottom_right
    #     """
    #     x_origin = {True: 'left', False: 'right'}
    #     y_origin = {True: 'bottom', False: 'top'}
    #     x_res, y_res = self.resolutions
    #     self._obj.attrs.update(origin="{0}_{1}".format(
    #         y_origin.get(y_res >= 0), x_origin.get(x_res >= 0)))

    # @property
    # def origin(self):
    #     """
    #     Gets the origin of the Dataset in human readable format.

    #     Returns
    #     -------
    #     origin: str
    #         Origin of the Dataset.
    #     """

    #     if not self._obj.attrs.get('origin'):
    #         self._compute_origin()
    #     return self._obj.attrs.get('origin')

    # def _update_on_origin(self, origin):
    #     """
    #     Updates the Dataset (coordinate systems, transforms, DataArrays etc.) according to the provided origin.
    #     Parameters
    #     ----------
    #     origin: str
    #         Origin to assign to the Dataset
    #     """
    #     yo, xo = self.origin.split('_')
    #     nyo, nxo = origin.split('_')
    #     y_coords = self.y_coords
    #     x_coords = self.x_coords
    #     if yo != nyo:
    #         y_coords = self.y_coords[::-1]
    #     if xo != nxo:
    #         x_coords = self.x_coords[::-1]
    #     for data_var, data_value in self._obj.data_vars.items():
    #         if not self._is_raster_data_array(data_value):
    #             continue
    #         if yo != nyo:
    #             data_value[:] = data_value.loc[{self.y_dim: y_coords}].values
    #         if xo != nxo:
    #             data_value[:] = data_value.loc[{self.x_dim: x_coords}].values
    #         self._obj[data_var] = data_value
    #     self._obj.coords.update({self.x_dim: x_coords, self.y_dim: y_coords})
    #     self.initialize_geo_attributes()

    # @origin.setter
    # def origin(self, value):
    #     """
    #     Sets the origin of the Dataset and updates the Dataset with respect to the new origin.

    #     Parameters
    #     ----------
    #     value: str
    #         Origin to be assigned to Dataset. It can be one of top_left, bottom_left, top_right, bottom_right

    #     """
    #     allowed_origins = ['top_left', 'bottom_left',
    #                        'top_right', 'bottom_right']
    #     if not isinstance(value, str) and value not in allowed_origins:
    #         raise IOError("Either provided value is not string or doesn't belong to one of {}".format(
    #             allowed_origins))
    #     self._update_on_origin(value)
    #     self._obj.attrs.update(origin=value)

    # def _compute_bounds(self):
    #     # TODO: Validate this
    #     x_res, _, x_origin, _, y_res, y_origin = self.transform
    #     x_end = x_origin + self.x_size * x_res
    #     y_end = y_origin + self.y_size * y_res
    #     x_options = np.array([x_origin, x_end])
    #     y_options = np.array([y_origin, y_end])
    #     self._obj.attrs.update(
    #         bounds=(x_options.min(), y_options.min(), x_options.max(), y_options.max()))

    # @property
    # def bounds(self):
    #     """
    #     Gets the bounds of the data.

    #     Returns
    #     -------
    #     bounds: tuple
    #         Bounds of the data (left, bottom, right, top)
    #     """
    #     if not self._obj.attrs.get('bounds', None):
    #         self._compute_bounds()
    #     return self._obj.attrs.get('bounds')

    def search_projection(self):
        """
        Finds the projection system of the Dataset. The method searches
        whether there exist any value to attribute `crs` in Dataset or any
        DataArray or if grid mapping as in netCDF exists for any DataArray.
        The method then converts the CRS to the proj4 string and adds to
        attribute of Dataset and DataArray named crs.
        """
        # Search for the projection.
        # 1. Search on Dataset level
        crs = self._obj.attrs.get("crs", None)
        if not crs:
            for data_array in self._obj.data_vars.values():
                # If the DataArray inside the DataSet has the crs, use it.
                crs = data_array.attrs.get("crs")
                if crs:
                    break

                # If the DataArray has grid mapping as in netCDF format. It
                # uses this to determine the crs
                grid_mapping = data_array.attrs.get('grid_mapping_name', None)
                if grid_mapping:
                    crs = data_array.attrs
                    break

        # If crs is found assign it to Dataset and all DataArrays to maintain consistency
        if crs is None:
            warnings.warn(
                "The projection information isn't available in the given \
                dataset. Please supply the projection system to use \
                projection based functionalities like reprojection."
            )
            return None

        return XCRS.from_any(crs).to_proj4()

    @property
    def projection(self):
        """
        Gets the projection/CRS system of the Dataset

        Returns
        -------
        projection: str
            Projection/CRS in proj4 string
        """
        return self._obj.attrs.get("crs")
        # if self._obj.attrs.get("crs", None) is None:
        #     self._obj.attrs.update(crs=self.search_projection())
        #     for dataarray in self._obj.data_vars.values():
        #         if self.is_raster(dataarray):
        #             dataarray.geo.projection = self._obj.attrs.get("crs")
        # return self._obj.attrs.get("crs")

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
        assert isinstance(proj, str) or isinstance(proj, int) \
            or isinstance(proj, dict)
        self._obj.attrs.update(crs=XCRS.from_any(proj).to_proj4())
        # Add the crs information all raster DataArrays as well
        for data_values in self._obj.data_vars.values():
            if self.is_raster(value=data_values):
                data_values.geo.projection = self._obj.attrs.get("crs")

    # @property
    # def x_dim(self):
    #     """
    #     Gets name of X dimension
    #     Returns
    #     -------
    #     x_dim: str
    #         Name of the X dimension

    #     """
    #     for dim in self._obj.dims.keys():
    #         if dim in X_DIMS:
    #             return dim
    #     raise AttributeError(
    #         "x dimension name isn't understood. Valid names are {}".format(X_DIMS))

    # @property
    # def x_size(self):
    #     """
    #     Gets the size of X dimension
    #     Returns
    #     -------
    #     xsize: int
    #         Size of X dimension
    #     """
    #     return self._obj.dims.get(self.x_dim)

    # @property
    # def x_coords(self):
    #     """
    #     Gets the X coordinates.
    #     Returns
    #     -------
    #     xcoords: xarray.DataArray
    #         X coordinates of the Dataset
    #     """
    #     return self._obj.coords.get(self.x_dim)

    # @property
    # def y_dim(self):
    #     """
    #     Gets name of Y dimension
    #     Returns
    #     -------
    #     y_dim: str
    #         Name of the y dimension

    #     """
    #     for dim in self._obj.dims.keys():
    #         if dim in Y_DIMS:
    #             return dim
    #     raise AttributeError(
    #         "y dimension name isn't understood. Valid names are {}".format(Y_DIMS))

    # @property
    # def y_size(self):
    #     """
    #     Gets the size of Y dimension
    #     Returns
    #     -------
    #     ysize: int
    #         Size of Y dimension
    #     """
    #     return self._obj.dims.get(self.y_dim)

    # @property
    # def y_coords(self):
    #     """
    #     Gets the Y coordinates of the Dataset.

    #     Returns
    #     -------
    #     ycoords: xarray.DataArray
    #         Y Coordinates of the Dataset
    #     """
    #     return self._obj.coords.get(self.y_dim)

    @property
    def band_dim(self):
        """
        Gets name of band dimension

        Returns
        -------
        band_dim: str
            Name of the band dimension

        """
        DeprecationWarning(f"The band dim will be depricated soo in future.")
        for dim in self._obj.dims.keys():
            if dim in Z_DIMS:
                return dim
        raise AttributeError(
            "band dimension name isn't understood. Valid names are {}".format(Z_DIMS))

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
        raise AttributeError(
            "time dimension name isn't understood, Valid names are {}".format(T_DIMS))

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

    # @staticmethod
    # def __validate_resampling(resampling: enums.Resampling or str = None):
    #     """
    #     Validates if the resampling is valid <rasterio.enums.Resampling> or strings. If the resampling is the string,
    #     it fetches the corresponding <rasterio.enums.Resampling> object.

    #     Parameters
    #     ----------
    #     resampling: rasterio.warp.Resampling or str
    #         User provided resampling method

    #     Returns
    #     -------
    #     resampling: rasterio.warp.Resampling
    #         Validated resampling method

    #     """
    #     if resampling is not None:
    #         if isinstance(resampling, enums.Resampling):
    #             return resampling
    #         elif isinstance(resampling, str):
    #             try:
    #                 return getattr(enums.Resampling, resampling)
    #             except AttributeError:
    #                 raise IOError("Invalid resampling method")
    #         else:
    #             raise IOError(
    #                 "Resampling method can only be <rasterio.warp.Resampling> or string")
    #     else:
    #         return enums.Resampling.nearest

    def reproject(self, target_crs, resolution=None, target_height=None,
                  target_width=None, resampling=enums.Resampling.nearest,
                  source_nodata=0, target_nodata=0, memory_limit=0,
                  threads=os.cpu_count()):
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
        out_ds = []
        for var_key, var_value in self._obj.data_vars.items():
            if not self.is_raster(var_value):
                continue
            out_ds.append({
                var_key: var_value.geo.reproject(
                    target_crs,
                    resolution=resolution,
                    target_height=target_height,
                    target_width=target_width,
                    resampling=resampling,
                    source_nodata=source_nodata,
                    target_nodata=target_nodata,
                    memory_limit=memory_limit,
                    threads=threads
                )
            })
        out_ds = xr.merge(out_ds)
        out_ds.attrs.update(**self._obj.attrs)
        return out_ds

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
        out_ds = []
        for var_key, var_value in self._obj.data_vars.items():
            if not self.is_raster(var_value):
                continue
            out_ds.append({
                var_key: var_value.geo.sample(
                    vector_file=vector_file,
                    geometry_name=geometry_name,
                    value_name=value_name
                )
            })
        out_ds = xr.merge(out_ds)
        out_ds.attrs.update(**self._obj.attrs)
        return out_ds

    def stats(self):
        """
        Calculates general statistics mean, standard deviation, max, min of for each band.

        Returns
        -------
        statistics: pandas.Dataframe
            DataFrame with  statistics
        """
        out_ds = []
        for var_key, var_value in self._obj.data_vars.items():
            if not self.is_raster(var_value):
                continue
            out_ds.append({
                var_key: var_value.geo.stats()
            })
        out_ds = xr.merge(out_ds)
        out_ds.attrs.update(**self._obj.attrs)
        return out_ds

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
        out_ds = []
        for var_key, var_value in self._obj.data_vars.items():
            if not self.is_raster(var_value):
                continue
            out_ds.append({
                var_key: var_value.geo.zonal_stats(
                    vector_file=vector_file,
                    geometry_name=geometry_name,
                    value_name=value_name
                )
            })
        out_ds = xr.merge(out_ds)
        out_ds.attrs.update(**self._obj.attrs)
        return out_ds

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
        vf = self.__get_geodataframe(
            vector_file=vector_file, geometry_name=geometry_name)

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
            ds_subset[data_var] = ds_subset[data_var].where(
                ds_subset.mask != mask_value)
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
        vf = self.__get_geodataframe(
            vector_file=vector_file, geometry_name=geometry_name)
        with rasterio.Env():
            if value_name is not None:
                assert value_name in vf.columns, "`value_name` should be valid name. For defaults leave it None"
                assert vf.geometry.size == vf.get(value_name).size, \
                    "Rows in `value_name` column and geometries are different"
                geom_iterator = zip(vf.geometry, vf.get(value_name))
            else:
                geom_iterator = zip(vf.geometry, [1] * vf.geometry.size)

            mask = features.rasterize(
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
                current_file_name = "{0}_{1}_{2}".format(
                    file_prefix, data_var, current_time)
                current_file_name = re.sub(r'\W', '_', current_file_name)
                current_file_path = (
                    output_path / current_file_name).with_suffix('.tif')
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
                    ds_out.write(data_value.sel(
                        **{self.time_dim: current_time}).values, bands_out)
                    if overviews:
                        factors = [2, 4, 8, 16]
                        ds_out.build_overviews(
                            factors, enums.Resampling.average)
                        ds_out.update_tags(
                            ns='rio_overview', resampling='average')
                    ds_out.close()
