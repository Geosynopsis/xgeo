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

    def to_geotiff(self, output_path='.', prefix=None, overviews=True,
                   bigtiff=True, compress='lzw', num_threads='ALL_CPUS',
                   tiled=True, chunked=False, dims=None,
                   band_descriptions=None):
        """
        Creates one or multiple Geotiffs for the Dataset. If the Dataset has
        muliple raster arrays separate geotiffs are created in paths with
        following pattern:
            output_path/<prifix>_<variable_name>.tif

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
        chunked: bool
            If the Dataset is a dask array with chunks whether to use these
            chunks in windowed writing.
        dims: dict
            If the Dataset has more than three dimensions, a geotiff cannot
            be created unless the dimensions are mentioned. For example, if
            you have x, y, band, time dimensions, you can either create
            single file for one timestamp by providing {'time': time_value}
            or single file for on band dimension {'band': band_value}.
        band_descriptions: dict
            Band descriptions for each variable ordered by band number.
        Returns
        -------
        filepaths: dict
            Filepaths corresponding to the variables
        """
        if band_descriptions:
            assert isinstance(band_descriptions, dict), "Band \
            descriptions for Dataset should be a dictionary with variable as \
            key and band description list ordered as per band indices as value."

        file_paths = {}
        for data_var, dataarray in self._obj.data_vars.items():
            if not self.is_raster(dataarray):
                continue
            prefix = "_".join(filter(None, [prefix, data_var]))
            file_paths[data_var] = dataarray.geo.to_geotiff(
                output_path=output_path,
                prefix=prefix,
                overviews=overviews,
                bigtiff=bigtiff,
                compress=compress,
                num_threads=num_threads,
                tiled=tiled,
                chunked=chunked,
                dims=dims,
                band_descriptions=(band_descriptions or {}).get(data_var)
            )
        return file_paths
