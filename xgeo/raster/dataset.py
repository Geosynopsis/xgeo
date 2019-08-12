"""
XGeoDatasetAccessor adds the geospatial functionalities to the xarray
Dataset. The accessor makes use of the versatility of xarray together with
the geospatial operations provided by rasterio together with many custom
operations that are used in general day to day task in the geospatial world.
"""

import os
import warnings

import rasterio.enums as enums
import xarray as xr
from xgeo.crs import XCRS
from xgeo.raster.base import XGeoBaseAccessor
from xgeo.utils import T_DIMS, DEFAULT_DIMS, Z_DIMS


@xr.register_dataset_accessor('geo')
class XGeoDatasetAccessor(XGeoBaseAccessor):

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

        # If crs is found assign it to Dataset and all DataArrays to maintain
        # consistency
        if crs is None:
            warnings.warn(
                "The projection information isn't available in the given \
                dataset. Please supply the projection system to use \
                projection based functionalities like reprojection."
            )
            return None

        return XCRS.from_any(crs).to_proj4()

    def __warn_depricated(self, value):
        DeprecationWarning(
            "{} will be depricated in the future version of the \
            library.".format(value)
        )

    @property
    def band_dim(self):
        """
        Gets name of band dimension

        Returns
        -------
        band_dim: str
            Name of the band dimension

        """
        self.__warn_depricated('band_dim')
        for dim in self._obj.dims.keys():
            if dim in Z_DIMS:
                return dim
        raise AttributeError(
            "band dimension name isn't understood. Valid names are \
                 {}".format(Z_DIMS)
        )

    @property
    def band_size(self):
        """
        Gets the size of band dimension

        Returns
        -------
        bands: int
            Size of band dimension
        """
        self.__warn_depricated('band_size')
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
        self.__warn_depricated('band_coords')
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
        self.__warn_depricated('time_dim')
        for dim in self._obj.dims.keys():
            if dim in T_DIMS:
                return dim
        raise AttributeError(
            "time dimension name isn't understood, Valid names are \
            {}".format(T_DIMS)
        )

    @property
    def time_size(self):
        """
        Gets the size of time dimension

        Returns
        -------
        times: int
            Size of time dimension
        """
        self.__warn_depricated('time_size')
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
        self.__warn_depricated('time_coords')
        return self._obj.coords.get(self.time_dim)

    def reproject(self, target_crs, resolutions=None, target_height=None,
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

        resolutions: tuple (int or float, int or float) (Optional)
            Target resolution (xres, yres)

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
                    resolutions=resolutions,
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

    def resample(self, resolutions=None, target_height=None, target_width=None,
                 resampling='nearest', source_nodata=0, target_nodata=0,
                 memory_limit=0, threads=os.cpu_count()):
        """Upsamples or downsamples the xarray dataarray.

        Parameters
        ----------

        resolutions : tuple, optional
            Output resolution (x resolution, y resolution), by default None

        target_height : int, optional
            Output height, by default None

        target_width : int, optional
            Output width, by default None

        resampling : rasterio.enums.Resampling or string, optional
            Resampling method, by default enums.Resampling.nearest

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
        resampled_ds: xarray.DataArray
            Resampled DataArray
        """
        out_ds = []
        for var_key, var_value in self._obj.data_vars.items():
            if not self.is_raster(var_value):
                continue
            out_ds.append({
                var_key: var_value.geo.resample(
                    resolutions=resolutions,
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

    def sample(self, vector_file, geometry_field="geometry", label_field="id"):
        """
        Samples the pixel for the given regions. Each sample pixel have all
        the data values.

        Parameters
        ----------
        vector_file: str
            Name of the vector file to be used for the sampling. The vector
            file can be any one supported by fiona.
        geometry_field: str
            Name of the geometry in the vector file, if it doesn't default to
            'geometry'"
        label_field: str
            Name of the value of each region. This value will be associated
            with each pixels.

        Returns
        -------
        samples: dict(value, xr.Dataset)
            Dictionary with label as key and 2D dataset as value.

        Examples
        --------
            >>> import xgeo  # In order to use the xgeo accessor
            >>> import xarray as xr
            >>> ds = xr.open_dataset('test.nc')
            >>> df_sample = ds.geo.sample(vector_file='test.shp', label_field="class")

        """
        out_ds = {}
        for var_key, var_value in self._obj.data_vars.items():
            if not self.is_raster(var_value):
                continue
            labeled_data = var_value.geo.sample(
                vector_file=vector_file,
                geometry_field=geometry_field,
                label_field=label_field
            )
            for lab_key, lab_val in labeled_data.items():
                out_ds[lab_key] = out_ds.get(lab_key, []) + [{
                    var_key: lab_val
                }]
        for lab_key, lab_val in out_ds.items():
            out_ds[lab_key] = xr.merge(out_ds.get(lab_key))

        return out_ds

    def stats(self):
        """
        Calculates general statistics mean, standard deviation, max, min of
        for each band.

        Returns
        -------
        statistics: xr.Dataset 
            Dataset with non local dimensions and additional
            `stats` dimension.
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

    def zonal_stats(self, vector_file, geometry_field="geometry", label_field="id"):
        """
        Calculates statistics for regions in the vector file.

        Parameters
        ----------
        vector_file: str or geopandas.GeoDataFrame
            Vector file with regions/zones for which statistics needs to be
            calculated

        geometry_field: str
            Name of the geometry column in vector file. Default is "geometry"

        label_field: str
            Name of the value column for each of which the statistics need to
            be calculated. Default is "id"

        Returns
        -------
        zonal_statistics: xr.Dataset
            Dataset with zonal statistics which has label_field and stats as
            new dimensions additional to all non local dimensions.

        """
        out_ds = []
        for var_key, var_value in self._obj.data_vars.items():
            if not self.is_raster(var_value):
                continue
            out_ds.append({
                var_key: var_value.geo.zonal_stats(
                    vector_file=vector_file,
                    geometry_field=geometry_field,
                    label_field=label_field
                )
            })
        out_ds = xr.merge(out_ds)
        out_ds.attrs.update(**self._obj.attrs)
        return out_ds

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
