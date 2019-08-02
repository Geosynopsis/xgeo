"""

XGeoDataArryAccessor adds the geospatial functionalities to the xarray
DataArray. The accessor makes use of the versatility of xarray together
with the geospatial operations provided by rasterio together with many
custom operations that are used in general day to day task in the
geospatial world.

"""
import collections
import os
import pathlib
from copy import deepcopy

import numpy as np
import rasterio
import rasterio.enums as enums
import rasterio.warp as warp
import rasterio.windows as windows
import xarray as xr
from xgeo.crs import XCRS
from xgeo.raster.base import XGeoBaseAccessor


@xr.register_dataarray_accessor('geo')
class XGeoDataArrayAccessor(XGeoBaseAccessor):

    def reproject(self, target_crs, resolution=None, target_height=None,
                  target_width=None, resampling=enums.Resampling.nearest,
                  source_nodata=0, target_nodata=0, memory_limit=0,
                  threads=os.cpu_count()):
        """
        Reprojects and resamples the DataArray.

        Parameters
        ----------
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
        dsout: xarray.DataArray
            DataArray with the reprojected rasters.

        Examples
        --------
            >>> import xgeo  # In order to use the xgeo accessor
            >>> import xarray as xr
            >>> ds = xr.open_rasterio('test.tif')
            >>> ds = ds.to_dataset(name='data')
            >>> ds_reprojected = ds.geo.reproject(target_crs=4326)


        """

        with rasterio.Env():
            # Create the transform parameters like affine transforms of source
            # and destination, width and height of the destination raster

            left, bottom, right, top = self.bounds
            src_transform = rasterio.Affine(*self.transform)

            dst_transform, width, height = warp.calculate_default_transform(
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

            dst_projection = XCRS.from_any(target_crs).to_string()

            # Re-project the raster DataArray from source to the destination
            # dataset
            resampling = self._validate_resampling(resampling)

            # In following section, we initialize the output data array. As
            # only the shape along the x and y dimension changes, the rest of
            # the dimensions of the output data array should be same as the
            # original.
            dst_shape = collections.OrderedDict(self._obj.sizes)
            dst_shape[self.x_dim] = width
            dst_shape[self.y_dim] = height

            dst_coords = collections.OrderedDict(self._obj.coords)
            # Since the reprojection changes the coordinates on x and y
            # dimension, here we delete the coordinates and we will recompute
            # the coordinates from the transformation after the initialization
            # of the data array.
            del dst_coords[self.x_dim]
            del dst_coords[self.y_dim]

            # Prepare essential attributes for the new raster DataArray. The
            # geotransform and the crs system of the projected system changes
            # therefore, those values should be overwritten with the new
            # values.
            dst_attrs = deepcopy(self._obj.attrs)
            # dst_attrs.update(
            #     transform=tuple(dst_transform[:6]),
            #     crs=dst_projection
            # )

            dst_dataarray = xr.DataArray(
                np.ma.asanyarray(
                    np.empty(
                        shape=list(dst_shape.values()),
                        dtype=self._obj.dtype
                    )
                ),
                dims=self._obj.dims,
                attrs=dst_attrs,
                coords=dst_coords
            )
            # Recompute the coordinates from the new geotransform
            dst_dataarray.geo.transform = tuple(dst_transform[:6])
            dst_dataarray.geo.projection = dst_projection

        # The rasterio reprojection only supports either two or three
        # dimensional images. Therefore, we iterate the reprojection by
        # selecting only 2D image using coordinates of non locational
        # dimensions.
        non_loc_coords = map(
            np.ravel,
            np.meshgrid(
                *(self._obj.coords.get(dim).values for dim in self.non_loc_dims)
            )
        )
        for temp_coords in zip(*non_loc_coords):
            loc_dict = dict(zip(self.non_loc_dims, temp_coords))
            warp.reproject(
                self._obj.loc[loc_dict].values,
                dst_dataarray.loc[loc_dict].values,
                src_nodata=source_nodata,
                dst_nodata=target_nodata,  # doesn't support negative and nan
                dst_crs=XCRS.from_any(target_crs),
                src_crs=XCRS.from_any(self.projection),
                dst_transform=dst_transform,
                src_transform=src_transform,
                num_threads=threads or os.cpu_count(),
                resampling=resampling,
                warp_mem_limit=memory_limit
            )

        dst_dataarray.geo.init_geoparams()
        return dst_dataarray

    def resample(self, resolution=None, target_height=None, target_width=None,
                 resampling=enums.Resampling.nearest):

        assert resolution or all([target_height, target_width]), \
            "Either resolution or target_height and target_width parameters \
            should be provided."

        return self.reproject(
            target_crs=self.projection,
            resolution=resolution,
            target_height=target_height,
            target_width=target_width,
            resampling=resampling
        )

    def sample(self, vector_file, geometry_name="geometry", value_name="id"):
        """
        Samples the pixel for the given regions. Each sample pixel have all
        the data values for each timestamp and each band.

        Parameters
        ----------
        vector_file: str
            Name of the vector file to be used for the sampling. The vector
            file can be any one supported by geopandas.
        geometry_name: str
            Name of the geometry in the vector file, if it doesn't default to
            'geometry'"
        value_name: str
            Name of the value of each region. This value will be associated
            with each pixels.

        Returns
        -------
        samples: pandas.Dataframe
            Samples of pixels contained and touched by each regions in
            pandas.Dataframe.

        Examples
        --------
            >>> import xgeo  # In order to use the xgeo accessor
            >>> import xarray as xr
            >>> ds = xr.open_rasterio('test.tif')
            >>> ds = ds.to_dataset(name='data')
            >>> df_sample = ds.geo.sample(vector_file='test.shp',
                value_name="class")

        """
        try:
            import pandas as pd
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "`pandas` module should be installed to use this functionality"
            )

        # Get geopandas.GeoDataFrame object for given vector file
        vector = self._get_geodataframe(
            vector_file=vector_file,
            geometry_name=geometry_name
        )

        # Add mask to the Dataset matching the regions in the vector file
        mask = self.get_mask(
            vector_file=vector_file,
            geometry_name=geometry_name,
            value_name=value_name
        )

        data_array = deepcopy(self._obj)
        data_array.coords.update(**{value_name: mask})

        # Collect all pixel and it values for each region.
        dataframe_aggregate = []
        for bound in vector.bounds.values:
            # Subset the data as per the values and change it to pandas.
            dataarray_subset = data_array.sel({
                self.x_dim: slice(bound[0], bound[2]),
                self.y_dim: slice(bound[3], bound[1])
            })

            dataframe = dataarray_subset.to_dataframe()

            # Select valid and non nan rows
            dataframe = dataframe.where(
                dataframe[value_name].isin(vector[value_name])
            ).dropna()
            dataframe_aggregate.append(dataframe)
        return pd.concat(dataframe_aggregate)

    def stats(self, name=None):
        """
        Calculates general statistics mean, standard deviation, max, min of
        for each band.

        Returns
        -------
        statistics: pandas.Dataframe
            DataFrame with  statistics
        """
        name = name or self._obj.name or "data"

        da = xr.DataArray(
            xr.concat([
                self._obj.mean(dim=[self.x_dim, self.y_dim]),
                self._obj.std(dim=[self.x_dim, self.y_dim]),
                self._obj.min(dim=[self.x_dim, self.y_dim]),
                self._obj.max(dim=[self.x_dim, self.y_dim]),
            ], dim="stat"),
            name=name
        )
        da.coords.update({"stat": ["mean", "std", "min", "max"]})
        return da

    def zonal_stats(self, vector_file, geometry_name="geometry", value_name="id"):
        """
        Calculates statistics for regions in the vector file.

        Parameters
        ----------
        vector_file: str or geopandas.GeoDataFrame
            Vector file with regions/zones for which statistics needs to be
            calculated

        geometry_name: str
            Name of the geometry column in vector file. Default is "geometry"

        value_name: str
            Name of the value column for each of which the statistics need to
            be calculated. Default is "id"

        Returns
        -------
        zonal_statistics: pandas.Dataframe
            DataFrame with Statistics

        """
        # Get geopandas.GeoDataframe object for given vector file.
        vector = self._get_geodataframe(
            vector_file=vector_file,
            geometry_name=geometry_name
        )

        # Add mask with rasterized regions in the given vector file.
        mask = self.get_mask(
            vector_file=vector_file,
            geometry_name=geometry_name,
            value_name=value_name
        )

        # Collect statistics for the regions
        stat_collection = []
        value_coords = []
        for val in np.unique(vector.get(value_name)):
            value_coords.append(val)
            temp_val = self._obj.where(mask == val)
            stat_collection.append(xr.concat([
                temp_val.mean(dim=[self.x_dim, self.y_dim], skipna=True),
                temp_val.std(dim=[self.x_dim, self.y_dim], skipna=True),
                temp_val.min(dim=[self.x_dim, self.y_dim], skipna=True),
                temp_val.max(dim=[self.x_dim, self.y_dim], skipna=True)
            ], dim="stat"))

        data_array = xr.concat(stat_collection, dim=value_name)
        data_array.coords.update({
            "stat": ["mean", "std", "min", "max"],
            value_name: value_coords
        })

        return data_array

    def to_geotiff(self, output_path='.', prefix=None, overviews=True,
                   bigtiff=True, compress='lzw', num_threads='ALL_CPUS',
                   tiled=True, chunked=False):
        """
        Creates Geotiffs from the DataArray. The geotiffs are created in
        following path:
            output_path/<prefix>_<variable_name>_<timestamp>.tif

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
        assert len(self._obj.dims) <= 3, "Exporting to geotiff is only \
            supported for array that have 2 or 3 dimensions"

        # Check the output directory path and create if it doesn't already
        # exist.
        output_path = pathlib.Path(output_path)
        if not output_path.exists():
            output_path.mkdir(parents=True)

        # Create the filename and filepath from given attributes.
        filename = "_".join(filter(None, [prefix, self._obj.name or "data"]))
        filepath = (output_path / filename).with_suffix('.tif')

        # The dimension apart from x and y dimension is the band dimension.
        # If the band dimension isn't available that means its a raster
        # with a single band. Therefore in the following script, we figure
        # out the number of bands required in the geotiff.
        loc_dims = [self.x_dim, self.y_dim]
        band_dims = set(self._obj.dims).difference(loc_dims)
        if band_dims:
            band_dim = band_dims.pop()
            out_bands = np.arange(1, self._obj.sizes.get(band_dim)+1)
            band_size = len(out_bands)
        else:
            out_bands = 1
            band_size = 1

        create_params = dict(
            driver='GTiff',
            height=self.y_size,
            width=self.x_size,
            dtype=str(self._obj.dtype),
            count=band_size,
            crs=self.projection,
            transform=rasterio.Affine(*self.transform),
            bigtiff="YES" if bigtiff else "NO",
            copmress=compress,
            tiled="YES" if tiled else "NO",
            NUM_THREADS=num_threads
        )
        with rasterio.open(str(filepath), mode='w', **create_params) as ds_out:
            ds_out.update_tags(**self._obj.attrs)

            # If the dataarray is a dask data array, there are chunk values
            # associated with the data array. Therefore, we make a windowed
            # write to the geotiff which will not overwhelm memory. The chunk
            # values are in same order as the dimensions. For example if the
            # dimension is ('band', 'x', 'y'), the chunks are represented in
            # format like ((1,), (100, 100, 100, .., 80), (100, 100, 100, ...,
            # 80)). From the format above, its clear that the user supplied
            # chunk size is 1, 100 and 100 in band, x and y directions
            # respectively. We use this understanding to derive the sizes of
            # window.
            xchunk = self.x_size
            ychunk = self.y_size
            if chunked and self._obj.chunks is not None:
                chunks = dict(zip(self._obj.dims, self._obj.chunks))
                xchunk = max(set(chunks.get(self.x_dim)))
                ychunk = max(set(chunks.get(self.y_dim)))

            for xstart in range(0, self.x_size, xchunk):
                for ystart in range(0, self.y_size, ychunk):
                    xwin = min(xchunk, self.x_size - xstart)
                    ywin = min(xchunk, self.y_size - ystart)
                    xend = xstart + xwin
                    yend = ystart + ywin

                    ds_out.write(
                        self._obj.isel({
                            self.x_dim: slice(xstart, xend),
                            self.y_dim: slice(ystart, yend)
                        }).values,
                        indexes=out_bands,
                        window=windows.Window(xstart, ystart, xwin, ywin)
                    )
            ds_out.close()
        if overviews:
            # In order to build the overviews of the raster data, I don't
            # know why but for some reason, the data has to be written and
            # reopened in editing mode.
            with rasterio.open(str(filepath), 'r+') as ds_out:
                factors = [2, 4, 8, 16]
                ds_out.build_overviews(factors, enums.Resampling.average)
                ds_out.update_tags(ns='rio_overview', resampling='average')
                ds_out.close()
