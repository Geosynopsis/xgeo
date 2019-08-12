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
from functools import partial

import numpy as np
import pandas as pd
import rasterio
import rasterio.enums as enums
import rasterio.transform as transform
import rasterio.warp as warp
import rasterio.windows as windows
import xarray as xr
from xgeo.crs import XCRS
from xgeo.raster.base import XGeoBaseAccessor
from xgeo.utils import get_geodataframe


@xr.register_dataarray_accessor('geo')
class XGeoDataArrayAccessor(XGeoBaseAccessor):

    def reproject(self, target_crs, resolutions=None, target_height=None,
                  target_width=None, resampling=enums.Resampling.nearest,
                  source_nodata=0, target_nodata=0, memory_limit=0,
                  threads=os.cpu_count()):
        """
        Reprojects and resamples the DataArray.

        Parameters
        ----------
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
                resolution=resolutions,
                dst_height=target_height,
                dst_width=target_width
            )

            # If the resolution is given, the origin of the transform is
            # shifted by the factor of old and new resolution. For some reason
            # warp automatically doensn't handle this. Therefore,we shift the
            # origin by factor in the section below.

            # Affine operation overview for not to get confused.

            #   Affine.translation(tx, ty) T = | 1 0 tx |
            #                                  | 0 1 ty |
            #                                  | 0 0 1  |

            # Affine.scale(sx, sy) S = | sx  0  0 |
            #                          | 0   sy 1 |
            #                          | 0    0 1 |

            # T * S = |sx  0  tx|
            #         |0   sy ty|
            #         |0   0   1|

            if resolutions and resolutions != self.resolutions:
                x_res, y_res = self.resolutions
                xt_res, yt_res = resolutions
                dst_transform = transform.Affine.translation(
                    dst_transform.xoff + (x_res - xt_res) / 2.0,
                    dst_transform.yoff + (y_res - yt_res) / 2.0
                ) * transform.Affine.scale(
                    xt_res,
                    yt_res
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
            dst_dataarray.geo.init_geoparams()

        # The rasterio reprojection only supports either two or three
        # dimensional images. Therefore, we iterate the reprojection by
        # selecting only 2D image using coordinates of non locational
        # dimensions.
        reproject_partial = partial(
            warp.reproject,
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
        if self.non_loc_dims is not None:
            non_loc_coords = map(
                np.ravel,
                np.meshgrid(
                    *(self._obj.coords.get(dim).values for dim in self.non_loc_dims)
                )
            )

            for temp_coords in zip(*non_loc_coords):
                loc_dict = dict(zip(self.non_loc_dims, temp_coords))
                reproject_partial(
                    self._obj.loc[loc_dict].values,
                    dst_dataarray.loc[loc_dict].values,
                )
        else:
            reproject_partial(
                self._obj.values,
                dst_dataarray.values,
            )
        return dst_dataarray

    def resample(self, resolutions=None, target_height=None, target_width=None,
                 resampling=enums.Resampling.nearest):
        """Upsamples or downsamples the xarray dataarray.

        Parameters
        ----------
        resolution : tuple, optional
            Output resolution (x resolution, y resolution), by default None
        target_height : int, optional
            Output height, by default None
        target_width : int, optional
            Output width, by default None
        resampling : rasterio.enums.Resampling or string, optional
            Resampling method, by default enums.Resampling.nearest

        Returns
        -------
        resampled_ds: xarray.DataArray
            Resampled DataArray
        """

        assert resolutions or all([target_height, target_width]), \
            "Either resolution or target_height and target_width parameters \
            should be provided."

        return self.reproject(
            target_crs=self.projection,
            resolutions=resolutions,
            target_height=target_height,
            target_width=target_width,
            resampling=resampling
        )

    def sample(self, vector_file, geometry_field="geometry", label_field="id"):
        """
        Samples the pixel for the given regions. Each sample pixel have all
        the data values.

        Parameters
        ----------
        vector_file: str
            Name of the vector file to be used for the sampling. The vector
            file can be any one supported by geopandas.
        geometry_field: str, optional
            Name of the geometry field in the vector file, by default 'geometry'
        label_field: str, optional
            Name of the label field of each region, by default 'id'

        Returns
        -------
        samples: dict(value, xr.DataArray)
            Dictionary with label as key and 2D Dataarray (No of pixels,
            Pixel data) as value.

        Examples
        --------
            >>> import xgeo  # In order to use the xgeo accessor
            >>> import xarray as xr
            >>> ds = xr.open_rasterio('test.tif')
            >>> ds = ds.to_dataset(name='data')
            >>> df_sample = ds.geo.sample(vector_file='test.shp',
                label_field="class")

        """
        # Get geopandas.GeoDataFrame object for given vector file
        vector = get_geodataframe(
            vector_file=vector_file,
            projection=self.projection,
            geometry_field=geometry_field
        )

        # Add mask to the Dataset matching the regions in the vector file
        mask = self.get_mask(
            vector=vector,
            geometry_field=geometry_field,
            label_field=label_field
        )

        labeled_data = {}
        for val in vector[label_field].unique():
            labeled_data[val] = self._obj.where(mask == val).stack(
                pixel=(self.x_dim, self.y_dim),
                pixel_data=(self.non_loc_dims)
            ).dropna(dim="pixel", how='all')
        return labeled_data

    def stats(self):
        """
        Calculates general statistics mean, standard deviation, max, min of
        for each band.

        Returns
        -------
        statistics: xr.DataArray
            Dataarray with statics with `stat` as new dimension along with
            all non local dimensions.
        """
        name = self._obj.name or "data"

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

    def zonal_stats(self, vector_file, geometry_field="geometry", label_field="id"):
        """
        Calculates statistics for regions in the vector file.

        Parameters
        ----------
        vector_file: str or geopandas.GeoDataFrame
            Vector file with regions/zones for which statistics needs to be
            calculated

        geometry_field: str, optional
            Name of the geometry column in vector file, by default "geometry"

        label_field: str, optional
            Name of the label column for each of which the statistics need to
            be calculated, by default "id"

        Returns
        -------
        zonal_statistics: xr.DataArray
            Dataarray with zonal statistics with zone_id/value and stats as new
            dimension along with the non local dimensions.

        """
        # Get geopandas.GeoDataframe object for given vector file.
        vector = get_geodataframe(
            vector_file=vector_file,
            projection=self.projection,
            geometry_field=geometry_field
        )

        # Add mask with rasterized regions in the given vector file.
        mask = self.get_mask(
            vector=vector,
            geometry_field=geometry_field,
            label_field=label_field
        )

        # Collect statistics for the regions
        stat_collection = []
        value_coords = []
        for val in np.unique(vector.get(label_field)):
            value_coords.append(val)
            temp_val = self._obj.where(mask == val)
            stat_collection.append(xr.concat([
                temp_val.mean(dim=[self.x_dim, self.y_dim], skipna=True),
                temp_val.std(dim=[self.x_dim, self.y_dim], skipna=True),
                temp_val.min(dim=[self.x_dim, self.y_dim], skipna=True),
                temp_val.max(dim=[self.x_dim, self.y_dim], skipna=True)
            ], dim="stat"))

        data_array = xr.concat(stat_collection, dim=label_field)
        data_array.coords.update({
            "stat": ["mean", "std", "min", "max"],
            label_field: value_coords
        })

        return data_array

    def to_geotiff(self, output_path='.', prefix=None, overviews=True,
                   bigtiff=True, compress='lzw', num_threads='ALL_CPUS',
                   tiled=True, chunked=False, dims=None,
                   band_descriptions=None):
        """
        Creates one Geotiff file for the DataArray. If the Dataset has
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
        band_descriptions: list
            Descriptions of band ordered by band number.

        Returns
        -------
        filepath: str
            Path of the file created
        """
        band_descriptions = band_descriptions or []
        assert isinstance(band_descriptions, list), "band description \
            for DataArray should be a list in order of the band indices."

        if dims:
            assert isinstance(dims, dict), "dims only accepts the \
                dictionary as input."

        obj = self._obj
        if dims is not None:
            obj = self._obj.sel(dims)

        assert len(obj.dims) <= 3, "Exporting to geotiff is only \
            supported for array that have 2 or 3 dimensions"

        # Check the output directory path and create if it doesn't already
        # exist.
        output_path = pathlib.Path(output_path)
        if not output_path.exists():
            output_path.mkdir(parents=True)

        # Create the filename and filepath from given attributes.
        # prefix = "_".join(filter(None, [prefix, obj.name]))
        filepath = (output_path / prefix).with_suffix('.tif')

        # The dimension apart from x and y dimension is the band dimension.
        # If the band dimension isn't available that means its a raster
        # with a single band. Therefore in the following script, we figure
        # out the number of bands required in the geotiff.
        loc_dims = [self.x_dim, self.y_dim]
        band_dims = set(obj.dims).difference(loc_dims)
        if band_dims:
            band_dim = band_dims.pop()
            out_bands = np.arange(1, self._obj.sizes.get(band_dim)+1)
            band_size = len(out_bands)
        else:
            out_bands = [1]
            band_size = 1

        create_params = dict(
            driver='GTiff',
            height=self.y_size,
            width=self.x_size,
            dtype=str(obj.dtype),
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
                chunks = dict(zip(obj.dims, obj.chunks))
                xchunk = max(set(chunks.get(self.x_dim)))
                ychunk = max(set(chunks.get(self.y_dim)))

            for xstart in range(0, self.x_size, xchunk):
                for ystart in range(0, self.y_size, ychunk):
                    xwin = min(xchunk, self.x_size - xstart)
                    ywin = min(xchunk, self.y_size - ystart)
                    xend = xstart + xwin
                    yend = ystart + ywin

                    ds_out.write(
                        obj.isel({
                            self.x_dim: slice(xstart, xend),
                            self.y_dim: slice(ystart, yend)
                        }).values,
                        indexes=out_bands,
                        window=windows.Window(xstart, ystart, xwin, ywin)
                    )

            for b, description in enumerate(band_descriptions, start=1):
                ds_out.set_band_description(b, description)

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
        return str(filepath)
