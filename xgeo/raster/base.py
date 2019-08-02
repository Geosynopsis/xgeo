"""
The Base Accessor that provides a base class for DataArray and Dataset
Accessors.

"""
import warnings
import numpy as np
import rasterio
import rasterio.enums as enums
import rasterio.features as features
import xarray as xr
from xgeo.crs import XCRS
from xgeo.utils import DEFAULT_DIMS, T_DIMS, X_DIMS, Y_DIMS, Z_DIMS
from pathlib import Path

class XGeoBaseAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def init_geoparams(self):
        self._obj.attrs.update(
            resolutions=self._get_resolutions(),
            transform=self._get_transform(),
            origin=self._get_origin(),
            bounds=self._get_bounds()
        )

    def _get_resolutions(self):
        x_res = self.x_coords.diff(self.x_dim)
        y_res = self.y_coords.diff(self.y_dim)

        assert not(
            not x_res.any() or not x_res.all or
            not y_res.any() or not y_res.all()
        ), "The resolutions are inconsistent.The library isn't capable of \
            handling inconsistent resolutions."

        return (x_res.values.min(), y_res.values.min())

    def is_raster(self, value):
        return isinstance(value, xr.DataArray) and all([
            x in value.dims for x in [self.x_dim, self.y_dim]
        ])

    @property
    def resolutions(self):
        if self._obj.attrs.get('resolutions', None) is None:
            resolutions = self._get_resolutions()
            self._obj.attrs.update(resolutions=resolutions)
        return self._obj.attrs.get('resolutions')

    def _get_transform(self):
        x_res, y_res = self.resolutions
        x_origin = self.x_coords.isel({self.x_dim: 0}) - x_res / 2.0
        y_origin = self.y_coords.isel({self.y_dim: 0}) - y_res / 2.0
        transform = (x_res, 0, x_origin.values.min(),
                     0, y_res, y_origin.values.min())
        return transform

    @property
    def transform(self):
        if self._obj.attrs.get("transform") is None:
            transform = self._get_transform()
            self._obj.attrs.update(transform=transform)
        return self._obj.attrs.get("transform")

    def _recompute_coords(self, transform):
        x_res, _, x_origin, _, y_res, y_origin = transform
        x_coords = x_origin + x_res / 2.0 + np.arange(0, self.x_size) * x_res
        y_coords = y_origin + y_res / 2.0 + np.arange(0, self.y_size) * y_res

        self._obj.coords.update({
            self.x_dim: x_coords,
            self.y_dim: y_coords
        })

    @transform.setter
    def transform(self, transform):
        assert isinstance(transform, (tuple, list, np.ndarray)) and \
            len(transform) == 6, "`transform` variable should be either \
            tuple or list with 6 numbers"

        self._obj.attrs.update(transform=tuple(transform))
        self._recompute_coords(transform=tuple(transform))

    @property
    def projection(self):
        crs = self._obj.attrs.get("crs")
        if crs:
            self._obj.attrs.update(crs=XCRS.from_any(crs).to_proj4())
        return self._obj.attrs.get("crs")

    @projection.setter
    def projection(self, value):
        assert isinstance(value, (str, int, dict)), "The projection should be \
        either string, integer or dictionary"
        self._obj.attrs.update(crs=XCRS.from_any(value).to_proj4())

    def _get_origin(self):
        x_origin = {True: 'left', False: 'right'}
        y_origin = {True: 'bottom', False: 'top'}
        x_res, y_res = self.resolutions
        return "{0}_{1}".format(
            y_origin.get(y_res >= 0),
            x_origin.get(x_res >= 0)
        )

    @property
    def origin(self):
        if not self._obj.attrs.get('origin', None):
            origin = self._get_origin()
            self._obj.attrs.update(origin=origin)
        return self._obj.attrs.get('origin')

    def _set_origin(self, origin):
        yo, xo = self.origin.split('_')
        nyo, nxo = origin.split('_')
        if yo != nyo:
            self._obj = self._obj.reindex({self.y_dim: self.y_coords[::-1]})
        if xo != nxo:
            self._obj = self._obj.reindex({self.x_dim: self.x_coords[::-1]})
        self.init_geoparams()

    @origin.setter
    def origin(self, value):
        allowed_origins = [
            'top_left', 'bottom_left', 'top_right', 'bottom_right'
        ]
        if isinstance(value, str) and value in allowed_origins:
            self._set_origin(value)
            self._obj.attrs.update(origin=value)
        else:
            raise IOError("Either the value provided isn't string or doesn't \
                belong to one of {}".format(allowed_origins))

    def _get_bounds(self):
        x_res, _, x_origin, _, y_res, y_origin = self.transform
        x_end = x_origin + self.x_size * x_res
        y_end = y_origin + self.y_size * y_res
        x_options = np.array([x_origin, x_end])
        y_options = np.array([y_origin, y_end])

        return (x_options.min(), y_options.min(),
                x_options.max(), y_options.max())

    @property
    def bounds(self):
        if self._obj.attrs.get('bounds') is None:
            bounds = self._get_bounds()
            self._obj.attrs.update(bounds=bounds)
        return self._obj.attrs.get('bounds')

    @property
    def x_dim(self):
        for dim in self._obj.dims:
            if dim in X_DIMS:
                return dim
        raise AttributeError(
            "x dimension isn't understood. Valid names are {}.".format(X_DIMS)
        )

    @property
    def y_dim(self):
        for dim in self._obj.dims:
            if dim in Y_DIMS:
                return dim
        raise AttributeError(
            "y dimension isn't understood. Valid names are {}.".format(Y_DIMS)
        )
    
    @property
    def non_loc_dims(self):
        dims = set(self._obj.dims).difference([self.x_dim, self.y_dim])
        return sorted(dims)

    # @property
    # def band_dim(self):
    #     for dim in self._obj.dims:
    #         if dim in Z_DIMS:
    #             return dim
    #     raise AttributeError(
    #         "band dimension isn't understood. Valid names are {}.".format(
    #             Z_DIMS
    #         )
    #     )

    @property
    def x_coords(self):
        return self._obj.coords.get(self.x_dim)

    @property
    def y_coords(self):
        return self._obj.coords.get(self.y_dim)

    # @property
    # def band_coords(self):
    #     return self._obj.coords.get(self.band_dim)

    @property
    def x_size(self):
        return self._obj.sizes.get(self.x_dim)

    @property
    def y_size(self):
        return self._obj.sizes.get(self.y_dim)

    # @property
    # def band_size(self):
    #     return self._obj.sizes.get(self.band_dim)

    @staticmethod
    def _validate_resampling(resampling=enums.Resampling.nearest):
        """
        Validates if the resampling is valid <rasterio.enums.Resampling> or
        strings. If the resampling is the string, it fetches the
        corresponding <rasterio.enums.Resampling> object.

        Parameters
        ----------
        resampling: rasterio.warp.Resampling or str
            User provided resampling method

        Returns
        -------
        resampling: rasterio.warp.Resampling
            Validated resampling method

        """
        assert isinstance(resampling, (enums.Resampling, str)), \
            "Method can be either <rasterio.warp.Resampling> or string"

        try:
            if isinstance(resampling, enums.Resampling):
                return resampling
            else:
                return getattr(enums.Resampling, resampling)
        except AttributeError:
            raise IOError("Invalid resampling method")

    def _indices_to_bounds(self, indices):
        """
        Creates bounds (x minimum, y minimum, x maximum, y maximum) from the
        indices or coordinates in raster coordinate system.

        Parameters
        ----------
        indices: tuple/list
            Indices of the bounds (x mimimum index, y minimum index, x
            maximum index, y maximum index).

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

    def xrslice(self, indices=None, bounds=None):
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
            bounds = self._indices_to_bounds(indices)
        x_min, y_min, x_max, y_max = bounds
        obj_subset = self._obj.sel({
            self.x_dim: slice(x_min, x_max),
            self.y_dim: slice(y_max, y_min)
        })
        obj_subset.geo._init_geoparams()
        return obj_subset

    def _get_geodataframe(self, vector_file, geometry_name="geometry"):
        """
        Gets GeoDataFrame for given vector file. It carries out following
        list of tasks.
            - Sets the geometry from given geometry_name
            - Reprojects the dataframe to the CRS system of the Dataset
            - Checks whether the Dataframe is empty and whether any geometry
            lies within the bound of the dataset.
        Parameters
        ----------
        vector_file: str or geopandas.GeoDataFrame
            Vector file to be read

        geometry_name: str
            Name of the geometry in the vector file if it doesn't default to
            "geometry"

        Returns
        -------
        vector_gdf: geopandas.GeoDataFrame
            GeoDataFrame for the given vector file
        """
        try:
            import geopandas
            import shapely.geometry
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "`fiona` module is needed to use this function. Please \
                install it and try again"
            )

        # Read and reproject the vector file
        assert isinstance(vector_file, (str, Path, geopandas.GeoDataFrame)), \
            "Invalid vector_file. The `vector_file` should either be path to \
            file or geopandas.GeoDataFrame"

        if isinstance(vector_file, (str, Path)):
            vector = geopandas.read_file(vector_file)
        else:
            vector = vector_file

        # Set geometry of the geodataframe
        vector = vector.set_geometry(geometry_name)

        # If the projection system exists. Reproject the vector to the
        # projection system of the data.
        if self.projection:
            vector = vector.to_crs(XCRS.from_any(self.projection).to_dict())

        # Validate that the vector file isn't empty and at least one of the
        # geometries in the vector file is intersecting the raster bounds.
        assert not vector.empty, "Vector file doesn't contain any geometry"
        raster_bound = shapely.geometry.box(*self.bounds)
        assert any([raster_bound.intersects(feature) for feature in vector.geometry]
                   ), "No geometry in vector file are intersects the image bound"
        return vector

    def subset(self, vector_file, geometry_name="geometry", crop=False,
               extent_only=False, invert=False):
        """
        Subset the DataArray with the vector file.

        Parameters
        ----------
        vector_file: str or geopandas.GeoDataFrame
            Path to the vector file. Any vector file supported by GDAL are
            supported.

        geometry_name: str
            Column name that describes the geometries in the vector file.
            Default value is "geometry"

        crop: bool
            If True, the output DataArray bounds is approximately equal to the
            total bounds of the geometry. The default value is False

        extent_only: bool
            If True, the output DataArray consists all the data that are within
            the total bounds of the geometry. Default value is True. If
            extent_only is True, the crop is by default True.

        invert: bool
            If True, the output DataArray contains values that are only
            outside of the geometries. Default value is False. This doesn't
            have effect if extent_only is True

        Returns
        -------
        da: xarray.DataArray
            Subset DataArray
        """

        # Re-structure user input for special cases.
        # If extent_only is true, crop is always true.
        if extent_only:
            crop = True

        # Get GeoDataframe from given vector file
        vf = self._get_geodataframe(
            vector_file=vector_file,
            geometry_name=geometry_name
        )

        if crop:
            obj = self.xrslice(bounds=vf.total_bounds)

            # If extent_only the subset dataset doesn't need to be masked
            if extent_only:
                return obj
        else:
            obj = self._obj.copy()

        # Create a rasterized mask from the GeoDataframe and add as coordinate.
        mask = obj.geo.get_mask(vector_file=vf, geometry_name=geometry_name)
        mask_value = 1 if invert else 0

        return obj.where(mask != mask_value)

    def get_mask(self, vector_file, geometry_name="geometry", value_name=None):
        vector = self._get_geodataframe(
            vector_file=vector_file,
            geometry_name=geometry_name
        )
        with rasterio.Env():
            if value_name is not None:
                assert value_name in vector.columns, "`value_name` should be \
                valid name. For defaults leave it None"
                assert vector.geometry.size == vector.get(value_name).size, \
                    "Rows in `value_name` column and geometries are different"
                geom_iterator = zip(vector.geometry, vector.get(value_name))
            else:
                geom_iterator = zip(
                    vector.geometry, [1] * vector.geometry.size)

            mask = features.rasterize(
                geom_iterator,
                transform=self.transform,
                out_shape=(self.y_size, self.x_size)
            )

            return xr.DataArray(mask, dims=(self.y_dim, self.x_dim))
