"""
The Base Accessor that provides a base class for DataArray and Dataset
Accessors.

"""
import numpy as np
import rasterio
import rasterio.enums as enums
import rasterio.features as features
import xarray as xr
from xgeo.crs import XCRS
from xgeo.utils import X_DIMS, Y_DIMS, get_geodataframe


class XGeoBaseAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self.init_geoparams()
        if hasattr(self, "search_projection"):
            self.projection = self.search_projection()

    def init_geoparams(self):
        """Initializes the parameters related to the overall geometry of the
        xarray object. It either finds or calculates the resolution of the
        object, the affine parameters, origin of the object and the overall
        bounds of the object.
        """

        self._obj.attrs.update(
            resolutions=self._get_resolutions(),
            transform=self._get_transform(),
            origin=self._get_origin(),
            bounds=self._get_bounds()
        )

    def _get_resolutions(self):
        """Calculates the resolutions along x and y dimensions of the xarray
        object.

        Returns
        -------
        resolutions: tuple
            Resolutions of the object along the x and y dimensions (x
            resolution, y resolution).
        """
        x_res = self.x_coords.diff(self.x_dim)
        y_res = self.y_coords.diff(self.y_dim)

        assert not(
            not x_res.any() or not x_res.all or
            not y_res.any() or not y_res.all()
        ), "The resolutions are inconsistent.The library isn't capable of \
            handling inconsistent resolutions."

        return (x_res.values.min(), y_res.values.min())

    def is_raster(self, value):
        """Checks whether the Dataarray provided is a raster of not. In order
        to qualify for raster, the data array must have x and y dimensions.

        Parameters
        ----------
        value : <xarray.DataArray>
            The DataArray which has to be checked whether it is a raster or not.

        Returns
        -------
        is_raster: bool
            True if the value is a raster dataarray else False.
        """
        return isinstance(value, xr.DataArray) and all([
            x in value.dims for x in [self.x_dim, self.y_dim]
        ])

    @property
    def resolutions(self):
        """Resolution along x and y dimensions of the given xarray object.

        Returns
        -------
        resolutions: tuplse
            (x resolution, y resolution)
        """
        if self._obj.attrs.get('resolutions', None) is None:
            resolutions = self._get_resolutions()
            self._obj.attrs.update(resolutions=resolutions)
        return self._obj.attrs.get('resolutions')

    def _get_transform(self):
        """Calculates the transform from x and y coordinates.

        Returns
        -------
        transform: tuple
            Transform in affine format.
            (x resolution, 0, x origin, 0, y resolution, y origin)
        """
        x_res, y_res = self.resolutions
        x_origin = self.x_coords.isel({self.x_dim: 0}) - x_res / 2.0
        y_origin = self.y_coords.isel({self.y_dim: 0}) - y_res / 2.0
        transform = (x_res, 0, x_origin.values.min(),
                     0, y_res, y_origin.values.min())
        return transform

    @property
    def transform(self):
        """Geo transform of the object in affine format.

        Returns
        -------
        transform: tuple
            (x resolution, 0, x origin, 0, y resolution, y origin)
        """
        if self._obj.attrs.get("transform") is None:
            transform = self._get_transform()
            self._obj.attrs.update(transform=transform)
        return self._obj.attrs.get("transform")

    def _recompute_coords(self, transform):
        """Recomputes coordinate of the xarray object using the given transform.

        Parameters
        ----------
        transform : tuple
            Geotransform of the object in affine format.
            (x resolution, 0, x origin, 0, y resolution, y origin)
        """
        x_res, _, x_origin, _, y_res, y_origin = transform
        x_coords = x_origin + x_res / 2.0 + np.arange(0, self.x_size) * x_res
        y_coords = y_origin + y_res / 2.0 + np.arange(0, self.y_size) * y_res

        self._obj.coords.update({
            self.x_dim: x_coords,
            self.y_dim: y_coords
        })

    @transform.setter
    def transform(self, transform):
        """Sets the geotransform of the object and recomputes the coordinates
        according to the new transform.

        Parameters
        ----------
        transform : tuple
            Geotransform of the object in affine format.
            (x resolution, 0, x origin, 0, y resolution, y origin)
        """
        assert isinstance(transform, (tuple, list, np.ndarray)) and \
            len(transform) == 6, "`transform` variable should be either \
            tuple or list with 6 numbers"

        self._obj.attrs.update(transform=tuple(transform))
        self._recompute_coords(transform=tuple(transform))

    @property
    def projection(self):
        """Projection of the object as Proj4 string.

        Returns
        -------
        projection: str
            Projection in Proj4 string.
        """
        crs = self._obj.attrs.get("crs")
        if crs:
            self._obj.attrs.update(crs=XCRS.from_any(crs).to_proj4())
        return self._obj.attrs.get("crs")

    @projection.setter
    def projection(self, value):
        """Set the projection of the object.

        Parameters
        ----------
        value : str or int or dict
            Projection system in Proj4 Dictionary or String, CF Dictionary,
            WKT String or EPSG String or Int format.
        """
        assert isinstance(value, (str, int, dict)), "The projection should be \
        either string, integer or dictionary"
        self._obj.attrs.update(crs=XCRS.from_any(value).to_proj4())

    def _get_origin(self):
        """Computes the origin of the object

        Returns
        -------
        origin: str
            Origin of the object eg. top_left
        """
        x_origin = {True: 'left', False: 'right'}
        y_origin = {True: 'bottom', False: 'top'}
        x_res, y_res = self.resolutions
        return "{0}_{1}".format(
            y_origin.get(y_res >= 0),
            x_origin.get(x_res >= 0)
        )

    @property
    def origin(self):
        """Origin of the object

        Returns
        -------
        origin: str
            Origin of the object eg. top_left
        """
        if not self._obj.attrs.get('origin', None):
            origin = self._get_origin()
            self._obj.attrs.update(origin=origin)
        return self._obj.attrs.get('origin')

    def _set_origin(self, origin):
        """Sets the origin and reindexes the object according to the new origin.

        Parameters
        ----------
        origin : str
            Desired origin for the object.
        """
        yo, xo = self.origin.split('_')
        nyo, nxo = origin.split('_')
        if yo != nyo:
            self._obj = self._obj.reindex({self.y_dim: self.y_coords[::-1]})
        if xo != nxo:
            self._obj = self._obj.reindex({self.x_dim: self.x_coords[::-1]})
        self.init_geoparams()

    @origin.setter
    def origin(self, value):
        """Sets the origin of the object

        Parameters
        ----------
        value : str
            Desired origin for the object.

        Raises
        ------
        IOError
            If the origin isn't a string or if the origin isn't one of the
            allowed origins.
        """
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
        """Get bounds of the object.

        Returns
        -------
        bounds: tuple
            Bounds of the object (x minimum, y minimum, x maximum, y maximum)
        """
        x_res, _, x_origin, _, y_res, y_origin = self.transform
        x_end = x_origin + self.x_size * x_res
        y_end = y_origin + self.y_size * y_res
        x_options = np.array([x_origin, x_end])
        y_options = np.array([y_origin, y_end])

        return (x_options.min(), y_options.min(),
                x_options.max(), y_options.max())

    @property
    def bounds(self):
        """Bounds of the object.

        Returns
        -------
        bounds: tuple
            Bounds of the object (x minimum, y minimum, x maximum, y maximum)
        """
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

    @property
    def x_coords(self):
        return self._obj.coords.get(self.x_dim)

    @property
    def y_coords(self):
        return self._obj.coords.get(self.y_dim)

    @property
    def x_size(self):
        return self._obj.sizes.get(self.x_dim)

    @property
    def y_size(self):
        return self._obj.sizes.get(self.y_dim)

    @staticmethod
    def _validate_resampling(resampling=enums.Resampling.nearest):
        """
        Validates if the resampling is valid <rasterio.enums.Resampling> or
        strings. If the resampling is the string, it fetches the
        corresponding <rasterio.enums.Resampling> object.

        Parameters
        ----------
        resampling: rasterio.warp.Resampling or strout_ds.attrs.update(**self._obj.attrs)

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
        obj_subset.geo.init_geoparams()
        return obj_subset

    def subset(self, vector_file, geometry_field="geometry", crop=False,
               extent_only=False, invert=False):
        """
        Subset the object with the vector file.

        Parameters
        ----------
        vector_file: str or geopandas.GeoDataFrame
            Path to the vector file. Any vector file supported by GDAL are
            supported.

        geometry_field: str
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
        da: xarray.DataArray or xarray.Dataset
            Subset DataArray or Dataset
        """

        # Re-structure user input for special cases.
        # If extent_only is true, crop is always true.
        if extent_only:
            crop = True

        # Get GeoDataframe from given vector file
        vector = get_geodataframe(
            vector_file=vector_file,
            projection=self.projection,
            geometry_field=geometry_field
        )

        if crop:
            obj = self.xrslice(bounds=vector.bounds)

            # If extent_only the subset dataset doesn't need to be masked
            if extent_only:
                return obj
        else:
            obj = self._obj.copy()

        # Create a rasterized mask from the GeoDataframe and add as coordinate.
        mask = obj.geo.get_mask(vector=vector, geometry_field=geometry_field)
        mask_value = 1 if invert else 0

        return obj.where(mask != mask_value)

    def get_mask(self, vector, geometry_field="geometry", label_field=None):
        """Creates a mask based on the given vector file

        Parameters
        ----------
        vector_file : str or Path
            Vector file from which a mask has to be created
        geometry_field : str, optional
            Name of the geometry in vector file, by default "geometry"
        label_field : str, optional
            Name of the value column which should be used to populate the
            masked regions, by default None

        Returns
        -------
        mask: xarray.DataArray
            Mask
        """
        with rasterio.Env():
            if label_field is not None:
                assert label_field in vector.columns, "`label_field` should be \
                   valid name. For defaults leave it None"
                geom_iterator = zip(
                    vector[geometry_field], vector[label_field])
            else:
                geom_iterator = zip(
                    vector[geometry_field],
                    [1]*len(vector[geometry_field])
                )

            mask = features.rasterize(
                geom_iterator,
                transform=self.transform,
                out_shape=(self.y_size, self.x_size)
            )

            return xr.DataArray(mask, dims=(self.y_dim, self.x_dim))

    def as_2d(self):
        return self._obj.stack(
            pixel=(self.x_dim, self.y_dim),
            pixel_data=self.non_loc_dims
        )
