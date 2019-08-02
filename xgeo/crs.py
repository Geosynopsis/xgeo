from copy import deepcopy
from rasterio.crs import CRS
from rasterio.errors import CRSError

CF2PROJ4_ATTRIBUTES = dict(
    false_easting="x_0",
    false_northing="y_0",
    scale_factor_at_projection_origin='k_0',
    scale_factor_at_central_meridian='k_0',
    longitude_of_central_meridian='lon_0',
    longitude_of_projection_origin='lon_0',
    latitude_of_projection_origin='lat_0',
    straight_vertical_longitude_from_pole='lon_0'
)

CF2PROJ4_GM_NAME = dict(
    albers_conical_equal_area="aea",
    azimuthal_equidistant="aeqd",
    lambert_azimuthal_equal_area="laea",
    lambert_conformal_conic="lcc",
    lambert_cylindrical_equal_area="cea",
    mercator="merc",
    orthographic="ortho",
    polar_stereographic="stere",
    transverse_mercator="tmerc"
)


class XCRS(CRS):
    @classmethod
    def from_cf(cls, cf_dict: dict):
        """
        Makes CRS from Climate and Forecast Convention grid_mapping

        Parameters
        ----------
        cf_dict: dict
            CF grid_mapping in dictionary format

        Returns
        -------
        CRS: XCRS
        """
        cf_dict = deepcopy(cf_dict)
        # According to the CF convention, optional grid mapping attribute
        # "crs_wkt" or "spatial_ref" (in earlier version of convention) could
        # be used to specify coordinate system properties in WKT format.
        # Therefore, if any of those attribute are present, we can directly use
        # them to get our projection system.
        crs_wkt = cf_dict.get("crs_wkt", cf_dict.get("spatial_ref"))
        if crs_wkt is not None:
            return cls.from_wkt(crs_wkt)

        # If the projection system isn't present in the wkt optional parameter,
        # we use the mapping parameters to convert the projection system
        # defined by CF convention to the Proj4 using the mapping scheme
        # defined by
        # https://github.com/cf-convention/cf-conventions/wiki/Mapping-from-CF-Grid-Mapping-Attributes-to-CRS-WKT-Elements

        grid_mapping = cf_dict.pop('grid_mapping_name')
        proj4_dict = dict(proj=CF2PROJ4_GM_NAME.get(grid_mapping))
        for param_key, param_value in cf_dict.items():
            proj4_dict[CF2PROJ4_ATTRIBUTES.get(param_key)] = param_value
        return cls.from_dict(proj4_dict)

    @classmethod
    def from_any(cls, proj: dict or str or int):
        """
        Makes CRS from any supported system

        Parameters
        ----------
        proj: dict or str or int
            Projection in PROJ, EPSG, WKT or CF.

        Returns
        -------
        CRS: XCRS
        """
        if isinstance(proj, dict):
            try:
                return cls.from_dict(proj)
            except CRSError:
                return cls.from_cf(proj)
            except Exception:
                raise CRSError("{} is neither a PROJ4 or CF dict".format(proj))
        elif isinstance(proj, str):
            for method in ['from_string', 'from_proj4', 'from_wkt']:
                try:
                    return getattr(cls, method)(proj)
                except CRSError:
                    pass
            raise CRSError(
                "{} is neither a EPSG, PROJ4 or WKT string".format(proj))
        elif isinstance(proj, int):
            try:
                return cls.from_epsg(proj)
            except CRSError:
                raise CRSError("{} is not valid EPSG code".format(proj))
        elif isinstance(proj, cls):
            return proj
        else:
            raise IOError(
                "{} is not valid data type. Only dictionary, string or integer are supported".format(proj))
