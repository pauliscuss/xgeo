#!/usr/bin/env python
# -*- coding: utf-8 -*-


import xarray as xr
import rasterio
from affine import Affine
import numpy as np

class GridType(object):
    """
    Enumeration with grid types.
    """
    UNKNOWN = 0
    RGRID = 1 # equidistant regular grid
    UGRID = 2 # unstructured grid
    UCGRID = 3 # Unit-catchment grid

def get_grid_type(da, xdim='lon', ydim='lat'):
    x = da[xdim].values
    y = da[ydim].values
    ncol = x.size
    nrow = y.size
    if ncol == 1 and nrow == 1:
        grid_type = GridType.UNKNOWN
    else:
        dxs = np.diff(x)
        dx = dxs[0]
        atolx = abs(1.0e-6 * dx)
        dys = np.diff(y)
        dy = dys[0]
        atoly = abs(1.0e-6 * dy)
        if np.allclose(dxs, dx, atolx) and np.allclose(dys, dy, atoly):
            grid_type = GridType.RGRID
        else:
            grid_type = GridType.UNKNOWN
    return grid_type

def get_nodata(da):
    # read nodata
    if '_FillValue' in da.attrs.keys():
        fill = da.attrs['_FillValue']
    elif 'missing_value' in da.attrs.keys():
        fill = da.attrs['missing_value']
    else:
        fill = np.nan # assuming mask and scale applied
    return fill

def set_nodata(da, nodata):
    if nodata is None or get_nodata(da) == nodata:
        pass # do nothing
    else:
        da = da.fillna(nodata)
        da.attrs.update({'_FillValue': nodata})
    return da


def get_shape(da, xdim='lon', ydim='lat'):
    "get (nrows, ncols)"
    ncol = da[xdim].size
    nrow = da[ydim].size
    return nrow, ncol

def get_transform(da, xdim='lon', ydim='lat'):
    "get transform. only for regular (equidistant) grids"
    # based on https://gitlab.com/deltares/imod-python/blob/master/imod/util.py
    assert get_grid_type(da) == GridType.RGRID, "a transform can only be defined for a reqular grid"
    x = da[xdim].values
    y = da[ydim].values
    dx = np.diff(x)[0]
    dy = np.diff(y)[0]
    xmin = x[0] - 0.5 * dx # as xarray used midpoint coordinates
    ymax = y[0] - 0.5 * dy
    return Affine(dx, 0.0, xmin, 0.0, dy, ymax)

def get_rio_profile(da, xdim='lon', ydim='lat'):
    height, width = get_shape(da, xdim, ydim)
    try:
        crs = get_crs(da)
    except ValueError:
        crs = rasterio.crs.CRS.from_epsg(4326)
        # TODO add warning
    profile = dict(
        transform = get_transform(da, xdim, ydim),
        height = height,
        width = width,
        dtype = da.dtype,
        nodata = get_nodata(da),
        crs = crs,
        count = 1 if da.ndim == 2 else da.shape[0]
    )
    return profile

def set_crs(da, crs, crs_attr_name='crs'):
    "translates rasterio crs to cf1.6 confention attributes"
    # TODO: complete list from https://cf-pcmdi.llnl.gov/trac/wiki/Cf2CrsWkt
    crs = _check_crs(crs)
    if crs.is_epsg_code:
        epsg = int(crs.get('init').split(':')[-1])
    else:
        epsg = None
    if epsg == 4326:
        grid_mapping_name = 'latitude_longitude'
    if crs_attr_name not in [v for v in da.coords.keys()]:
        da[crs_attr_name] = xr.Variable([], None)
        # if dataset, make crs part of the coordinates. this is automatic if dataaraay
        if hasattr(da, 'set_coords'): 
            da = da.set_coords([crs_attr_name])
    crs_dict = dict(long_name = 'CRS definition',
                    grid_mapping_name = grid_mapping_name,
                    proj4 = crs.to_string())
    da[crs_attr_name].attrs = crs_dict
    return da

def get_crs(da, crs_attr_name='crs'):
    names = ['proj4', 'wkt', 'epsg']
    _crs = getattr(da, crs_attr_name, None)
    if _crs is not None:
        for n in names:
            if n in _crs.attrs.keys():
                try:
                    crs = _check_crs(_crs.attrs[n])
                    if crs:
                        break
                except ValueError:
                    pass
        if not crs:
            raise ValueError('not valid crs found of types {}'.format(', '.join(names)))
    else:
        raise ValueError('dim {} not found in dataset'.format(crs_attr_name))
    return crs

# internal
def _check_crs(crs):
    if isinstance(crs, rasterio.crs.CRS):
        pass
    elif isinstance(crs, dict):
        crs = rasterio.crs.CRS(**crs)
    elif isinstance(crs, int):
        crs = rasterio.crs.CRS.from_epsg(crs)
    elif isinstance(crs, str):
        if crs.startswith('+'):
            crs = rasterio.crs.CRS.from_string(crs)
        else:
            crs = rasterio.crs.CRS.from_wkt(crs)
    else:
        crs = None
    return crs 

def _prepare_dataarray(name, shape, transform, crs=4326, zcoords=None,
                  dims=['time', 'lat', 'lon'], dtype=np.float32):
    """
    Prepares destination DataArray for reprojection.
    """
    assert len(shape) == len(dims), "shape and dims do should have equal length"
    height, width = shape[-2:]
    ydim, xdim = dims[-2:]
    # create coords
    # from: http://xarray.pydata.org/en/stable/generated/xarray.open_rasterio.html
    x, y = (
        np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
        * transform
    )
    coords = {ydim: y[:, 0], xdim: x[0, :]}

    # add third dimension 
    if len(shape) == 3:
        if zcoords is None:
            zcoords = np.arange(dims[0])
        coords.update({dims[0]: zcoords})

    # create dataarray
    dst = xr.DataArray(
        data=np.empty(shape, dtype),
        coords=coords,
        dims=dims,
    )
    dst = set_crs(dst, crs)
    return dst

def _check_dims(da, xdim='lon', ydim='lat', zdim=None):
    assert (da.ndim >= 2) & (da.ndim <= 3), "dataarray should have 2 or 3 dimensions"
    dims = [ydim, xdim]
    if zdim:
        dims = [zdim] + dims
    elif da.ndim == 3:
        dims = [d for d in da.dims if d not in dims] + dims
    assert np.all([dim in da.dims for dim in dims]), "not all dimensions found in DataArray"
    assert da.ndim == len(dims), "DataArray has more dimension than {:d} dimensions".format(len(dims))
    da = da.transpose(*tuple(dims)) # order of dimensions is z,y,x
    return da, dims

# formatter 
def pcr_mapstack_formatter(i, prefix, **kwargs):
    prefix = prefix[:8] if len(prefix) > 8 else prefix # max 8 char
    below_thousand = i % 1000
    above_thousand = i / 1000
    mapname = str(prefix + '%0' + str(8-len(prefix)) + '.f.%03.f') % (above_thousand, below_thousand)
    return mapname

# rasterio 
def rasterio_grid(like):
    with rasterio.open(like) as template_ds:
        dst_crs = template_ds.crs
        dst_transform = template_ds.transform
        dst_height = template_ds.height
        dst_width = template_ds.width
    return dst_crs, dst_transform, dst_height, dst_width