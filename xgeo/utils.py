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

def set_crs(ds, crs, crs_attr_name='crs'):
    "translates rasterio crs to cf1.6 confention attributes"
    # TODO: complete list from https://cf-pcmdi.llnl.gov/trac/wiki/Cf2CrsWkt
    if crs.is_epsg_code:
        epsg = int(crs.get('init').split(':')[-1])
    else:
        epsg = None
    if epsg == 4326:
        grid_mapping_name = 'latitude_longitude'
        
    crs_dict = dict(long_name = 'CRS definition',
                    grid_mapping_name = grid_mapping_name,
                    proj4 = crs.to_string())
    ds[crs_attr_name] = crs_dict

def get_crs(ds, crs_attr_name='crs'):
    crs_dict = ds[crs_attr_name].attrs
    proj4 = getattr(crs_dict, 'proj4', None)
    if proj4:
        crs = rasterio.crs.CRS.from_string(proj4)
    else:
        raise ValueError('no proj4 string found in dataset')
    return crs

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