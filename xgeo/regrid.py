import numpy as np
import xarray as xr
import rasterio
import affine
from rasterio.warp import Resampling
from xgeo import utils

def clip(da, bounds, xdim='lon', ydim='lat'):
    """clip a dataarray with spatial extent"""
    xmin, ymin, xmax, ymax = bounds
    dy = np.diff(da[ydim].values)[0]
    yslice = slice(ymax, ymin) if dy < 0 else slice(ymin, ymax)
    dx = np.diff(da[xdim].values)[0]
    xslice = slice(xmax, xmin) if dx < 0 else slice(xmin, xmax)
    return da.sel({ydim:yslice, xdim:xslice})

def flipdim(da, dim='lat'):
    """flip a dataarray at <dim>. dafault along latitutde axis"""
    return da.sel(**{dim:slice(None, None, -1)})

def get_reproject_kwargs(da, resampling, like=None, dst_crs=None, src_crs=None, 
    xdim='lon', ydim='lat', reproject_kwargs={}):
    """get destination crs, transform, height and width based on <like> raster or given <dst_crs>"""
    assert like or dst_crs, "either like or dst_crs should be provided"
    resampling_methods = {e.name: e for e in Resampling}
    if isinstance(resampling, str):
        resampling = resampling_methods.get(resampling)
    if resampling is None:
        raise ValueError("Invalid resampling method. Available methods are: {}".format('; '.join(resampling_methods.keys())))
    
    # source grid definition 
    if src_crs is None:
        src_crs = utils.get_crs(da)
    src_transform = utils.get_transform(da, xdim=xdim, ydim=ydim)
    
    # destination grid definition
    if like:
        dst_crs, dst_transform, dst_height, dst_width = utils.rasterio_grid(like)
    elif dst_crs:
        src_height, src_width = utils.get_shape(da, xdim=xdim, ydim=ydim)
        bounds = rasterio.transform.array_bounds(src_height, src_width, src_transform)
        # get destination raster definition
        dst_transform, dst_width, dst_height = rasterio.warp.calculate_default_transform(
            src_crs, dst_crs, src_width, src_height, *bounds
        )
    
    # combine all reproject kwargs
    reproject_kwargs.update(
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        dst_height=dst_height, 
        dst_width=dst_width,
        resampling=resampling,
    )
    return reproject_kwargs

def reproject_rasterio(da, resampling, like=None, dst_crs=None, src_crs=None, 
    xdim='lon', ydim='lat', reproject_kwargs={}):
    """reproject xarray dataarray using rasterio warp functionality"""
    # check and reorder dimensions
    da, dims = utils._check_dims(da, xdim=xdim, ydim=ydim)
    
    # get reproject kwargs 
    reproject_kwargs = get_reproject_kwargs(da, resampling, like=like, dst_crs=dst_crs, src_crs=src_crs, 
        xdim=xdim, ydim=ydim, reproject_kwargs=reproject_kwargs)

    # prepare output grid
    dst_shape = [da[da.dims[0]].size] if len(dims) == 3 else []
    zcoords = da[da.dims[0]].values if len(dims) == 3 else None
    dst_shape = tuple(dst_shape + [reproject_kwargs['dst_height'], reproject_kwargs['dst_width']])
    dst_da = utils._prepare_dataarray(name=da.name, shape=dst_shape, transform=reproject_kwargs['dst_transform'],  
        crs=reproject_kwargs['dst_crs'], zcoords=zcoords, dims=dims, dtype=da.dtype)
    
    # warp
    rasterio.warp.reproject(
        source=da.values,
        destination=dst_da.values,
        **reproject_kwargs,
    )

    return dst_da

