import numpy as np
import xarray as xr
import rasterio
import affine
from rasterio.warp import Resampling
from xgeo import utils

def clip(da, xlim, ylim, xdim='lon', ydim='lat'):
    return da.sel({ydim:slice(ylim[0], ylim[1]), xdim:slice(xlim[0], xlim[1])})

def flipdim(da, dim='lat'):
    "flip axis along dim. dafault along latitutde y-axis"
    return da.sel(**{dim:slice(None, None, -1)})

def get_reproject_kwargs(da, resampling, like=None, dst_crs=None, src_crs=None, xdim='lon', ydim='lat', reproject_kwargs={}):
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


    # # prepare output grid
    # dst_da = _prepare_grid(dst_transform, dst_width, dst_height, xdim=xdim, ydim=ydim, dtype=da.dtype)
    
    # # warp
    # rasterio.warp.reproject(
    #     source=da.values,
    #     destination=dst_da.values,
    #     **reproject_kwargs,
    # )

    # return dst_da

def _prepare_grid(dst_transform, dst_width, dst_height, xdim='lon', ydim='lat', dtype=np.float32):
    """
    Prepares destination DataArray for reprojection.
    """
    # from: http://xarray.pydata.org/en/stable/generated/xarray.open_rasterio.html
    x, y = (
        np.meshgrid(np.arange(dst_width) + 0.5, np.arange(dst_height) + 0.5)
        * dst_transform
    )
    dst = xr.DataArray(
        data=np.empty((dst_height, dst_width), dtype),
        coords={ydim: y[:, 0], xdim: x[0, :]},
        dims=(ydim, xdim),
    )
    return dst