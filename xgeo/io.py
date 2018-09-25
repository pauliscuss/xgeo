#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import xarray as xr
import rasterio
import rasterio.warp
from pathlib import Path
from os.path import basename, dirname, join

from xgeo import utils, regrid

def to_raster(da, fn, driver=None, nodata=np.nan, force_NtoS=True,
    xdim='lon', ydim='lat', zdim=None, chunks=None, reproject_kwargs={}, **kwargs):
    """xarray DataArray to gdal raster"""
    if driver is None:
        driver = _driver_from_ext(fn)
    
    # prepare data
    da, kwargs = _prep_da_to_raster(da, 
        driver=driver, nodata=nodata, force_NtoS=force_NtoS,
        xdim=xdim, ydim=ydim, zdim=zdim, chunks=chunks, **kwargs)

    # reproject
    if len(reproject_kwargs) > 0:
        like = reproject_kwargs.get('like')
        dst_crs = reproject_kwargs.get('dst_crs')
        src_crs = kwargs['crs']
        if like or (dst_crs and dst_crs != src_crs):
            reproject_kwargs.update(src_crs=src_crs, xdim=xdim, ydim=ydim)
            reproject_kwargs = regrid.get_reproject_kwargs(da, **reproject_kwargs)
    
    # write data 
    _write_raster(da, fn, reproject_kwargs=reproject_kwargs, **kwargs)
    
def to_mapstack(da, out_dir, formatter=utils.pcr_mapstack_formatter, prefix=None, 
    driver=None, nodata=np.nan, force_NtoS=True,
    xdim='lon', ydim='lat', zdim='time', chunks=None, 
    reproject_kwargs={}, **kwargs):
    """xarray DataArray to gdal type mapstack"""
    # get driver 
    if driver is None:
        fn_tmp = join(out_dir, formatter(i=0, z=da[zdim].values[0], prefix=prefix))
        driver = _driver_from_ext(fn_tmp)
    
    # prepare data
    da, kwargs = _prep_da_to_raster(da, 
        driver=driver, nodata=nodata, force_NtoS=force_NtoS,
        xdim=xdim, ydim=ydim, zdim=zdim, chunks=chunks, **kwargs)
    kwargs['count'] = 1

    # reproject
    if len(reproject_kwargs) > 0:
        like = reproject_kwargs.get('like')
        dst_crs = reproject_kwargs.get('dst_crs')
        src_crs = kwargs['crs']
        if like or (dst_crs and dst_crs != src_crs):
            reproject_kwargs.update(src_crs=src_crs, xdim=xdim, ydim=ydim)
            reproject_kwargs = regrid.get_reproject_kwargs(da, **reproject_kwargs)

    # write data
    prefix = prefix if prefix else da.name
    for i, z in enumerate(da[zdim].values):
        fn = join(out_dir, formatter(i=i, z=z, prefix=prefix))
        _write_raster(da, fn, **kwargs)


# internal functions

def _write_raster(da, path, reproject_kwargs={}, **kwargs):
    """write prepared xarray DataArray to gdal type. All properties must be set in kwargs"""
    # update creation options
    reproject = len(reproject_kwargs.keys()) > 0
    if reproject:
        kwargs.update({
            'crs': reproject_kwargs['dst_crs'],
            'transform': reproject_kwargs['dst_transform'],
            'width': reproject_kwargs['dst_width'],
            'height': reproject_kwargs['dst_height']
        })

        # Adjust block size if necessary.
        if ('blockxsize' in kwargs and
                kwargs['width'] < kwargs['blockxsize']):
            del kwargs['blockxsize']
        if ('blockysize' in kwargs and
                kwargs['height'] < kwargs['blockysize']):
            del kwargs['blockysize']

    # write data
    with rasterio.Env():
        with rasterio.open(path, "w", **kwargs) as dst:
            data = da.data if kwargs['count'] == 1 else da.data[None, :, :] # make sure data has 3 dims
            if reproject:
                rasterio.warp.reproject(
                        source=data,
                        destination=rasterio.band(dst, list(range(kwargs['count']))),
                        **reproject_kwargs
                    )
            else:
                for i in range(kwargs['count']):
                    if getattr(kwargs, 'tiled', None) == 'yes':
                        for _, window in dst.block_windows():
                            row_slice, col_slice = window.toslices()
                            data_wdw = np.array(data[i, row_slice, col_slice])
                            dst.write(data_wdw, i+1, window=window)
                    else:
                        dst.write(np.array(data[i, :, :]), i+1) 

def _prep_da_to_raster(da, driver, nodata, force_NtoS,
             xdim, ydim, zdim, chunks, **kwargs):   
    # check dimensions
    dims = _check_dims(da, xdim=xdim, ydim=ydim, zdim=zdim)
    da = da.transpose(*tuple(dims)) # order of dimensions is z,y,x
    
    # make sure orientation of raster is North to South
    dy = np.diff(da[ydim].values)[0]
    if force_NtoS and dy > 0:
        da = regrid.flipdim(da, dim=ydim)
    
    # set nodata
    da = utils.set_nodata(da, nodata)
    
    # get transform, dimensions, dtype, nodata
    kwargs.update(**utils.get_rio_profile(da, xdim=xdim, ydim=ydim))
    kwargs.update(driver = driver) # set driver

    # set PCR value scale property:
    if driver == 'PCRaster':
        # see https://www.gdal.org/frmt_various.html#PCRaster 
        pcr_vs = {'b': 'VS_BOOLEAN', 'f': 'VS_SCALAR', 'i': 'VS_NOMINAL'}
        kwargs.update(PCRASTER_VALUESCALE=pcr_vs.get(kwargs['dtype'].kind))

    # make chunks for tiled writing
    if chunks and driver == 'GTiff':
        da = da.chunk(chunks)
        kwargs.update(blockxsize=int(chunks[xdim]), blockysize=int(chunks[ydim]), tiled='yes')

    return da, kwargs

def _check_dims(da, xdim='lon', ydim='lat', zdim=None):
    dims = [ydim, xdim]
    if zdim:
        dims = [zdim] + dims
    assert np.all([dim in da.dims for dim in dims]), "not all dimensions found in DataArray"
    assert da.ndim == len(dims), "DataArray has more dimension than {:d} dimensions".format(len(dims))
    return dims

def _driver_from_ext(fn):
    assert isinstance(fn, str), "file name fn should be of string type"
    path = Path(fn)
    ext = path.suffix.lower()
    if ext in (".tif", ".tiff"):
        driver = "GTiff"
    elif ext == ".asc":
        driver = "AAIGrid"
    elif ext == ".map":
        driver = "PCRaster"
    else:
        raise ValueError("Unknown extension {}, specifiy driver".format(ext))
    return driver

