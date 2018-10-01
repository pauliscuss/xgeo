#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import xarray as xr
import dask.array
import pandas as pd
from datetime import datetime
import rasterio
import rasterio.warp
from pathlib import Path
from os.path import basename, dirname, join, getsize
import glob
import os

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
        print(reproject_kwargs)

    # write data
    prefix = prefix if prefix else da.name
    for i, z in enumerate(da[zdim].values):
        fn = join(out_dir, formatter(i=i, z=z, prefix=prefix))
        _write_raster(da.sel(**{zdim: z}), fn, reproject_kwargs=reproject_kwargs, **kwargs)

def from_bin(filename, nrow, ncol, transform, dtype=np.float32, nodata=np.nan, chunks=None, 
    zstart=0, zdim='time', freq='d', epsg=4326, name=None, **kwargs):
    """read binary file into xarray dataaray with geo coordinates and metadata"""
    if epsg==4326:
        ydim, xdim = 'lat', 'lon'
    else:
        ydim, xdim = 'y', 'x'
    # make memmap
    a = memmap3d(filename, nrow, ncol, dtype=np.float32, nodata=nodata, **kwargs)
    # create xarray datdaarray
    da = from_np(a, transform, chunks=None, zstart=zstart, zdim=zdim, ydim=ydim, xdim=xdim, freq=freq)
    # set metadata
    da = utils.set_crs(da, crs=epsg)
    source = basename(filename)
    da.attrs.update({'_FillValue': np.nan, 'source': source}) # from_np sets nodata to NaN
    if name is None:
        name = source.split('.')[0]
    da.name = name
    return da

def from_mfbin(datadir, prefix, nrow, ncol, transform, ext='.bin', dtype=np.float32, nodata=np.nan, chunks=None, 
    zdim='time', freq='d', epsg=4326, name=None, **kwargs):
    """read multiple bin files into xarray dataarray with geo coordinates and metadata and concatenate along z dimension
    filename format is <datadir>/<prefix><date><ext>"""
    fns = glob.glob(join(datadir, '{}*[0-9]{}'.format(prefix, ext)))
    da_list = []
    name = name if name else basename(prefix)
    for fn in fns:
        zstart = _parse_zstart_fn(fn, prefix, ext)
        da_list.append(from_bin(fn, nrow, ncol, transform=transform, dtype=dtype, nodata=nodata, chunks=chunks, 
            zstart=zstart, zdim=zdim, freq=freq, epsg=epsg, name=name, **kwargs)
        )
    da = xr.concat(da_list, dim=zdim).sortby(zdim)
    return da


def from_np(a, transform, chunks=None, zstart=0, zdim='time', ydim='lat', xdim='lon', freq='d'):
    """numpy (memmap) array to xarray dataarray
    get dataarray coordinates and dims based on a 3D numpy array, transform and dimension names
    to set a datetime zdim, use zstart=datetime(yyyy,mm,dd)
    assume order of dimensions is (z, y, x)"""
    x = _np_2_dask(a, chunks=chunks)

    # get coords
    bounds = rasterio.transform.array_bounds(transform=transform, height=a.shape[1], width=a.shape[2])
    xmin, ymin, xmax, ymax = bounds
    dx, dy = transform[0], transform[4]
    coords = dict()
    coords[xdim] = np.arange(xmin + dx / 2., xmax, dx)
    coords[ydim] = np.arange(ymax + dy / 2., ymin, dy)
    if isinstance(zstart, datetime):
        coords[zdim] = pd.date_range(start=zstart, freq=freq, periods=a.shape[0])
    else:
        coords[zdim] = np.arange(zstart, zstart+a.shape[0])

    return xr.DataArray(x, coords=coords, dims=(zdim, ydim, xdim))

# internal functions
# output 
def _write_raster(da, path, reproject_kwargs={}, **kwargs):
    """write prepared xarray DataArray to gdal type. All properties must be set in kwargs"""
    # make data 3D for consistency
    data = da.data[None, :, :] if kwargs['count'] == 1 else da.data # make sure data has 3 dims
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
        
        # regrid
        data_out = np.zeros((kwargs['count'], kwargs['width'], kwargs['height']), dtype=kwargs['dtype'])
        rasterio.warp.reproject(
            source=data,
            destination=data,
            **reproject_kwargs
        )
        data = data_out

    # write data
    with rasterio.Env():
        with rasterio.open(path, "w", **kwargs) as dst:
            for i in range(kwargs['count']):
                if getattr(kwargs, 'tiled', None) == 'yes':
                    for _, window in dst.block_windows():
                        row_slice, col_slice = window.toslices()
                        data_wdw = np.array(data[i, row_slice, col_slice])
                        dst.write(data_wdw, i+1, window=window)
                else:
                    dst.write(np.array(data[i, :, :]), i+1) 
    if os.path.isfile(path + 'aux.xml'):
        os.unlink(os.path.isfile(path + 'aux.xml'))

def _prep_da_to_raster(da, driver, nodata, force_NtoS,
             xdim, ydim, zdim, chunks, **kwargs):   
    # check and reorder dimensions
    da, _ = utils._check_dims(da, xdim=xdim, ydim=ydim, zdim=zdim)
    
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

# input 
def memmap3d(filename, nrow, ncol, dtype=np.float32, nodata=np.nan, **kwargs):
    """Make a 3d memory map of a single binary file and convert nodata to NaNs"""
    bytes_per_record = nrow * ncol * np.nbytes[dtype]
    nrecords = int(getsize(str(filename)) / bytes_per_record)
    shape = (nrecords, nrow, ncol)

    a = np.memmap(filename=str(filename), dtype=dtype, mode="r+", shape=shape, **kwargs)
    
    if not np.isnan(nodata):
        isnodata = np.isclose(a, nodata)
        a[isnodata] = np.nan
    return a

def _np_2_dask(a, chunks=None):
    """if chucks is None, chunck along first dimension for 3d arrays"""
    if chunks is None:
        if a.ndim == 3:
            chunks = (1, a.shape[1], a.shape[2])
        else:
            chunks = a.shape
    return dask.array.from_array(a, chunks=chunks)

def _parse_zstart_fn(fn, prefix, ext):
    z0 = basename(fn)[len(prefix):-len(ext)]
    try:
        # assume yyyymmdd
        y = int(z0[:4]) if len(z0) >= 4 else None
        m = int(z0[4:6]) if len(z0) >= 6 else 1
        d = int(z0[6:8]) if len(z0) == 8 else 1
    except ValueError:
        y = None
    zstart=datetime(y,m,d) if y else 0
    return zstart